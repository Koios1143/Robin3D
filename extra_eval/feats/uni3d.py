"""Uni3D-g point encoder, as Chat-Scene's / Robin3D's ``extract_3d_feat`` (ZzZZCHS/Uni3D fork).

Copied from Chat-Scene's ``beacon3d/extract_uni3d_gt.py`` in this project (validated there
against Chat-Scene's official ScanNet Uni3D features): the point encoder vendored from
ZzZZCHS/Uni3D ``models/point_encoder.py`` with a pure-torch farthest point sampling in place
of ``pointnet2_ops`` (same greedy rule, seeded at point 0), and ``encode_object`` = every
point of the object, centred, max-radius scaled, colour / 255, duplicated to >= 64 points.
"""

import numpy as np
import timm
import torch
import torch.nn as nn


def fps_torch(xyz: torch.Tensor, n: int) -> torch.Tensor:
    """Farthest point sampling, [B, N, 3] -> [B, n, 3]; greedy from index 0 like pointnet2."""
    b, npts, _ = xyz.shape
    idx = torch.zeros(b, n, dtype=torch.long, device=xyz.device)
    dist = torch.full((b, npts), float("inf"), device=xyz.device)
    far = torch.zeros(b, dtype=torch.long, device=xyz.device)
    ar = torch.arange(b, device=xyz.device)
    for i in range(n):
        idx[:, i] = far
        centroid = xyz[ar, far].unsqueeze(1)
        d = ((xyz - centroid) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return xyz[ar.unsqueeze(1), idx]


def knn_point(nsample, xyz, new_xyz):
    dist = -2 * torch.matmul(new_xyz, xyz.permute(0, 2, 1))
    dist += torch.sum(new_xyz**2, -1).unsqueeze(-1)
    dist += torch.sum(xyz**2, -1).unsqueeze(1)
    return torch.topk(dist, nsample, dim=-1, largest=False, sorted=False)[1]


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group, self.group_size = num_group, group_size

    def forward(self, xyz, color):
        b, n, _ = xyz.shape
        center = fps_torch(xyz, self.num_group)
        idx = knn_point(self.group_size, xyz, center)
        idx = (idx + torch.arange(0, b, device=xyz.device).view(-1, 1, 1) * n).view(-1)
        nb = xyz.view(b * n, -1)[idx].view(b, self.num_group, self.group_size, 3)
        nc = color.view(b * n, -1)[idx].view(b, self.num_group, self.group_size, 3)
        nb = nb - center.unsqueeze(2)
        return nb, center, torch.cat((nb, nc), dim=-1)


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        pg = point_groups.reshape(bs * g, n, 6)
        feature = self.first_conv(pg.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        return torch.max(feature, dim=2, keepdim=False)[0].reshape(bs, g, self.encoder_channel)


class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, pc_feat_dim=1408, embed_dim=1024, group_size=64,
                 num_group=512, pc_encoder_dim=512):
        super().__init__()
        self.group_divider = Group(num_group, group_size)
        self.encoder = Encoder(pc_encoder_dim)
        self.encoder2trans = nn.Linear(pc_encoder_dim, pc_feat_dim)
        self.trans2embed = nn.Linear(pc_feat_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, pc_feat_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, pc_feat_dim))
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, pc_feat_dim))
        self.visual = point_transformer

    def forward(self, pts, colors):
        _, center, features = self.group_divider(pts, colors)
        tokens = self.encoder2trans(self.encoder(features))
        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(tokens.size(0), -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1) + torch.cat((cls_pos, self.pos_embed(center)), dim=1)
        x = self.visual.pos_drop(x)
        for blk in self.visual.blocks:
            x = blk(x)
        x = self.visual.norm(x[:, 0, :])
        x = self.visual.fc_norm(x)
        return self.trans2embed(x)


class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc):
        return self.point_encoder(pc[:, :, :3].contiguous(), pc[:, :, 3:].contiguous())


def load_uni3d(ckpt_path, device):
    vit = timm.create_model("eva_giant_patch14_560.m30m_ft_in22k_in1k", pretrained=False, drop_path_rate=0.0)
    model = Uni3D(PointcloudEncoder(vit))
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["module"] if "module" in ck else ck
    if next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("point_encoder.visual.")]
    if missing or unexpected:
        raise RuntimeError(f"uni3d checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    return model.to(device).eval()


@torch.no_grad()
def encode_object(model, coords, colors, device):
    pc = coords - coords.mean(0)
    pc = pc / np.max(np.sqrt(np.sum(pc**2, axis=1)))
    x = torch.cat([torch.from_numpy(pc), torch.from_numpy(colors / 255.0)], dim=-1).float().unsqueeze(0).to(device)
    while x.shape[1] < 64:
        x = torch.cat([x, x], dim=1)
    return model.encode_pc(x).squeeze(0).float().cpu()
