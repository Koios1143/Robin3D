#!/usr/bin/env python3
"""Mask3D proposals (``mask3d_infer.py``) + scene packages (``prepare_scenes.py``) -> Robin3D inputs.

Writes, keyed ``f"{alias}_{i:02}"`` over the 150 proposals in score order, the five files
Robin3D's ``val_file_dict`` takes (``<prefix>_*.pt`` in ``--out_dir``):

* ``mask3d_feat_pos`` / ``mask3d_feats``: the proposal's Mask3D query embedding / feature.
* ``uni3d_feats``: Uni3D-g over the proposal's points, as Chat-Scene's / Robin3D's
  ``extract_3d_feat`` (centred, max-radius scaled, colour / 255; encoder in ``uni3d.py``).
* ``videofeats``: DINOv2-giant multi-view features, the rule of Chat-Scene's
  ``scannet_videofeat.py`` (ZzZZCHS/openscene; Robin3D swaps dinov2-large for -giant, hence
  1536-d): per frame, project the scene points (OpenScene ``PointCloudToImageMapper``,
  depth-checked, 10 px border cut), take the 2D convex hull of the proposal's visible
  points, skip it if its area < one 16x16-grid cell, average the DINOv2 patch tokens in
  the hull's bounding box, then average over frames weighted by hull area. Proposals seen
  in no frame get no entry (Robin3D then feeds zeros), as in the released files.
  ``--upright``: 3RScan / MultiScan frames are often sideways (portrait capture); each frame
  is rotated by a multiple of 90 degrees so the world's up points up in the image before
  DINOv2, and the projections are rotated with it. ScanNet frames are already upright.
* ``attributes``: ``{"objects": Mask3D ScanNet200 class names, "locs": [150, 6] box
  centre + size}`` over the proposal points (``prepare_scannet_mask3d_attributes.py``), and
  ``gt_attributes`` ``{"ids": GT object ids, "locs": their boxes}`` in the same frame, which
  ``extra_eval/score.py`` uses to map a grounded proposal to a Beacon3D object.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class PointCloudToImageMapper:
    """OpenScene ``fusion_util.PointCloudToImageMapper`` (Apache-2.0), per-frame intrinsics."""

    def __init__(self, image_dim, visibility_threshold=0.25, cut_bound=0):
        self.image_dim, self.vis_thres, self.cut_bound = image_dim, visibility_threshold, cut_bound

    def compute_mapping(self, camera_to_world, coords, depth, intrinsic):
        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        p = np.matmul(np.linalg.inv(camera_to_world), coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int)
        inside = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
            * (pi[0] < self.image_dim[0] - self.cut_bound) * (pi[1] < self.image_dim[1] - self.cut_bound)
        depth_cur = depth[pi[1][inside], pi[0][inside]]
        inside[inside] = np.abs(depth_cur - p[2][inside]) <= self.vis_thres * depth_cur
        mapping[0][inside] = pi[1][inside]
        mapping[1][inside] = pi[0][inside]
        mapping[2][inside] = 1
        return mapping.T


def upright_k(pose):
    """Number of 90-degree CCW turns (np.rot90) that make world +z point up in the image."""
    up = pose[:3, :3].T @ np.array([0.0, 0.0, 1.0])  # world up in OpenCV camera axes
    x, y = up[0], up[1]
    if abs(y) >= abs(x):
        return 0 if y < 0 else 2
    return 1 if x > 0 else 3


def rot_rc(r, c, k, H, W):
    """Pixel (row, col) of an H x W image after np.rot90(img, k)."""
    if k == 0:
        return r, c
    if k == 1:
        return W - 1 - c, r
    if k == 2:
        return H - 1 - r, W - 1 - c
    return c, H - 1 - r


@torch.no_grad()
def dinov2_tokens(images, processor, model, device, bz=32):
    out = []
    for i in range(0, len(images), bz):
        inputs = processor(images=images[i:i + bz], return_tensors="pt").to(device)
        h = model(**inputs).last_hidden_state[:, 1:].float().cpu()
        out.append(h.reshape(-1, 16, 16, h.shape[-1]))
    return torch.cat(out, 0)


def videofeats(pdir, coords, masks, processor, model, device, upright):
    import cv2
    fdir = pdir / "frames"
    ids = sorted(int(p.name[:-4]) for p in fdir.glob("*.jpg"))
    frames = []
    for i in ids:
        pose = np.loadtxt(fdir / f"{i}.pose.txt")
        if not np.isfinite(pose).all():
            continue
        k = upright_k(pose) if upright else 0
        img = Image.open(fdir / f"{i}.jpg").convert("RGB")
        frames.append((i, pose, k, np.rot90(np.asarray(img), k).copy()))
    tokens = dinov2_tokens([Image.fromarray(f[3]) for f in frames], processor, model, device)
    inst_feats = {}
    for (i, pose, k, img), tok in zip(frames, tokens):
        depth = cv2.imread(str(fdir / f"{i}.depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
        K = np.loadtxt(fdir / f"{i}.K.txt")
        H, W = depth.shape
        mapper = PointCloudToImageMapper((W, H), 0.25, 10)
        mapping = mapper.compute_mapping(pose, coords, depth, K)
        vis = mapping[:, 2].astype(bool)
        if not vis.any():
            continue
        rr, cc = rot_rc(mapping[:, 0], mapping[:, 1], k, H, W)
        Hr, Wr = img.shape[:2]
        dH, dW = Hr // 16, Wr // 16
        for j, m in enumerate(masks):
            sel = m[vis[m]]
            pts = np.stack([rr[sel], cc[sel]], 1)
            if len(pts) < 3:
                continue
            try:
                hull = ConvexHull(pts)
            except Exception:
                continue
            if hull.volume < dH * dW:
                continue
            hv = pts[hull.vertices]
            x0, y0 = hv.min(0)
            x1, y1 = hv.max(0)
            crop = tok[(x0 // dH):((x1 + dH - 1) // dH), (y0 // dW):((y1 + dW - 1) // dW)]
            inst_feats.setdefault(j, []).append((hull.volume, crop.flatten(0, 1).mean(0)))
    out = {}
    for j, lst in inst_feats.items():
        tot = sum(w for w, _ in lst)
        out[j] = sum((w / tot) * f for w, f in lst).float()
    return out, len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packages", required=True, help="prepare_scenes.py <out_dir>/<domain>")
    ap.add_argument("--mask3d", required=True, help="mask3d_infer.py --out_dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True, help="file prefix, e.g. 3rscan")
    ap.add_argument("--uni3d_ckpt", required=True)
    ap.add_argument("--dinov2", required=True, help="facebook/dinov2-giant (HF id or local dir)")
    ap.add_argument("--upright", action="store_true")
    ap.add_argument("--scenes", default="", help="comma list of aliases (default: all packages)")
    args = ap.parse_args()
    from transformers import AutoImageProcessor, AutoModel
    from uni3d import encode_object, load_uni3d
    device = "cuda"
    torch.manual_seed(1457)
    pk = Path(args.packages)
    aliases = args.scenes.split(",") if args.scenes else sorted(json.load(open(pk / "scene_map.json")))
    processor = AutoImageProcessor.from_pretrained(args.dinov2)
    dino = AutoModel.from_pretrained(args.dinov2).to(device).eval()
    uni3d = load_uni3d(args.uni3d_ckpt, device)
    os.makedirs(args.out_dir, exist_ok=True)
    keys = ["mask3d_feats", "mask3d_feat_pos", "uni3d_feats", "videofeats"]
    out = {k: {} for k in keys}
    attrs, gt_attrs = {}, {}
    for alias in aliases:
        pts = np.load(pk / alias / "points.npz")
        coords, colors = pts["coords"].astype(np.float32), pts["colors"].astype(np.float32)
        m3d = torch.load(Path(args.mask3d) / f"{alias}.pt", weights_only=False)
        masks = m3d["masks"]
        locs = []
        for j, m in enumerate(masks):
            key = f"{alias}_{j:02}"
            out["mask3d_feats"][key] = m3d["feat"][j].float()
            out["mask3d_feat_pos"][key] = m3d["pos"][j].float()
            if len(m) == 0:
                locs.append(np.zeros(6, np.float32))
                continue
            pc = coords[m]
            locs.append(np.concatenate([(pc.max(0) + pc.min(0)) / 2, pc.max(0) - pc.min(0)]).astype(np.float32))
            out["uni3d_feats"][key] = encode_object(uni3d, pc, colors[m], device)
        vf, nf = videofeats(pk / alias, coords.astype(np.float64), masks, processor, dino, device, args.upright)
        for j, f in vf.items():
            out["videofeats"][f"{alias}_{j:02}"] = f
        attrs[alias] = {"objects": [str(c) for c in m3d["classes"]], "locs": torch.from_numpy(np.stack(locs))}
        inst = pts["inst"]
        gt_ids = [int(i) for i in np.unique(inst) if i > 0]  # 0 / -1: unannotated
        gt_attrs[alias] = {"ids": gt_ids, "locs": torch.from_numpy(np.stack(
            [np.concatenate([(coords[inst == i].max(0) + coords[inst == i].min(0)) / 2,
                             coords[inst == i].max(0) - coords[inst == i].min(0)]) for i in gt_ids]
        ).astype(np.float32)) if gt_ids else torch.zeros(0, 6)}
        print(f"{alias}: {len(masks)} proposals, {len(vf)} with 2D features from {nf} frames", flush=True)
    for k in keys:
        torch.save(out[k], os.path.join(args.out_dir, f"{args.prefix}_{k}.pt"))
    torch.save(attrs, os.path.join(args.out_dir, f"{args.prefix}_attributes.pt"))
    torch.save(gt_attrs, os.path.join(args.out_dir, f"{args.prefix}_gt_attributes.pt"))


if __name__ == "__main__":
    main()
