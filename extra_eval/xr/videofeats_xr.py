#!/usr/bin/env python3
"""Robin3D's 2D object features (DINOv2-giant multi-view) for Mask3D proposals of HM3D regions.

SceneVerse's HM3D release has no images, so RGB-D views are rendered from the HM3D v0.2 glb
(textured mesh, lighting baked into the textures -> flat shading), with the camera grid of the
Chat-Scene / GPT4Scene XR pipelines (``xr_eval/render_videofeats.py`` there; region points =
SceneVerse points + the offset in ``data/region_offsets.json``):
* 640 x 480, ScanNet's intrinsics (fx = fy = 577.87);
* camera 1.5 m above the region floor, 0.75 m grid, 8 yaws, 20 deg downward pitch; positions
  >= 0.25 m from the region's points, over floor, mostly open views; at most 12 positions.
The views are fused with Robin3D's rule for its released ScanNet features
(``extra_eval/feats/build_scene_feats.py: videofeats``: OpenScene projection with depth test
and 10 px border cut, convex hull of the proposal's visible points, skipped below one 16x16
grid cell, DINOv2-giant patch tokens averaged over the hull's bounding box, area-weighted mean
over views).

Only the proposals some XR scene selects (``mask3d_xr.py scenes`` -> ``scenes.pt``) are fused.
Output ``<out_dir>/videofeats.<shard>.pt`` = {"<scan>_<sub>": {rank: float32[1536]}} (proposals
seen in no view get no entry; Robin3D then feeds zeros, as for its released files).
"""

import argparse
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import ConvexHull, cKDTree  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "feats"))
from build_scene_feats import PointCloudToImageMapper, dinov2_tokens  # noqa: E402

W, H, F = 640, 480, 577.87
K = np.array([[F, 0, (W - 1) / 2], [0, F, (H - 1) / 2], [0, 0, 1]])
CAM_H, GRID_STEP, CLEAR, MAX_POS = 1.5, 0.75, 0.25, 12
YAWS = np.deg2rad(np.arange(0, 360, 45))
PITCH = np.deg2rad(-20)
CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def look_c2w(pos, yaw, pitch):
    """OpenCV camera-to-world (x right, y down, z forward), world z up."""
    z = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, pos
    return c2w


def camera_positions(pts, render):
    """render(c2w) -> (color, depth). Returns [(position, [(c2w, color, depth) x 8])]
    (the Chat-Scene fork's ``xr_eval/render_videofeats.py``)."""
    zf, z1 = np.percentile(pts[:, 2], 0.5), pts[:, 2].max()
    zc = min(zf + CAM_H, z1 - 0.3)
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    xs = np.arange(lo[0] + GRID_STEP / 2, hi[0], GRID_STEP)
    ys = np.arange(lo[1] + GRID_STEP / 2, hi[1], GRID_STEP)
    cand = np.array([(x, y, zc) for x in xs for y in ys]).reshape(-1, 3)
    if len(cand):
        cand = cand[cKDTree(pts).query(cand)[0] >= CLEAR]
    scored = []
    for pos in cand:
        down = render(look_c2w(pos, 0.0, np.deg2rad(-89.9)))[1]
        d = down[H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20]
        d = d[d > 0]
        if not len(d) or not 1.0 <= np.median(d) <= 2.0:
            continue
        views = [(c2w, *render(c2w)) for c2w in (look_c2w(pos, yaw, PITCH) for yaw in YAWS)]
        open_ = sum(bool((dep > 0).any()) and np.median(dep[dep > 0]) >= 0.6 for _, _, dep in views)
        scored.append((open_, pos, views))
    best = max((o for o, _, _ in scored), default=0)
    good = [(pos, views) for o, pos, views in scored if o >= min(5, best) and o > 0]
    if not good:  # fall back to the region centre
        pos = np.r_[pts[:, :2].mean(0), zc]
        return [(pos, [(c2w, *render(c2w)) for c2w in (look_c2w(pos, yaw, PITCH) for yaw in YAWS)])]
    P = np.array([g[0] for g in good])
    sel = [int(np.argmin(np.linalg.norm(P[:, :2] - pts[:, :2].mean(0), axis=1)))]
    dist = np.linalg.norm(P - P[sel[0]], axis=1)
    while len(sel) < min(MAX_POS, len(P)):
        j = int(np.argmax(dist))
        sel.append(j)
        dist = np.minimum(dist, np.linalg.norm(P - P[j], axis=1))
    return [good[j] for j in sel]


def fuse(views, tokens, pts, masks):
    """Robin3D's ``videofeats`` over rendered views; masks: {rank: point ids}."""
    acc = {}
    for (c2w, _color, depth), tok in zip(views, tokens):
        mapping = PointCloudToImageMapper((W, H), 0.25, 10).compute_mapping(c2w, pts, depth, K)
        vis = mapping[:, 2].astype(bool)
        if not vis.any():
            continue
        rr, cc = mapping[:, 0], mapping[:, 1]
        dH, dW = H // 16, W // 16
        for j, m in masks.items():
            sel = m[vis[m]]
            p2 = np.stack([rr[sel], cc[sel]], 1)
            if len(p2) < 3:
                continue
            try:
                hull = ConvexHull(p2)
            except Exception:
                continue
            if hull.volume < dH * dW:
                continue
            hv = p2[hull.vertices]
            x0, y0 = hv.min(0)
            x1, y1 = hv.max(0)
            crop = tok[(x0 // dH):((x1 + dH - 1) // dH), (y0 // dW):((y1 + dW - 1) // dW)]
            acc.setdefault(j, []).append((hull.volume, crop.flatten(0, 1).mean(0)))
    out = {}
    for j, lst in acc.items():
        tot = sum(w for w, _ in lst)
        out[int(j)] = sum((w / tot) * f for w, f in lst).float()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan_data", required=True, help="SceneVerse HM3D scan_data")
    ap.add_argument("--glb_root", required=True, help="HM3D v0.2 glb dir")
    ap.add_argument("--m3d_dir", required=True, help="mask3d_xr.py --out_dir (proposals/, scenes.pt)")
    ap.add_argument("--dinov2", required=True, help="facebook/dinov2-giant (HF id or local dir)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--offsets", default=os.path.join(HERE, "data", "region_offsets.json"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--regions", nargs="*", help="only these <scan>_<sub> (debug)")
    ap.add_argument("--dump_dir", default="", help="save the rendered views (debug)")
    args = ap.parse_args()
    from transformers import AutoImageProcessor, AutoModel

    offsets = json.load(open(args.offsets))
    need = {}
    for name, sc in torch.load(os.path.join(args.m3d_dir, "scenes.pt"), weights_only=False).items():
        scan = name.split("__")[0]
        for sub, r in sc["sel"]:
            need.setdefault(f"{scan}_{sub}", set()).add(int(r))
    regions = [r for r in sorted(need) if not args.regions or r in args.regions]
    by_scan = {}
    for reg in regions:
        by_scan.setdefault(reg.rsplit("_", 1)[0], []).append(reg)
    i, n = map(int, args.shard.split("/"))
    scans = sorted(by_scan)[i::n]
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"videofeats.{i}.pt")
    res = torch.load(out, weights_only=False) if os.path.exists(out) else {}
    device = "cuda"
    processor = AutoImageProcessor.from_pretrained(args.dinov2)
    dino = AutoModel.from_pretrained(args.dinov2).to(device).eval()
    renderer = pyrender.OffscreenRenderer(W, H)
    for k, scan in enumerate(scans):
        todo = [r for r in by_scan[scan] if r not in res]
        if not todo:
            continue
        h = scan.split("-")[1]
        scene = pyrender.Scene.from_trimesh_scene(trimesh.load(os.path.join(args.glb_root, scan, f"{h}.glb")),
                                                  bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        cam = scene.add(pyrender.IntrinsicsCamera(F, F, K[0, 2], K[1, 2], znear=0.05, zfar=60.0))

        def render(c2w):
            scene.set_pose(cam, c2w @ CV2GL)
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
            return color[..., :3], depth.astype(np.float64)

        for reg in todo:
            sub = reg.rsplit("_", 1)[1]
            pcd = torch.load(os.path.join(args.scan_data, "pcd_with_global_alignment", f"{reg}.pth"),
                             weights_only=False)
            pts = np.asarray(pcd[0], np.float64) + np.asarray(offsets[scan][sub]["T"])
            views = [v for _, vs in camera_positions(pts, render) for v in vs]
            if args.dump_dir:
                os.makedirs(os.path.join(args.dump_dir, reg), exist_ok=True)
                for f, (c2w, color, _) in enumerate(views):
                    Image.fromarray(color).save(os.path.join(args.dump_dir, reg, f"{f:03d}.jpg"))
            with torch.no_grad():
                tokens = dinov2_tokens([Image.fromarray(c) for _, c, _ in views], processor, dino, device)
            p = torch.load(os.path.join(args.m3d_dir, "proposals", f"{reg}.pt"), weights_only=False)
            masks = {r: np.asarray(p["masks"][r], np.int64) for r in sorted(need[reg])}
            res[reg] = fuse(views, tokens, pts, masks)
            print(f"[{k + 1}/{len(scans)}] {reg}: {len(views)} views, {len(res[reg])}/{len(masks)} proposals seen",
                  flush=True)
        scene.clear()
        torch.save(res, out + ".tmp")
        os.replace(out + ".tmp", out)
    renderer.delete()
    print(f"wrote {len(res)} regions to {out}")


if __name__ == "__main__":
    main()
