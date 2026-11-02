#!/usr/bin/env python3
"""Robin3D's 3D object features (Uni3D-g) and boxes for Mask3D proposals of HM3D regions.

Per proposal some XR scene selects (``mask3d_xr.py scenes`` -> ``scenes.pt``): Uni3D-g over the
proposal's points of SceneVerse's region point cloud, as Robin3D's ScanNet features
(``extra_eval/feats/uni3d.py: encode_object``: centred, max-radius scaled, colour / 255), and
its axis-aligned box (centre + size) in the scan frame (region points + the region's offset,
``data/region_offsets.json``), as Robin3D's ``attributes`` ``locs``. A proposal whose points
all coincide (Uni3D's scaling would divide by zero) gets no feature (Robin3D feeds zeros).

Output ``<out_dir>/uni3d.<shard>.pt`` = {"<scan>_<sub>": {rank: {"feat": float32[1024],
"loc": float32[6], "npts": int}}}.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "feats"))
from uni3d import encode_object, load_uni3d  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan_data", required=True, help="SceneVerse HM3D scan_data")
    ap.add_argument("--m3d_dir", required=True, help="mask3d_xr.py --out_dir (proposals/, scenes.pt)")
    ap.add_argument("--uni3d_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--offsets", default=os.path.join(HERE, "data", "region_offsets.json"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--regions", nargs="*", help="only these <scan>_<sub> (debug)")
    args = ap.parse_args()
    offsets = json.load(open(args.offsets))
    need = {}
    for name, sc in torch.load(os.path.join(args.m3d_dir, "scenes.pt"), weights_only=False).items():
        scan = name.split("__")[0]
        for sub, r in sc["sel"]:
            need.setdefault(f"{scan}_{sub}", set()).add(int(r))
    regions = [r for r in sorted(need) if not args.regions or r in args.regions]
    i, n = map(int, args.shard.split("/"))
    regions = regions[i::n]
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"uni3d.{i}.pt")
    res = torch.load(out, weights_only=False) if os.path.exists(out) else {}
    device = "cuda"
    torch.manual_seed(1457)  # as build_scene_feats.py (Uni3D's FPS / kNN grouping is deterministic)
    model = load_uni3d(args.uni3d_ckpt, device)
    for k, reg in enumerate(regions):
        if reg in res:
            continue
        scan, sub = reg.rsplit("_", 1)
        pcd = torch.load(os.path.join(args.scan_data, "pcd_with_global_alignment", f"{reg}.pth"), weights_only=False)
        xyz = np.asarray(pcd[0], np.float64)
        rgb = np.asarray(pcd[1], np.float64)
        T = np.asarray(offsets[scan][sub]["T"])
        p = torch.load(os.path.join(args.m3d_dir, "proposals", f"{reg}.pt"), weights_only=False)
        cur = {}
        for r in sorted(need[reg]):
            m = np.asarray(p["masks"][r], np.int64)
            pc = xyz[m]
            ent = {"npts": len(m), "loc": torch.zeros(6)}
            if len(m):
                w = pc + T
                ent["loc"] = torch.from_numpy(np.concatenate([(w.max(0) + w.min(0)) / 2, w.max(0) - w.min(0)])).float()
            if len(m) and np.ptp(pc, axis=0).max() > 0:
                with torch.no_grad():
                    ent["feat"] = encode_object(model, pc, rgb[m], device)
            cur[r] = ent
        res[reg] = cur
        print(f"[{k + 1}/{len(regions)}] {reg}: {len(cur)} proposals, "
              f"{sum('feat' in e for e in cur.values())} encoded", flush=True)
        if (k + 1) % 50 == 0:
            torch.save(res, out + ".tmp")
            os.replace(out + ".tmp", out)
    torch.save(res, out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"wrote {len(res)} regions to {out}")


if __name__ == "__main__":
    main()
