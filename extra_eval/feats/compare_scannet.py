#!/usr/bin/env python3
"""Check re-extracted ScanNet features against Robin3D's released ones, row by row.

For each scene in ``--mine`` (``build_scene_feats.py`` output dir for ScanNet scenes) and
each released ``mask3d150sort`` file it reports, over the 150 proposal rows: the mean
cosine of row i with released row i, the share of rows with cosine > 0.99, and for boxes
the mean 3D IoU of row i with released row i (rows are in score order on both sides, so
these only agree if proposals, order and features all match). Order-free ("matched"):
each released proposal is paired with the re-extracted proposal of highest box IoU; the
report gives the mean of that IoU and, over pairs with IoU >= 0.9, the feature cosines.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def box_iou(a, b):
    amin, amax = a[:, :3] - a[:, 3:] / 2, a[:, :3] + a[:, 3:] / 2
    bmin, bmax = b[:, :3] - b[:, 3:] / 2, b[:, :3] + b[:, 3:] / 2
    inter = (torch.minimum(amax, bmax) - torch.maximum(amin, bmin)).clamp(min=0).prod(1)
    return inter / (a[:, 3:].prod(1) + b[:, 3:].prod(1) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mine", required=True, help="dir with <prefix>_{mask3d_feats,...}.pt from build_scene_feats.py")
    ap.add_argument("--prefix", default="scannet")
    ap.add_argument("--released", required=True, help="Robin3D mask3d150sort_uni3d_dinov2G dir")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    names = {"mask3d_feats": "scannet_mask3d_mask3d_feats.pt", "mask3d_feat_pos": "scannet_mask3d_mask3d_feat_pos.pt",
             "uni3d_feats": "scannet_mask3d_uni3d_feats.pt", "videofeats": "scannet_mask3d_videofeats.pt"}
    mine_attr = torch.load(os.path.join(args.mine, f"{args.prefix}_attributes.pt"))
    rel_attr = torch.load(os.path.join(args.released, "scannet_mask3d_val_attributes.pt"))
    scenes = sorted(mine_attr)
    report = {}
    match = {}
    for s in scenes:
        a, b = mine_attr[s]["locs"], rel_attr[s]["locs"]
        full = torch.stack([box_iou(a, b[j:j + 1].expand_as(a)) for j in range(len(b))])  # [rel, mine]
        best, idx = full.max(1)
        match[s] = [(j, int(idx[j])) for j in range(len(b)) if best[j] >= 0.9]
        report[s] = {"box_iou_rowwise": float(box_iou(a, b).mean()), "matched_box_iou": float(best.mean()),
                     "matched>=0.9": len(match[s])}
    for key, fname in names.items():
        mine = torch.load(os.path.join(args.mine, f"{args.prefix}_{key}.pt"))
        rel = torch.load(os.path.join(args.released, fname))
        for s in scenes:
            ids = [f"{s}_{i:02}" for i in range(150)]
            both = [k for k in ids if k in mine and k in rel]
            if not both:
                report[s][key] = None
                continue
            cos = F.cosine_similarity(torch.stack([mine[k].float() for k in both]),
                                      torch.stack([rel[k].float() for k in both]), dim=1)
            pairs = [(f"{s}_{j:02}", f"{s}_{i:02}") for j, i in match[s]]
            pairs = [(r_, m_) for r_, m_ in pairs if r_ in rel and m_ in mine]
            mcos = F.cosine_similarity(torch.stack([mine[m_].float() for _, m_ in pairs]),
                                       torch.stack([rel[r_].float() for r_, _ in pairs]), dim=1) if pairs else torch.zeros(1)
            report[s][key] = {"matched_cos": round(float(mcos.mean()), 4), "rows": len(both), "only_mine": sum(k in mine and k not in rel for k in ids),
                              "only_released": sum(k in rel and k not in mine for k in ids),
                              "mean_cos": round(float(cos.mean()), 4), "cos>0.99": round(float((cos > 0.99).float().mean()), 4)}
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
