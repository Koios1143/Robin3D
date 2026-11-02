#!/usr/bin/env python3
"""Mask3D proposals + query features for SceneVerse HM3D regions, and XR scene selections.

Robin3D's ScanNet inputs are the 150 best Mask3D (ScanNet200 ``scannet200_val.ckpt``)
instances of a scene, each with the 128-d feature and positional embedding of its query
(``extra_eval/feats/mask3d_infer.py``). XR has no ScanNet scans, so Mask3D runs on each HM3D
region separately (the same input and setting as the proposals LEO / Chat-Scene / GPT4Scene
use for XR: Grounded 3D-LLM's region preprocessing, LEO's ``prepare_regions.py``):

``infer`` (GPU): per region ``<out_dir>/proposals/<scan>_<sub>.pt`` = mask3d_infer.py's
  output (``masks`` as point ids of SceneVerse's region point cloud, ``scores``, ``classes``,
  ``query``, ``feat``, ``pos``) plus ``qxyz`` (the query's sampled point, region frame, metres)
  and ``rng`` (the region's min / max, the range Mask3D normalises query positions by).
  Kept: the 150 best (``--keep``); a region with few points may have fewer.

``scenes``: an XR scene (a set of regions of one scan) gets the 150 best proposals of its
  regions pooled by score (``xr_common.pool``). Mask3D's positional embedding of a query is
  ``query_projection(pos_enc(xyz, range))`` with the region's own range, so the pooled
  queries of different regions would share one frame; ``scenes`` recomputes it for every
  selected instance with its query point moved into the scan frame (+ the region's offset,
  ``extra_eval/xr/data/region_offsets.json``) and the range of all the scene's regions -- what
  Mask3D would give if the scene were one scan, with the same query point. Writes
  ``<out_dir>/scenes.pt`` = {scene name: {"sel": [(sub, rank)], "pos_scene": [N, 128],
  "pos_region": [N, 128]}}.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "feats"))
from xr_common import all_scenes, pool, scene_name  # noqa: E402


def mask_and_scores(mask_cls, mask_pred, num_classes, topk):
    """``InstanceSegmentation.get_mask_and_scores`` keeping the instance index; top-k capped
    at the number of (instance, class) pairs (a region with < 4 instances has fewer than 750)."""
    labels = torch.arange(num_classes).unsqueeze(0).repeat(len(mask_cls), 1).flatten(0, 1)
    scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(min(topk, labels.numel()), sorted=True)
    inst = topk_indices // num_classes
    mask_pred = mask_pred[:, inst]
    result_pred_mask = (mask_pred > 0).float()
    mask_scores = (mask_pred.float().sigmoid() * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
    return scores_per_query * mask_scores, result_pred_mask, labels[topk_indices], inst


@torch.no_grad()
def run_region(model, cfg, batch, hooks, keep):
    import MinkowskiEngine as ME
    from sklearn.cluster import DBSCAN
    data, target, _ = batch
    raw_coordinates = data.features[:, -3:]
    x = ME.SparseTensor(coordinates=data.coordinates, features=data.features[:, :-3], device=model.device)
    hooks.clear()
    output = model.forward(x, point2segment=[t["point2segment"].to(model.device) for t in target],
                           raw_coordinates=raw_coordinates, is_eval=True)
    logits = torch.softmax(output["pred_logits"][0], dim=-1)[..., :-1].cpu()  # [Q, 200]
    masks = output["pred_masks"][0].detach().cpu()[target[0]["point2segment"].cpu()]  # low-res pts x Q
    coords = raw_coordinates.cpu().numpy()
    new_masks, new_logits, new_query = [], [], []
    for q in range(masks.shape[1]):  # DBSCAN split, as eval_instance_step
        cur = (masks[:, q] > 0).numpy()
        if not cur.any():
            continue
        clusters = DBSCAN(eps=cfg.general.dbscan_eps, min_samples=cfg.general.dbscan_min_points,
                          n_jobs=-1).fit(coords[cur]).labels_
        new_mask = torch.zeros(cur.shape, dtype=int)
        new_mask[cur] = torch.from_numpy(clusters) + 1
        for cid in np.unique(clusters):
            if cid != -1:
                new_masks.append(masks[:, q] * (new_mask == cid + 1))
                new_logits.append(logits[q])
                new_query.append(q)
    qxyz, (mins, maxs) = hooks["query_xyz"]
    res = {"qxyz_all": qxyz, "rng": torch.stack([mins, maxs])}
    if not new_masks:
        return {**res, "masks": [], "scores": np.zeros(0, np.float32), "classes": np.zeros(0, object),
                "query": np.zeros(0, np.int64), "feat": torch.zeros(0, 128), "pos": torch.zeros(0, 128),
                "qxyz": torch.zeros(0, 3)}
    scores, m, classes, inst = mask_and_scores(torch.stack(new_logits), torch.stack(new_masks).T,
                                               model.model.num_classes - 1, cfg.general.topk_per_image)
    query = torch.tensor(new_query)[inst]
    m = model.get_full_res_mask(m, data.inverse_maps[0], data.target_full[0]["point2segment"])
    order = scores.sort(descending=True).indices[:keep]
    classes = model.validation_dataset._remap_model_output(classes[order].cpu() + model.validation_dataset.label_offset)
    q = query[order]
    feat = hooks["decoder_norm"][-1][0].float().cpu()  # [Q, 128]
    pos = hooks["query_projection"][-1][0].float().cpu().T  # [D, Q] -> [Q, D]
    return {**res, "masks": [np.nonzero(m[:, i].numpy())[0] for i in order.tolist()],
            "scores": scores[order].numpy(), "classes": np.asarray(classes), "query": q.numpy(),
            "feat": feat[q], "pos": pos[q], "qxyz": qxyz[q]}


def load_model(args):
    from mask3d_infer import build
    sys.path.insert(0, args.mask3d_root)
    # Mask3D's loss / matcher import detectron2 helpers; inference never calls them
    sys.path.append(os.path.join(HERE, "..", "feats", "d2shim"))
    cwd = os.getcwd()
    os.chdir(args.mask3d_root)  # Mask3D resolves some config paths relative to its root
    cfg, model, dataset, collate = build(args.mask3d_root, os.path.abspath(os.path.join(cwd, args.ckpt)),
                                         os.path.abspath(os.path.join(cwd, args.data_dir)))
    os.chdir(cwd)
    model.validation_dataset = dataset
    return cfg, model, dataset, collate


def infer(args):
    cfg, model, dataset, collate = load_model(args)
    model = model.cuda().eval()
    hooks = {}

    def hook(name):
        def f(_m, _inp, out):
            hooks.setdefault(name, []).append(out.detach())
        return f

    def pos_enc_hook(_m, inp, kwargs, _out):  # the last call before query_projection = the queries'
        hooks["_pos_enc_last"] = (inp[0], kwargs.get("input_range"))

    def query_hook(_m, _inp, out):
        xyz, (mins, maxs) = hooks["_pos_enc_last"]
        hooks["query_xyz"] = (xyz[0].float().cpu(), (mins[0].float().cpu(), maxs[0].float().cpu()))
        hooks.setdefault("query_projection", []).append(out.detach())

    model.model.decoder_norm.register_forward_hook(hook("decoder_norm"))
    model.model.pos_enc.register_forward_hook(pos_enc_hook, with_kwargs=True)
    model.model.query_projection.register_forward_hook(query_hook)
    out_dir = os.path.join(args.out_dir, "proposals")
    os.makedirs(out_dir, exist_ok=True)
    names = [os.path.basename(os.path.dirname(d["raw_filepath"])) for d in dataset.data]
    want = set(args.regions) if args.regions else None
    todo = [i for i in range(len(dataset)) if i % args.num_shards == args.shard
            and (want is None or names[i] in want)
            and not os.path.exists(os.path.join(out_dir, f"{names[i]}.pt"))]
    print(f"{len(todo)} of {len(dataset)} regions to run", flush=True)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, todo), batch_size=1, shuffle=False,
                                         num_workers=4, collate_fn=collate)
    for k, batch in enumerate(loader):
        name = batch[2][0]
        res = run_region(model, cfg, batch, hooks, args.keep)
        out = os.path.join(out_dir, f"{name}.pt")
        torch.save(res, out + ".tmp")
        os.replace(out + ".tmp", out)
        print(k, name, len(res["masks"]), "instances from", len(set(res["query"].tolist())), "queries", flush=True)


@torch.no_grad()
def scenes(args):
    _, model, _, _ = load_model(args)
    pos_enc, proj = model.model.pos_enc.eval(), model.model.query_projection.eval()
    offsets = json.load(open(args.offsets))
    out = {}
    cache = {}
    skipped = 0
    for scan, regs in all_scenes(args.xr_dir):
        if not all(os.path.exists(os.path.join(args.out_dir, "proposals", f"{scan}_{s}.pt")) for s in regs):
            skipped += 1  # a region not run (debug subset)
            continue
        props = {}
        for sub in regs:
            reg = f"{scan}_{sub}"
            if reg not in cache:
                p = torch.load(os.path.join(args.out_dir, "proposals", f"{reg}.pt"), weights_only=False)
                cache[reg] = {k: p[k] for k in ("scores", "qxyz", "rng", "pos")}
            props[sub] = cache[reg]
        sel = pool(regs, {s: p["scores"] for s, p in props.items()})
        T = {s: torch.tensor(offsets[scan][s]["T"], dtype=torch.float32) for s in regs}
        rngs = torch.stack([props[s]["rng"] + T[s] for s in regs])  # [R, 2, 3]
        mins, maxs = rngs[:, 0].min(0).values, rngs[:, 1].max(0).values
        entry = {"sel": sel, "pos_region": torch.zeros(len(sel), 128), "pos_scene": torch.zeros(len(sel), 128)}
        if sel:
            xyz = torch.stack([props[s]["qxyz"][r] + T[s] for s, r in sel])[None]  # [1, N, 3]
            # Mask3D runs pos_enc outside autocast in fp32; query_projection is a Conv1d MLP
            pe = pos_enc(xyz, input_range=[mins[None], maxs[None]])  # [1, D, N]
            entry["pos_scene"] = proj(pe)[0].T.float()
            entry["pos_region"] = torch.stack([props[s]["pos"][r] for s, r in sel])
        out[scene_name(scan, regs)] = entry
    torch.save(out, os.path.join(args.out_dir, "scenes.pt"))
    n = [len(v["sel"]) for v in out.values()]
    print(f"{len(out)} scenes ({skipped} skipped: regions without proposals), "
          f"instances mean {np.mean(n):.1f} min {min(n)} max {max(n)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["infer", "scenes"])
    ap.add_argument("--mask3d_root", required=True, help="clone of JonasSchult/Mask3D")
    ap.add_argument("--ckpt", required=True, help="scannet200_val.ckpt")
    ap.add_argument("--data_dir", required=True, help="processed scannet200 dir of the regions")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--xr_dir", default="", help="scenes: XR-QA / XR-SceneCaption dir")
    ap.add_argument("--offsets", default=os.path.join(HERE, "data", "region_offsets.json"))
    ap.add_argument("--keep", type=int, default=150)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--regions", nargs="*", help="infer: only these <scan>_<sub> (debug)")
    args = ap.parse_args()
    infer(args) if args.mode == "infer" else scenes(args)


if __name__ == "__main__":
    main()
