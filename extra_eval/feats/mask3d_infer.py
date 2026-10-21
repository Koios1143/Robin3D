#!/usr/bin/env python3
"""Mask3D (ScanNet200 ``scannet200_val.ckpt``) proposals + query features, Robin3D style.

Robin3D's released ScanNet inputs (``mask3d150sort``) are Mask3D's official ScanNet200 test
setting -- 150 queries, top-750 (query, class) pairs, DBSCAN eps 0.95 on segments -- cut to
the 150 highest-scoring instances, each carrying the 128-d feature and 128-d positional
embedding of the query it came from (a query can yield several instances: other classes or
DBSCAN clusters). Mask3D's export drops the query, so this script re-runs that
post-processing (``trainer.eval_instance_step`` + ``get_mask_and_scores``, copied with the
query index kept) and reads the features with forward hooks; Mask3D itself is unmodified.

Saved per instance (``--feat`` picks which hooked tensors are written as the features):
  ``feat``  = final ``decoder_norm`` output (what the class / mask heads read)
  ``pos``   = ``query_projection`` output (the query positional embedding)
``extra_eval/feats/compare_scannet.py`` checks both against Robin3D's released files.

Input: a Mask3D ``scannet200`` processed dir (``validation_database.yaml`` + ``<scene>.npy``:
xyz, rgb, normal, segment, label, instance). Output ``<out_dir>/<scene>.pt``:
``{"masks": [np.int64 point ids] x150, "scores", "classes" (ScanNet200 names), "query",
"feat" [150,128], "pos" [150,128], "feat_raw" [150,128]}``.
"""

import argparse
import os
import sys

import numpy as np
import torch


def build(mask3d_root, ckpt, data_dir):
    from hydra.experimental import compose, initialize_config_dir
    overrides = [
        # = scripts/scannet200/scannet200_val.sh (TEST)
        f"general.checkpoint={ckpt}", "data/datasets=scannet200", "general.num_targets=201",
        "data.num_labels=200", "general.eval_on_segments=true", "general.train_on_segments=true",
        "general.train_mode=false", "model.num_queries=150", "general.topk_per_image=750",
        "general.use_dbscan=true", "general.dbscan_eps=0.95",
        f"data.validation_dataset.data_dir={data_dir}",
        f"data.validation_dataset.label_db_filepath={data_dir}/label_database.yaml",
        f"data.validation_dataset.color_mean_std={data_dir}/color_mean_std.yaml",
        f"data.train_dataset.label_db_filepath={data_dir}/label_database.yaml",
        f"data.train_dataset.color_mean_std={data_dir}/color_mean_std.yaml",
        "data.train_dataset.data_dir=" + data_dir, "data.test_dataset.data_dir=" + data_dir,
        f"data.test_dataset.label_db_filepath={data_dir}/label_database.yaml",
        f"data.test_dataset.color_mean_std={data_dir}/color_mean_std.yaml",
        "data.train_dataset.mode=validation", "data.test_dataset.mode=validation",
    ]
    with initialize_config_dir(config_dir=os.path.join(mask3d_root, "conf")):
        cfg = compose(config_name="config_base_instance_segmentation.yaml", overrides=overrides)
    import yaml
    import datasets.semseg as semseg
    # Mask3D calls yaml.load(f) without a Loader, which PyYAML >= 6 rejects
    semseg.SemanticSegmentationDataset._load_yaml = staticmethod(
        lambda p: yaml.load(open(p), Loader=yaml.FullLoader))
    from trainer.trainer import InstanceSegmentation
    from utils.utils import load_checkpoint_with_missing_or_exsessive_keys
    import hydra
    model = InstanceSegmentation(cfg)
    cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    dataset = hydra.utils.instantiate(cfg.data.validation_dataset)
    collate = hydra.utils.instantiate(cfg.data.validation_collation)
    return cfg, model, dataset, collate


def mask_and_scores(mask_cls, mask_pred, num_queries, num_classes, topk):
    """``InstanceSegmentation.get_mask_and_scores`` that also returns the instance index."""
    labels = torch.arange(num_classes).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
    scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(topk if topk != -1 else num_queries, sorted=True)
    labels_per_query = labels[topk_indices]
    inst = topk_indices // num_classes
    mask_pred = mask_pred[:, inst]
    result_pred_mask = (mask_pred > 0).float()
    heatmap = mask_pred.float().sigmoid()
    mask_scores = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
    return scores_per_query * mask_scores, result_pred_mask, labels_per_query, inst


@torch.no_grad()
def run_scene(model, cfg, batch, hooks, keep):
    import MinkowskiEngine as ME
    from sklearn.cluster import DBSCAN
    data, target, file_names = batch
    raw_coordinates = data.features[:, -3:]
    feats = data.features[:, :-3]
    x = ME.SparseTensor(coordinates=data.coordinates, features=feats, device=model.device)
    hooks.clear()
    output = model.forward(x, point2segment=[t["point2segment"].to(model.device) for t in target],
                           raw_coordinates=raw_coordinates, is_eval=True)
    logits = torch.softmax(output["pred_logits"][0], dim=-1)[..., :-1].cpu()  # [Q, 200]
    masks = output["pred_masks"][0].detach().cpu()[target[0]["point2segment"].cpu()]  # low-res pts x Q
    coords = raw_coordinates.cpu().numpy()
    new_masks, new_logits, new_query = [], [], []
    for q in range(masks.shape[1]):  # DBSCAN split, as eval_instance_step
        cur = masks[:, q] > 0
        if coords[cur.numpy()].shape[0] == 0:
            continue
        clusters = DBSCAN(eps=cfg.general.dbscan_eps, min_samples=cfg.general.dbscan_min_points,
                          n_jobs=-1).fit(coords[cur.numpy()]).labels_
        new_mask = torch.zeros(cur.shape, dtype=int)
        new_mask[cur] = torch.from_numpy(clusters) + 1
        for cid in np.unique(clusters):
            if cid != -1:
                new_masks.append(masks[:, q] * (new_mask == cid + 1))
                new_logits.append(logits[q])
                new_query.append(q)
    scores, m, classes, inst = mask_and_scores(torch.stack(new_logits), torch.stack(new_masks).T,
                                               len(new_logits), model.model.num_classes - 1,
                                               cfg.general.topk_per_image)
    query = torch.tensor(new_query)[inst]
    m = model.get_full_res_mask(m, data.inverse_maps[0], data.target_full[0]["point2segment"])
    order = scores.sort(descending=True).indices[:keep]
    classes = model.validation_dataset._remap_model_output(classes[order].cpu() + model.validation_dataset.label_offset)
    q = query[order]
    feat = hooks["decoder_norm"][-1][0].float().cpu()  # [Q, 128]
    feat_raw = hooks["decoder_norm_in"][-1][0].float().cpu()
    pos = hooks["query_projection"][-1][0].float().cpu().T  # [D, Q] -> [Q, D]
    return {
        "masks": [np.nonzero(m[:, i].numpy())[0] for i in order.tolist()],
        "scores": scores[order].numpy(), "classes": np.asarray(classes), "query": q.numpy(),
        "feat": feat[q], "feat_raw": feat_raw[q], "pos": pos[q],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mask3d_root", required=True, help="clone of JonasSchult/Mask3D")
    ap.add_argument("--ckpt", required=True, help="scannet200_val.ckpt")
    ap.add_argument("--data_dir", required=True, help="processed scannet200 dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--keep", type=int, default=150)
    args = ap.parse_args()
    sys.path.insert(0, args.mask3d_root)
    # Mask3D's loss / matcher import detectron2 helpers; inference never calls them
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d2shim"))
    cwd = os.getcwd()
    os.chdir(args.mask3d_root)  # Mask3D resolves some config paths relative to its root
    cfg, model, dataset, collate = build(args.mask3d_root, os.path.abspath(os.path.join(cwd, args.ckpt)),
                                         os.path.abspath(os.path.join(cwd, args.data_dir)))
    os.chdir(cwd)
    model = model.cuda().eval()
    model.validation_dataset = dataset
    hooks = {}

    def hook(name, take_input=False):
        def f(_m, inp, out):
            hooks.setdefault(name, []).append((inp[0] if take_input else out).detach())
        return f
    model.model.decoder_norm.register_forward_hook(hook("decoder_norm"))
    model.model.decoder_norm.register_forward_hook(hook("decoder_norm_in", take_input=True))
    model.model.query_projection.register_forward_hook(hook("query_projection"))
    os.makedirs(args.out_dir, exist_ok=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    for batch in loader:
        name = batch[2][0]
        out = os.path.join(args.out_dir, f"{name}.pt")
        if os.path.exists(out):
            continue
        res = run_scene(model, cfg, batch, hooks, args.keep)
        torch.save(res, out)
        print(name, len(res["masks"]), "instances from", len(set(res["query"].tolist())), "queries", flush=True)


if __name__ == "__main__":
    main()
