#!/usr/bin/env python3
"""XR-QA / XR-SceneCaption (LSceneLLM) -> Robin3D feature / annotation files + ``scripts/xr_config.py``.

An XR scene is a set of SceneVerse HM3D regions of one scan. Its objects are Mask3D proposals
(the setting Robin3D is evaluated and trained with on ScanNet): its regions' proposals pooled by
score, the 150 best kept (``mask3d_xr.py scenes``; Robin3D has 150 object identifiers). No
ground truth is read, except the GT object count of a scene for the <=100 / >100 score split.
Each scene gets an alias ``xq<NNNN>_00`` / ``xc<NNNN>_00`` (Robin3D recovers a feature key's
scene as ``'_'.join(key.split('_')[:2])``); feature keys are ``<alias>_<column:02>``.

Features per selected instance: ``mask3d_feats`` (Mask3D query feature), ``mask3d_feat_pos``
(its positional embedding, recomputed in the scene's frame; ``_rpos`` files keep the per-region
one Mask3D gave), ``uni3d_feats`` (``uni3d_xr.py``), ``videofeats`` (``videofeats_xr.py``);
``attributes`` = {"objects": Mask3D ScanNet200 class names (``--label_db``), "locs": boxes in the scan frame}.

Prompts (Robin3D's user turn; the object list follows the question, as for ScanNet):
  XR-QA            raw: the question; short: + " Answer the question using a word or a
                   phrase." (Robin3D's ScanQA prompt)
  XR-SceneCaption  the XR question without its "### human:" / "### assistant:" wrapper
Train splits get Robin3D train items (``caption`` = one reference answer; a caption with several
references becomes several items).

Writes ``<anno_dir>/`` ``xr{qa,cap}_<split>_*.pt``, ``xr{qa,cap}_<split>[_<prompt>].json``,
``xr{qa,cap}_<split>_scene_map.json``; ``scripts/xr_config.py`` (``scripts/config.py`` with the
LLM path, plus ``val_file_dict`` / ``train_file_dict`` entries ``xrqa_<split>_<prompt>``,
``xrcap_<split>``, and ``..._rpos`` / ``..._no2d`` variants for val).
"""

import argparse
import glob
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xr_common import MAX_OBJ, cap_items, qa_items, scene_name, scenes  # noqa: E402

QA_SUFFIX = " Answer the question using a word or a phrase."


def load_shards(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        out.update(torch.load(f, map_location="cpu", weights_only=False))
    return out


def build(task, split, items, m3d, m3d_dir, uni3d, video, scan_data, anno_dir, prefix, props_cache, label_names):
    keys = ["mask3d_feats", "mask3d_feat_pos", "mask3d_feat_pos_rpos", "uni3d_feats", "videofeats"]
    out = {k: {} for k in keys}
    attrs, scene_map, alias = {}, {}, {}
    for k, (scan, regs) in enumerate(scenes(items)):
        a = f"{prefix}{k:04d}_00"
        alias[(scan, regs)] = a
        sc = m3d[scene_name(scan, regs)]
        names, locs = [], []
        for col, (sub, r) in enumerate(sc["sel"]):
            reg = f"{scan}_{sub}"
            if reg not in props_cache:
                p = torch.load(os.path.join(m3d_dir, "proposals", f"{reg}.pt"), weights_only=False)
                props_cache[reg] = {"feat": p["feat"], "classes": p["classes"]}
            p = props_cache[reg]
            key = f"{a}_{col:02}"
            out["mask3d_feats"][key] = p["feat"][r].float()
            out["mask3d_feat_pos"][key] = sc["pos_scene"][col].float()
            out["mask3d_feat_pos_rpos"][key] = sc["pos_region"][col].float()
            u = uni3d[reg][r]
            if "feat" in u:
                out["uni3d_feats"][key] = u["feat"]
            v = video.get(reg, {}).get(r)
            if v is not None:
                out["videofeats"][key] = v
            names.append(label_names.get(int(p["classes"][r]), str(p["classes"][r])))
            locs.append(u["loc"])
        # Robin3D's prompt always holds 150 objects (its ScanNet scenes all have 150 proposals);
        # a scene with fewer gets zero objects (absent feature keys are fed as zeros)
        names += [""] * (MAX_OBJ - len(names))
        locs += [torch.zeros(6)] * (MAX_OBJ - len(locs))
        if f"{a}_00" not in out["uni3d_feats"]:  # Robin3D lists a split's scenes from these keys
            out["uni3d_feats"][f"{a}_00"] = torch.zeros(1024)
        attrs[a] = {"objects": names, "locs": torch.stack(locs)}
        n_gt = sum(len(torch.load(os.path.join(scan_data, "instance_id_to_label", f"{scan}_{s}.pth"),
                                  weights_only=False)) for s in regs)
        scene_map[a] = {"scan": scan, "regions": list(regs), "n_obj": n_gt, "n_prop": len(sc["sel"])}
    base = os.path.join(anno_dir, f"{task}_{split}")
    for k in keys:
        torch.save(out[k], f"{base}_{k}.pt")
    torch.save(attrs, f"{base}_attributes.pt")
    json.dump(scene_map, open(f"{base}_scene_map.json", "w"))
    n = [m["n_prop"] for m in scene_map.values()]
    print(f"{task}_{split}: {len(items)} items, {len(scene_map)} scenes, proposals mean {sum(n) / max(len(n), 1):.1f}; "
          f"uni3d {len(out['uni3d_feats'])}, 2D {len(out['videofeats'])} of {len(out['mask3d_feats'])}")
    return alias


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xr_dir", required=True)
    ap.add_argument("--scan_data", required=True, help="SceneVerse HM3D scan_data")
    ap.add_argument("--m3d_dir", required=True, help="mask3d_xr.py --out_dir")
    ap.add_argument("--feats_dir", required=True, help="uni3d_xr.py / videofeats_xr.py --out_dir")
    ap.add_argument("--llama", required=True, help="Vicuna-7B v1.5 path or HF id")
    ap.add_argument("--anno_dir", default="annotations/xr")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--config_out", default="scripts/xr_config.py")
    ap.add_argument("--label_db", default="", help="Mask3D label_database.yaml (ScanNet200 id -> name)")
    ap.add_argument("--allow_missing", action="store_true",
                    help="debug: drop items whose scene has no features (mask3d_xr.py run on a region subset)")
    args = ap.parse_args()
    os.makedirs(args.anno_dir, exist_ok=True)
    m3d = torch.load(os.path.join(args.m3d_dir, "scenes.pt"), weights_only=False)
    label_names = {}
    if args.label_db:
        import yaml
        label_names = {int(k): v["name"] for k, v in yaml.safe_load(open(args.label_db)).items()}
    uni3d = load_shards(os.path.join(args.feats_dir, "uni3d.*.pt"))
    video = load_shards(os.path.join(args.feats_dir, "videofeats.*.pt"))
    print(f"scenes {len(m3d)}, uni3d regions {len(uni3d)}, 2D regions {len(video)}")
    anno = os.path.abspath(args.anno_dir)
    cfg = ["\n\n# ---- XR-QA / XR-SceneCaption (generated by extra_eval/xr/build_xr.py) ----"]
    props_cache = {}
    for split in args.splits:
        for task, items, prefix in (("xrqa", qa_items(args.xr_dir, split), "xq"),
                                    ("xrcap", cap_items(args.xr_dir, split), "xc")):
            if args.allow_missing:
                items = [it for it in items if scene_name(it["scan"], tuple(it["regions"])) in m3d]
            alias = build(task, split, items, m3d, args.m3d_dir, uni3d, video, args.scan_data, args.anno_dir, prefix, props_cache,
                          label_names)
            base = os.path.join(anno, f"{task}_{split}")
            prompts = {"raw": "", "short": QA_SUFFIX} if task == "xrqa" else {"": ""}
            for pname, suffix in prompts.items():
                out = []
                for i, it in enumerate(items):
                    a = alias[(it["scan"], tuple(it["regions"]))]
                    if split == "train":
                        out += [{"scene_id": a, "obj_id": 0, "prompt": it["question"] + suffix, "caption": ans}
                                for ans in it["answers"]]
                    else:
                        out.append({"scene_id": a, "obj_id": 0, "prompt": it["question"] + suffix,
                                    "ref_captions": it["answers"], "qid": i, "key": it["key"],
                                    **({"type": it["type"]} if "type" in it else {})})
                tag = f"_{pname}" if pname else ""
                js = f"{base}{tag}.json"
                json.dump(out, open(js, "w"), indent=1)
                name = f"{task}_{split}{tag}"
                # [feat_pos, mask3d_feats, uni3d, video, attributes, annotation, gt attributes (unused here)]
                variants = {"": ("mask3d_feat_pos", "videofeats")}
                if split != "train":
                    variants.update({"_rpos": ("mask3d_feat_pos_rpos", "videofeats"),
                                     "_no2d": ("mask3d_feat_pos", None)})
                d = "train_file_dict" if split == "train" else "val_file_dict"
                for v, (pos, vid) in variants.items():
                    files = [f"{base}_{pos}.pt", f"{base}_mask3d_feats.pt", f"{base}_uni3d_feats.pt",
                             f"{base}_{vid}.pt" if vid else None, f"{base}_attributes.pt", js,
                             f"{base}_attributes.pt"]
                    cfg.append(f"{d}[{name + v!r}] = {files!r}")
    src = open("scripts/config.py").read()
    src, n = re.subn(r'llama_model_path=".*?"', f"llama_model_path={args.llama!r}", src, count=1)
    assert n == 1
    open(args.config_out, "w").write(src + "\n".join(cfg) + "\n")
    print(f"config -> {args.config_out}")


if __name__ == "__main__":
    main()
