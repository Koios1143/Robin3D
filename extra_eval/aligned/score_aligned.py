#!/usr/bin/env python3
"""Re-score Robin3D's saved predictions with the common text normalisation (aligned_eval.py).

No inference: reads the prediction files the existing scorers used and writes
``extra_eval/results/aligned/<bench>__<variant>.json`` plus ``summary.json``.

Prediction cleanup (Robin3D only): ``models/chat3d.py`` decodes without skipping special
tokens and cuts at ``end_sym`` (``</s>``), so the only special tokens that can remain are the
grounding tokens ``<OBJxxx>`` (and, defensively, ``<s>`` / ``</s>`` / ``<unk>``). They are
removed; nothing else is changed (no truncation, no ``split_OBJ`` cut, no lower-casing).

* Scan2Cap (val, ``s2c``): Robin3D's own matching (``utils/eval.py: calc_scan2cap_score``):
  each prediction is for (scene, GT object, Mask3D proposal); it counts at IoU >= 0.25 / 0.5
  between the proposal box and the GT box, otherwise pred ``""``. Corpus objects without a
  prediction also get ``""``. References: ``deps/aligned/scan2cap_val_corpus.json``.
* ScanQA (val): ``clean_ptb`` and ``raw`` (+ EM / EM-R), refs = saved ``ref_captions``.
* SQA3D (test): EM / EM-R only.
* XR-SceneCaption / XR-QA: every saved zero-shot and finetune prediction file, keyed by the
  item (``qid``; all 2994 XR-QA / 283 XR-SceneCaption items, never deduplicated by the
  LSceneLLM episode id, unlike ``extra_eval/xr/score.py``; ``n_unique_episodes`` recorded).

    python extra_eval/aligned/score_aligned.py [--only scan2cap,xrqa]
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from aligned_eval import em_scores, score_captions, score_qa  # noqa: E402

OUT = REPO / "extra_eval" / "results" / "aligned"
SPECIAL = re.compile(r"<OBJ\d{3}>|</?s>|<unk>")
EXPECTED = {"scan2cap": 2022, "scanqa": 4675, "sqa3d_test": 3519, "xrcap": 283, "xrqa": 2994}
N_CORPUS = 2068
XR_UNIQUE = {"xrcap": 283, "xrqa": 2981}
XR_HEADLINE = {  # what the existing XR write-up reports (extra_eval/results/xr, xr-scene notes)
    "zeroshot": {"xrqa": "xrqa_val_short", "xrcap": "xrcap_val"},
    "finetune": "epoch2 (last epoch, stage-2 recipe; xrqa prompt = short)",
}


def strip_special(text):
    return " ".join(SPECIAL.sub(" ", text or "").split())


def rel(p):
    return os.path.relpath(p, REPO)


def load(path, n):
    preds = json.load(open(path))
    if len(preds) != n:
        raise SystemExit(f"{rel(path)}: expected {n} predictions, got {len(preds)}")
    return preds


def paper_pred_file(bench, tag):
    """The prediction file extra_eval/score.py scored (pred_dir recorded in its summary)."""
    pred_dir = json.load(open(REPO / "extra_eval" / "results" / "summary.json"))[bench]["pred_dir"]
    files = [f for f in glob.glob(str(REPO / pred_dir / f"preds_epoch*_step*_{tag}.json")) if "_rank" not in f]
    if len(files) != 1:
        raise SystemExit(f"{bench}: expected one prediction file for {tag} in {pred_dir}, got {files}")
    return files[0]


def scan2cap():
    import torch
    from utils.box_utils import box3d_iou, construct_bbox_corners
    data = os.environ.get("ROBIN3D_DATA")
    if not data:
        raise SystemExit("set ROBIN3D_DATA (source env.sh)")
    # = config.val_file_dict['scan2cap'][4] (proposals) and scannet_attribute_file (GT boxes)
    prop_attrs = torch.load(f"{data}/mask3d_uni3d_dinov2/mask3d150sort_uni3d_dinov2G/scannet_mask3d_val_attributes.pt",
                            map_location="cpu")
    gt_attrs = torch.load(f"{data}/chat3d-anno/scannet/scannet_val_attributes.pt", map_location="cpu")
    corpus = json.load(open(REPO / "deps" / "aligned" / "scan2cap_val_corpus.json"))
    if len(corpus) != N_CORPUS:
        raise SystemExit(f"scan2cap corpus: expected {N_CORPUS} keys, got {len(corpus)}")
    src = paper_pred_file("scan2cap", "check_scan2cap")
    preds = load(src, EXPECTED["scan2cap"])
    by_thr = {0.25: {}, 0.5: {}}
    for p in preds:
        key = f"{p['scene_id']}|{p['gt_id']}"
        if key not in corpus:
            raise SystemExit(f"scan2cap: prediction key {key} not in the corpus")
        pl = prop_attrs[p["scene_id"]]["locs"][p["pred_id"]].tolist()
        gl = gt_attrs[p["scene_id"]]["locs"][p["gt_id"]].tolist()
        iou = box3d_iou(construct_bbox_corners(pl[:3], pl[3:]), construct_bbox_corners(gl[:3], gl[3:]))
        for thr, d in by_thr.items():
            d[key] = strip_special(p["pred"]) if iou >= thr else ""
    missing = [k for k in corpus if k not in by_thr[0.25]]
    res = {}
    for thr, d in by_thr.items():
        matched = sum(bool(v) for v in d.values())
        preds_all = {k: d.get(k, "") for k in corpus}
        refs = {k: corpus[k] for k in corpus}
        variant = f"iou{int(thr * 100)}"
        res[variant] = {"source": rel(src), "n": len(preds_all), "n_predictions": len(preds),
                        "n_matched": matched, "n_corpus_without_prediction": len(missing),
                        "metrics": score_captions(refs, preds_all, ("s2c",))}
    return res


def scanqa():
    src = paper_pred_file("scanqa", "scanqa")
    preds = load(src, EXPECTED["scanqa"])
    refs = {i: list(p["ref_captions"]) for i, p in enumerate(preds)}
    hyp = {i: strip_special(p["pred"]) for i, p in enumerate(preds)}
    return {"val": {"source": rel(src), "n": len(hyp), "metrics": score_qa(refs, hyp, ("clean_ptb", "raw"))}}


def sqa3d():
    src = paper_pred_file("sqa3d_test", "sqa3d_test")
    preds = load(src, EXPECTED["sqa3d_test"])
    refs = {i: list(p["ref_captions"]) for i, p in enumerate(preds)}
    hyp = {i: strip_special(p["pred"]) for i, p in enumerate(preds)}
    old = json.load(open(REPO / "extra_eval" / "results" / "summary.json"))["sqa3d_test"]
    return {"test": {"source": rel(src), "n": len(hyp), "metrics": {"em": em_scores(hyp, refs)},
                     "existing": {"EM": old["EM1"], "EM-R": old["EM1_refined"]}}}


def xr(task):
    """task = xrcap | xrqa; every saved prediction file of the zero-shot and finetune runs."""
    res = {}
    files = sorted(glob.glob(str(REPO / "extra_eval" / "results" / "xr" / "*" / f"preds_epoch*_step*_{task}_val*.json")))
    files = [f for f in files if not f.endswith("_scores.json")]
    if not files:
        raise SystemExit(f"{task}: no prediction files")
    for f in files:
        run = Path(f).parent.name
        m = re.match(rf"preds_epoch(-?\d+)_step\d+_({task}_val\w*)\.json$", Path(f).name)
        epoch, pset = int(m.group(1)), m.group(2)
        anno_name = re.sub(r"_(rpos|no2d)$", "", pset)  # placement variants share the annotation file
        anno = json.load(open(REPO / "annotations" / "xr" / f"{anno_name}.json"))
        preds = load(f, EXPECTED[task])
        by_qid = {int(p["qid"]): p for p in preds}
        if sorted(by_qid) != sorted(a["qid"] for a in anno):
            raise SystemExit(f"{rel(f)}: qids differ from annotations/xr/{anno_name}.json")
        refs, hyp = {}, {}
        for a in anno:  # keyed per item (qid), never deduplicated by episode id
            p = by_qid[a["qid"]]
            if list(p["ref_captions"]) != list(a["ref_captions"]):
                raise SystemExit(f"{rel(f)}: qid {a['qid']} refs differ from the annotation")
            refs[a["qid"]] = list(a["ref_captions"])
            hyp[a["qid"]] = strip_special(p["pred"])
        if len(hyp) != EXPECTED[task]:
            raise SystemExit(f"{rel(f)}: expected {EXPECTED[task]} items, got {len(hyp)}")
        n_unique = len({a["key"] for a in anno})
        if n_unique != XR_UNIQUE[task]:
            raise SystemExit(f"{rel(f)}: expected {XR_UNIQUE[task]} unique episodes, got {n_unique}")
        variant = f"{run}_epoch{epoch}_{pset}" if epoch >= 0 else f"{run}_{pset}"
        if run == "zeroshot":
            headline = XR_HEADLINE["zeroshot"][task] == pset
        else:
            headline = epoch == 2 and pset in ("xrcap_val", "xrqa_val_short")
        metrics = (score_captions(refs, hyp, ("ptb", "raw")) if task == "xrcap"
                   else score_qa(refs, hyp, ("clean_ptb", "raw")))
        res[variant] = {"source": rel(f), "n": len(hyp), "n_unique_episodes": n_unique, "n_predictions": len(preds),
                        "headline": headline, "metrics": metrics}
    return res


BENCHES = {"scan2cap": scan2cap, "scanqa": scanqa, "sqa3d": sqa3d,
           "xrcap": lambda: xr("xrcap"), "xrqa": lambda: xr("xrqa")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated subset of " + ",".join(BENCHES))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary_file = OUT / "summary.json"
    summary = json.load(open(summary_file)) if summary_file.exists() else {}
    for bench in (args.only.split(",") if args.only else BENCHES):
        for variant, r in BENCHES[bench]().items():
            name = f"{bench}__{variant}"
            r = {"bench": bench, "variant": variant, **r}
            json.dump(r, open(OUT / f"{name}.json", "w"), indent=1)
            summary[name] = r
            print(name, json.dumps(r["metrics"], indent=1), flush=True)
        json.dump(summary, open(summary_file, "w"), indent=1)
    summary["_notes"] = {"xr_headline": XR_HEADLINE,
                         "prediction_cleanup": "removed <OBJxxx>, <s>, </s>, <unk>; whitespace collapsed"}
    json.dump(summary, open(summary_file, "w"), indent=1)


if __name__ == "__main__":
    main()
