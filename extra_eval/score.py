#!/usr/bin/env python3
"""Score a ``run_extra_eval.sh`` output dir (CPU only; pycocoevalcap needs Java).

* ScanQA: Robin3D's own ``calc_scanqa_score`` with the scorers ``tasks/train.py`` uses
  for ScanQA (BLEU-1..4, METEOR, ROUGE-L, CIDEr), plus EM@1 / refined EM with
  ``utils/helper.py``'s ``answer_match`` on ``clean_answer``-ed text (the same matching
  Robin3D's SQA3D scorer uses; its ScanQA scorer has the EM lines commented out).
* Real-3DQA: the official ``evaluate_debiased_exact_match.py`` (EM, refined EM) on
  ``debiased_test`` and on each rotation split, and ``evaluate_rotation_robustness.py``
  (One..Four, VRS) on the four rotation splits. An empty generation is replaced by
  ``<empty>`` first; otherwise the refined match counts it as correct, because the empty
  string is a substring of every answer.
* Beacon3D ScanNet: Robin3D grounds to its own Mask3D proposals. The first ``<OBJxxx>``
  (below the scene's proposal count, as in ``calc_scanrefer_score``) picks a proposal, and
  the proposal box is mapped to the GT object of highest 3D box IoU
  (``scannet_val_attributes.pt``). Variant ``""`` accepts any overlap (closest to picking
  an object, as GT-mask models do); ``"_iou25"`` also needs IoU >= 0.25. Both go through
  Beacon3D's ``evaluate_grounding.py``; QA goes through ``evaluate_qa.py`` with the GPT
  judge (``--qa_model``), chained to the any-overlap grounding.

* checks: ScanRefer (val; Acc@0.25/0.5 as in Robin3D's ``calc_scanrefer_score``) and
  SQA3D (test; Robin3D's ``calc_sqa3d_score``), plus, for these two and ScanQA, the share of
  questions whose generation is identical to the authors' released predictions
  (``--released_dir``). Robin3D's ``OBJ_lm_head`` (the logits of the ``<OBJxxx>`` tokens) is
  frozen and not in the checkpoint, so it comes from the seeded random init; identical
  generations show that this run rebuilt the same model.

Writes ``<result_dir>/<bench>/`` files and ``<result_dir>/summary.json``.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / "real3dqa"))

REAL3DQA_SPLITS = {"debiased_test": 1485, "rotation_0": 750, "rotation_90": 750,
                   "rotation_180": 750, "rotation_270": 750}
ROTATIONS = ["rotation_0", "rotation_90", "rotation_180", "rotation_270"]
EMPTY = "<empty>"
OBJ = re.compile(r"<OBJ(\d{3})>")
VARIANTS = {"": 0.0, "_iou25": 0.25}


def load_preds(pred_dir, name):
    files = glob.glob(os.path.join(pred_dir, f"preds_epoch*_step*_{name}.json"))
    files = [f for f in files if "_rank" not in os.path.basename(f)]
    if not files:
        return None
    assert len(files) == 1, f"several prediction files for {name}: {files}"
    return json.load(open(files[0]))


def rel(text):
    return text.replace(str(REPO) + "/", "")


def score_scanqa(preds, config):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from utils.eval import calc_scanqa_score
    # = scorers_scanqa in tasks/train.py
    scorers_scanqa = [(Bleu(), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(), "METEOR"),
                      (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]
    from utils.helper import answer_match, clean_answer
    assert len(preds) == 4675, len(preds)
    s = calc_scanqa_score(preds, PTBTokenizer(), scorers_scanqa, config)
    em = emr = 0
    for p in preds:
        pred = clean_answer(p["pred"])
        a, b = answer_match(pred, [clean_answer(c) for c in p["ref_captions"]])
        em += a
        emr += b
    s["[scanqa] EM1"] = em / len(preds)
    s["[scanqa] EM1_refined"] = emr / len(preds)
    return {k: 100 * float(v) for k, v in s.items()}  # fractions -> percent


def score_real3dqa(pred_dir, out, name):
    import evaluate_debiased_exact_match as dem
    import evaluate_rotation_robustness as rot
    res, files = {"empty_preds": {}}, {}
    for split, n in REAL3DQA_SPLITS.items():
        preds = load_preds(pred_dir, f"real3dqa_{split}")
        if preds is None:
            return None
        assert len(preds) == n, f"{split}: expected {n}, got {len(preds)}"
        rows = []
        for p in preds:
            empty = not dem.clean_answer(p["pred"]).strip()
            res["empty_preds"][split] = res["empty_preds"].get(split, 0) + empty
            # int ids: the rotation script int()s the suffix-stripped 90/180/270 ids
            # but not the 0-degree ones, so string ids would never intersect
            rows.append({"question_id": int(p["qid"]), "scene_id": p["scene_id"],
                         "response_gt": p["ref_captions"], "response_pred": EMPTY if empty else p["pred"]})
        files[split] = out / f"pred_{split}.json"
        json.dump(rows, open(files[split], "w"), indent=1)
        res[split] = dem.evaluate_em(str(files[split]))
    res["rotation"] = rot.quick_consistency_score([str(files[s]) for s in ROTATIONS])
    # the same numbers through the unmodified scripts' CLIs, for the record
    log = []
    for cmd in [[sys.executable, str(HERE / "real3dqa" / "evaluate_debiased_exact_match.py"),
                 str(files["debiased_test"]), "--name", name],
                [sys.executable, str(HERE / "real3dqa" / "evaluate_rotation_robustness.py")]
                + [str(files[s]) for s in ROTATIONS] + ["--name", name]]:
        log.append(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    (out / "official_scripts.txt").write_text(rel("\n".join(log)))
    return res


def agreement(preds, released_dir, name, prefix):
    """Share of items (matched by annotation index) whose generation equals the released one."""
    files = glob.glob(os.path.join(released_dir, f"preds_epoch1_step*_{name}.json"))
    if len(files) != 1:
        return None
    rel_pred = {p["eval_name_index"].rsplit("_", 1)[1]: p["pred"] for p in json.load(open(files[0]))}
    ours = {p["eval_name_index"].rsplit("_", 1)[1]: p["pred"] for p in preds}
    assert ours.keys() == rel_pred.keys(), name
    assert all(p["eval_name_index"].startswith(prefix) for p in preds)
    return 100 * sum(ours[k] == rel_pred[k] for k in ours) / len(ours)


def scanrefer_acc(preds, config):
    """``calc_scanrefer_score`` minus the unique/multiple lookup it reads from the authors' disk."""
    import torch
    from utils.box_utils import box3d_iou, construct_bbox_corners
    prop_attrs = torch.load(config.val_file_dict["scanrefer"][4], map_location="cpu")
    gt_attrs = torch.load(config.val_file_dict["scanrefer"][6], map_location="cpu")
    hit25 = hit50 = 0
    for p in preds:
        locs = prop_attrs[p["scene_id"]]["locs"]
        pid = next((int(m.group(1)) for m in OBJ.finditer(p["pred"]) if int(m.group(1)) < locs.shape[0]), 0)
        a, g = locs[pid].tolist(), gt_attrs[p["scene_id"]]["locs"][p["gt_id"]].tolist()
        iou = box3d_iou(construct_bbox_corners(a[:3], a[3:]), construct_bbox_corners(g[:3], g[3:]))
        hit25 += iou >= 0.25
        hit50 += iou >= 0.5
    return {"Acc@0.25": 100 * hit25 / len(preds), "Acc@0.50": 100 * hit50 / len(preds)}


def score_checks(pred_dir, released_dir, config):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from utils.eval import calc_sqa3d_score
    res = {}
    preds = load_preds(pred_dir, "check_scanrefer")
    if preds is not None:
        assert len(preds) == 9508, len(preds)
        res["scanrefer"] = scanrefer_acc(preds, config)
        res["scanrefer"]["same_as_released"] = agreement(preds, released_dir, "scanrefer", "check_scanrefer_")
    preds = load_preds(pred_dir, "sqa3d_test")
    if preds is not None:
        assert len(preds) == 3519, len(preds)
        s = calc_sqa3d_score(preds, PTBTokenizer(), [], config, "sqa3d_test")
        res["sqa3d_test"] = {k.split("] ")[1]: 100 * v for k, v in s.items()}
        res["sqa3d_test"]["same_as_released"] = agreement(preds, released_dir, "sqa3d_test", "sqa3d_test_")
    preds = load_preds(pred_dir, "scanqa")
    if preds is not None:
        res["scanqa"] = {"same_as_released": agreement(preds, released_dir, "scanqa", "scanqa_")}
    return res or None


def parse_printout(text):
    """``key: number`` lines of a Beacon3D evaluator printout -> dict (last one wins)."""
    res = {}
    for k, v in re.findall(r"([A-Za-z-]+) = ([-\d.]+)", text):  # "EM = 49.44 | EM-R = ... | GPT-Score = ..."
        res[k] = float(v)
    for line in text.splitlines():
        m = re.match(r"^\s*\[?([A-Za-z0-9 /()_-]+?)\]?:\s*([-\d.]+(?:,\s*[-\d.]+)*)\s*$", line)
        if m:
            vals = [float(v) for v in m.group(2).split(",")]
            res[m.group(1).strip()] = vals[0] if len(vals) == 1 else vals
    return res


def run_eval(name, cmd, beacon_root, bdir):
    print(f"$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=beacon_root, capture_output=True, text=True)
    out = rel(p.stdout + ("" if p.returncode == 0 else "\n[stderr]\n" + p.stderr))
    (bdir / f"score_{name}.txt").write_text(out)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print(p.stderr[-3000:])
        raise SystemExit(f"{bdir.name} {name} evaluator failed")
    return parse_printout(p.stdout)


def map_to_gt(pred, prop_locs, gt_locs):
    """First valid ``<OBJxxx>`` -> (proposal, GT object id of max box IoU, that IoU)."""
    from utils.box_utils import box3d_iou, construct_bbox_corners
    prop = next((int(m.group(1)) for m in OBJ.finditer(pred) if int(m.group(1)) < len(prop_locs)), None)
    if prop is None:
        return None, None, 0.0
    p = prop_locs[prop].tolist()
    pc = construct_bbox_corners(p[:3], p[3:])
    best, best_iou = None, 0.0
    for gid, g in enumerate(gt_locs.tolist()):
        iou = box3d_iou(pc, construct_bbox_corners(g[:3], g[3:]))
        if iou > best_iou:
            best, best_iou = gid, iou
    return prop, best, float(best_iou)


def score_beacon3d(pred_dir, out, beacon_root, qa_model, config):
    import torch
    beacon_root = Path(beacon_root).resolve()
    src = {t: json.load(open(beacon_root / "data" / "scannet" / f"scannet_{t}.json")) for t in ["grounding", "qa"]}
    preds = {}
    for task in src:
        p = load_preds(pred_dir, f"beacon3d_scannet_{task}")
        if p is None:
            return None
        preds[task] = {int(x["qid"]): x for x in p}
        assert sorted(preds[task]) == list(range(len(src[task]))), f"{task}: {len(p)} vs {len(src[task])}"
    prop_attrs = torch.load(config.val_file_dict["beacon3d_scannet_grounding"][4], map_location="cpu")
    gt_attrs = torch.load(config.val_file_dict["beacon3d_scannet_grounding"][6], map_location="cpu")
    res = {"qa_judge": qa_model}
    mapped = []
    for i, s in enumerate(src["grounding"]):
        scene = s["scan_id"]
        prop, gid, iou = map_to_gt(preds["grounding"][i]["pred"], prop_attrs[scene]["locs"], gt_attrs[scene]["locs"])
        mapped.append((prop, gid, iou))
    res["grounding_no_obj_token"] = sum(m[0] is None for m in mapped)
    for tag, thr in VARIANTS.items():
        infer = [{"item_id": s["item_id"], "scan_id": s["scan_id"], "pred": preds["grounding"][i]["pred"],
                  "pred_proposal": prop, "pred_object_id": gid, "pred_iou": round(iou, 4),
                  "target_id": s["target_id"],
                  "correct": gid is not None and gid in s["target_id"] and iou >= thr}
                 for i, (s, (prop, gid, iou)) in enumerate(zip(src["grounding"], mapped))]
        json.dump(infer, open(out / f"infer_grounding{tag}.json", "w"), indent=1)
        res[f"grounding{tag}"] = run_eval(
            f"grounding{tag}", [sys.executable, str(HERE / "beacon3d" / "beacon_grounding_shim.py"),
                                "--infer", str(out / f"infer_grounding{tag}.json"), "--domain", "scannet"],
            beacon_root, out)
    infer = [{"question_id": s["question_id"], "scene_id": s["scene_id"], "question": s["question"],
              "pred": preds["qa"][i]["pred"], "answers": s["answers"]} for i, s in enumerate(src["qa"])]
    path = out / "infer_qa.json"
    json.dump(infer, open(path, "w"), indent=1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("[beacon3d_scannet] qa: OPENAI_API_KEY unset, GPT-judged evaluation skipped")
        return res
    processed = out / "infer_qa_processed.json"
    processed.unlink(missing_ok=True)  # evaluate_qa.py silently reuses an existing one
    res["qa"] = run_eval("qa", [sys.executable, "evaluate_qa.py", "--infer", str(path), "--domain", "scannet",
                                "--model", qa_model, "--output", str(processed),
                                "--grounding", str(out / "infer_grounding_processed.json")], beacon_root, out)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True, help="run_extra_eval.sh output dir")
    ap.add_argument("--result_dir", default="extra_eval/results")
    ap.add_argument("--config", default="scripts/extra_eval_config.py")
    ap.add_argument("--benches", default="checks,scanqa,real3dqa,beacon3d_scannet")
    ap.add_argument("--released_dir", default="",
                    help="authors' released preds (ckpt_for_robin3d_final_performance/) for the checks")
    ap.add_argument("--beacon_root", default="extra_eval/beacon3d/Beacon3D")
    ap.add_argument("--qa_model", default="gpt-4o-mini-2024-07-18",
                    help="Beacon3D QA judge (upstream default gpt-4o-2024-11-20)")
    ap.add_argument("--name", default="Robin3D")
    args = ap.parse_args()
    os.chdir(REPO)
    cfg = {}
    exec(open(args.config).read(), cfg)
    config = SimpleNamespace(val_file_dict=cfg["val_file_dict"])
    result_dir = Path(args.result_dir).resolve()  # the Beacon3D evaluators run from their own dir
    summary_file = result_dir / "summary.json"
    summary = json.load(open(summary_file)) if summary_file.exists() else {}
    for bench in args.benches.split(","):
        out = result_dir / bench
        out.mkdir(parents=True, exist_ok=True)
        if bench == "real3dqa":
            res = score_real3dqa(args.pred_dir, out, args.name)
        elif bench == "beacon3d_scannet":
            res = score_beacon3d(args.pred_dir, out, args.beacon_root, args.qa_model, config)
        elif bench == "checks":
            res = score_checks(args.pred_dir, args.released_dir, config)
        elif bench == "scanqa":
            preds = load_preds(args.pred_dir, "scanqa")
            res = None if preds is None else score_scanqa(preds, config)
        else:
            raise SystemExit(f"unknown bench {bench}")
        if res is None:
            print(f"[{bench}] no predictions in {args.pred_dir}, skipped")
            continue
        res = {"pred_dir": os.path.relpath(args.pred_dir, REPO), **res}
        json.dump(res, open(out / "scores.json", "w"), indent=1)
        summary[bench] = res
        print(f"[{bench}]", json.dumps(res, indent=1))
    json.dump(summary, open(summary_file, "w"), indent=1)


if __name__ == "__main__":
    main()
