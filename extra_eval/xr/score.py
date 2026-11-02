#!/usr/bin/env python3
"""Score Robin3D predictions on XR-QA / XR-SceneCaption the way LSceneLLM does.

LSceneLLM (``tools/runner_Adaptive_LLM.py: validate / score_captions``) scores the raw
decoded text, only collapsing repeated spaces, with LL3DA's ``capeval`` (vendored in
``capeval/``), keyed by ``episode_id`` (``HM3D#<scan>#<region_id>#<question>``), so repeated
keys keep the last item. Scores are x100.

``--norm robin3d`` instead scores like Robin3D's own ScanQA metric (``utils/eval.py:
calc_scanqa_score``): the prediction is cut at its first ``<OBJxxx>`` (``split_OBJ``), loses a
trailing '.' and gets its first letter lower-cased; predictions and references then go through
the PTB tokenizer (no ``clean_answer``: commented out there). ``--norm both`` (default) writes both.

    python extra_eval/xr/score.py --preds <out>/preds_epoch-1_step0_xrqa_val_short.json \
        --anno annotations/xr/xrqa_val_short.json [--scene_map annotations/xr/xrqa_val_scene_map.json]
Writes ``<preds without .json>_scores.json``.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from capeval.bleu.bleu import Bleu  # noqa: E402
from capeval.cider.cider import Cider  # noqa: E402
from capeval.meteor.meteor import Meteor  # noqa: E402
from capeval.rouge.rouge import Rouge  # noqa: E402


def clean(s):
    return " ".join(w for w in s.split(" ") if w)


def score(corpus, cands):
    keys = list(cands)
    gts = OrderedDict((k, corpus[k]) for k in keys)
    res = OrderedDict((k, cands[k]) for k in keys)
    bleu = Bleu(4).compute_score(gts, res)
    cider = Cider().compute_score(gts, res)
    rouge = Rouge().compute_score(gts, res)
    meteor = Meteor().compute_score(gts, res)
    return {"n": len(keys), "CIDEr": 100 * cider[0], "BLEU-4": 100 * bleu[0][3],
            "METEOR": 100 * meteor[0], "ROUGE-L": 100 * rouge[0]}


def robin3d_norm(corpus, cands):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    sys.path.insert(0, os.path.join(HERE, "..", ".."))
    from utils.eval import split_OBJ

    def pred(p):
        p, _ = split_OBJ(p)
        if len(p) > 1:
            p = p[:-1] if p[-1] == "." else p
            p = p[0].lower() + p[1:]
        return p

    tok = PTBTokenizer()
    return (tok.tokenize({k: [{"caption": c.replace("\n", " ").strip()} for c in v] for k, v in corpus.items()}),
            tok.tokenize({k: [{"caption": pred(v[0])}] for k, v in cands.items()}))


def score_splits(corpus, cands, scene, n_obj):
    out = {"all": score(corpus, cands)}
    for name, keep in (("le100", lambda n: n <= 100), ("gt100", lambda n: n > 100)):
        sub = {k: v for k, v in cands.items() if n_obj and keep(n_obj[scene[k]])}
        if sub:
            out[name] = score(corpus, sub)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", required=True, help="preds_epoch*_step*_<task>.json")
    ap.add_argument("--anno", required=True)
    ap.add_argument("--scene_map", default="", help="adds scores split by GT object count (<=100 / >100)")
    ap.add_argument("--norm", default="both", choices=["raw", "robin3d", "both"])
    args = ap.parse_args()

    anno = json.load(open(args.anno))
    preds = {int(p["qid"]): p["pred"] for p in json.load(open(args.preds))}
    missing = [i for i in range(len(anno)) if anno[i]["qid"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(anno)} items have no prediction")
    corpus, cands, scene = {}, {}, {}
    for a in anno:
        corpus[a["key"]] = list(a["ref_captions"])
        cands[a["key"]] = [clean(preds[a["qid"]])]
        scene[a["key"]] = a["scene_id"]
    n_obj = {k: v["n_obj"] for k, v in json.load(open(args.scene_map)).items()} if args.scene_map else {}
    out = {}
    if args.norm in ("raw", "both"):
        out["raw"] = score_splits(corpus, cands, scene, n_obj)
    if args.norm in ("robin3d", "both"):
        out["robin3d"] = score_splits(*robin3d_norm(corpus, cands), scene, n_obj)
    out["mean_pred_words"] = sum(len(v[0].split()) for v in cands.values()) / len(cands)
    dst = args.preds[:-5] + "_scores.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
