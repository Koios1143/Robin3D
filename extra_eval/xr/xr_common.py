"""XR-QA / XR-SceneCaption (LSceneLLM) items and scenes, shared by the extra_eval/xr scripts.

An XR scene is a set of SceneVerse HM3D regions of one HM3D scan (XR-QA ``qa_pairs[].region_id``,
XR-SceneCaption ``room_id``, e.g. ``sub002-sub012-sub014``), in LSceneLLM's order. Item keys are
LSceneLLM's episode ids (``HM3D#<scan>#<regions>#<question>``; captions also get their index),
as in the Chat-Scene fork's ``xr_eval/build_xr.py``. ``pool`` picks a scene's objects.
"""

import json
import os
import re


def qa_items(xr_dir, split):
    """LSceneLLM's HD_Hm3dQADataset order."""
    items = []
    for scan, eps in json.load(open(os.path.join(xr_dir, "XR-QA", f"{split}.json"))).items():
        for ep in eps:
            for qa in ep["qa_pairs"]:
                items.append({
                    "scan": scan,
                    "regions": qa["region_id"].split("-"),
                    "question": qa["question"],
                    "answers": [qa["answer"]],
                    "type": qa["type"],
                    "key": "HM3D#{}#{}#{}".format(scan, qa["region_id"], qa["question"]),
                })
    return items


def cap_items(xr_dir, split):
    """The XR question without its ``### human:`` / ``### assistant:`` wrapper."""
    items = []
    for i, x in enumerate(json.load(open(os.path.join(xr_dir, "XR-SceneCaption", f"{split}.json")))):
        q = re.sub(r"^\s*###\s*human:\s*|\s*###\s*assistant:\s*$", "", x["question"]).strip()
        items.append({
            "scan": x["scan_id"],
            "regions": x["room_id"].split("-"),
            "question": q,
            "answers": x["answers"],
            "key": "HM3D#{}#{}#{}#{}".format(x["scan_id"], x["room_id"], q, i),
        })
    return items


def scenes(items):
    """(scan, regions) of the items, first-seen order."""
    out = {}
    for it in items:
        out.setdefault((it["scan"], tuple(it["regions"])), None)
    return list(out)


def all_scenes(xr_dir, splits=("train", "val")):
    out = {}
    for split in splits:
        for get in (qa_items, cap_items):
            for s in scenes(get(xr_dir, split)):
                out.setdefault(s, None)
    return list(out)


def scene_name(scan, regions):
    return f"{scan}__{'-'.join(regions)}"


MAX_OBJ = 150  # Robin3D's object identifiers <OBJ000>..<OBJ149>


def pool(regions, scores, n=MAX_OBJ):
    """The scene's ``n`` best Mask3D instances over its regions: [(sub, rank in its region)],
    by descending score; equal scores keep region order (as the Chat-Scene / GPT4Scene XR
    pipelines pool their top-100). ``scores``: {sub: per-region scores, descending}."""
    allp = [(sub, r, float(s)) for sub in regions for r, s in enumerate(scores[sub])]
    allp.sort(key=lambda x: -x[2])
    return [(sub, r) for sub, r, _ in allp[:n]]
