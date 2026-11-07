"""ScanRefer's unique / multiple lookup, built exactly as ScanRefer's
``lib/dataset.py`` (``_get_raw2label`` + ``_get_unique_multiple_lookup``) does, from
ScanRefer's OWN copy of the ScanNet label table (``data/scannet/meta_data/
scannetv2-labels.combined.tsv`` in github.com/daveredrum/ScanRefer).

That copy spells the NYU40 class "refrigerator"; ScanNet's original table spells it
"refridgerator", so a lookup built from ScanNet's table (e.g. the
``scanrefer_unique_multiple_lookup.json`` shipped with Chat-Scene) files refrigerators and
mini fridges under "others" and labels 86 val sentences multiple that ScanRefer labels unique
(1,759 vs 1,845 unique of 9,508).

Output format = Chat-Scene's: {scene_id: {object_id (str): 0 unique | 1 multiple}}.

    python build_scanrefer_lookup.py --scanrefer ScanRefer_filtered.json \
        --labels scanrefer_meta/scannetv2-labels.combined.tsv --out lookup.json
"""

import argparse
import csv
import json
from collections import defaultdict

# ScanRefer's DC.type2class
TYPE2CLASS = ["cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture",
              "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink",
              "bathtub", "others"]


def raw2label(labels_path):
    index = {n: i for i, n in enumerate(TYPE2CLASS)}
    out = {}
    with open(labels_path) as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for line in rd:
            out[line[1]] = index.get(line[7], index["others"])
    return out


def build(scanrefer, r2l):
    label = lambda d: r2l.get(" ".join(d["object_name"].split("_")), 17)
    sem = defaultdict(dict)
    for d in scanrefer:
        sem[d["scene_id"]].setdefault(d["object_id"], label(d))
    lookup = defaultdict(dict)
    for d in scanrefer:
        n = sum(1 for c in sem[d["scene_id"]].values() if c == label(d))
        lookup[d["scene_id"]][d["object_id"]] = 0 if n == 1 else 1
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scanrefer", required=True, help="ScanRefer_filtered.json (or the val split)")
    ap.add_argument("--labels", required=True, help="ScanRefer's scannetv2-labels.combined.tsv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r2l = raw2label(args.labels)
    assert r2l.get("refrigerator") == TYPE2CLASS.index("refrigerator"), \
        "label table maps 'refrigerator' to others: this is ScanNet's table, not ScanRefer's copy"
    lookup = build(json.load(open(args.scanrefer)), r2l)
    json.dump(lookup, open(args.out, "w"))
    print(f"{sum(len(v) for v in lookup.values())} objects in {len(lookup)} scenes -> {args.out}")


if __name__ == "__main__":
    main()
