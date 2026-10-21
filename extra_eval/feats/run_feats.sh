#!/bin/bash
# GPU part of the rebuild: Mask3D proposals + query features, then Uni3D / DINOv2 / boxes,
# for the packages made by run_prepare.sh. Writes $FEATS_WORK/robin3d_inputs/<domain>_*.pt.
#   bash extra_eval/feats/run_feats.sh                 # 3rscan multiscan
#   DOMAINS="scannet" bash extra_eval/feats/run_feats.sh
# With scannet, two Mask3D inputs are compared with Robin3D's released features:
# "scannet" = the package (axis-aligned coordinates, as 3RScan / MultiScan get) and
# "scannet_rawm3d" = Mask3D's own ScanNet preprocessing (raw coordinates, as Robin3D ran it;
# SCANNET_M3D_DIR = a processed scannet200 dir with those scenes, e.g. Mask3D's
# datasets/preprocessing/scannet_preprocessing.py output).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
W="${FEATS_WORK:?}"
OUT="$W/robin3d_inputs"
mkdir -p "$OUT"
feats() {  # <prefix> <packages domain> <mask3d out> [--upright]
  "${FEATS_ENV:?}/bin/python" extra_eval/feats/build_scene_feats.py --packages "$W/packages/$2" \
      --mask3d "$3" --out_dir "$OUT" --prefix "$1" --uni3d_ckpt "${UNI3D_CKPT:?}" --dinov2 "${DINOV2:?}" ${4:-}
}
for d in ${DOMAINS:-3rscan multiscan}; do
  DATA_DIR="$W/packages/$d/scannet200" OUT_DIR="$W/mask3d/$d" bash extra_eval/feats/run_mask3d.sh
  if [ "$d" = scannet ]; then
    feats scannet scannet "$W/mask3d/scannet"
    "${FEATS_ENV}/bin/python" - "$W/packages/scannet/scene_map.json" "${SCANNET_M3D_DIR:?}" "$W/m3d_scannet_raw" <<'PY'
import json, shutil, sys, yaml
from pathlib import Path
scenes, src, dst = json.load(open(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
db = [x for x in yaml.safe_load(open(src / "validation_database.yaml"))
      if f"scene{x['scene']:04d}_{x['sub_scene']:02d}" in scenes]
assert len(db) == len(scenes), (len(db), len(scenes))
yaml.safe_dump(db, open(dst / "validation_database.yaml", "w"))
for f in ["label_database.yaml", "color_mean_std.yaml"]:
    shutil.copy(src / f, dst / f)
PY
    DATA_DIR="$W/m3d_scannet_raw" OUT_DIR="$W/mask3d/scannet_rawm3d" bash extra_eval/feats/run_mask3d.sh
    feats scannet_rawm3d scannet "$W/mask3d/scannet_rawm3d"
    for p in scannet scannet_rawm3d; do
      "${FEATS_ENV}/bin/python" extra_eval/feats/compare_scannet.py --mine "$OUT" --prefix $p \
          --released "${ROBIN3D_DATA:?}/mask3d_uni3d_dinov2/mask3d150sort_uni3d_dinov2G" --out "$OUT/compare_$p.json"
    done
  else
    feats "$d" "$d" "$W/mask3d/$d" --upright
  fi
done
echo FEATS_DONE
