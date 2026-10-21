#!/bin/bash
# Scene packages + Mask3D inputs for 3RScan, MultiScan (Beacon3D scans) and, with
# SCANNET_CHECK=1, the 16 Beacon3D ScanNet scenes, rebuilt the same way to validate the
# rebuild against Robin3D's released ScanNet features (CPU only).
#   bash extra_eval/feats/run_prepare.sh
# Paths from env.sh: FEATS_ENV, FEATS_WORK (output root), RSCAN_RAW (3RScan download with
# scans/<id>/{labels.instances.annotated.v2.ply,mesh.refined.0.010000.segs.v2.json,sequence.zip}),
# MSCAN_RAW (unzipped MultiScan scenes: <scene>/<scene>.{ply,json,jsonl,mp4,depth.zlib}), RSCAN_SCAN_DATA / MULTISCAN_SCAN_DATA
# (SceneVerse scan_data), MASK3D_META (Mask3D scannet200 label_database.yaml + color_mean_std.yaml);
# for the ScanNet check SCANNET_SCANS (ply/txt/segs) and SCANNET_SENS (<scene>/<scene>.sens).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
PY="${FEATS_ENV:?}/bin/python"
W="${FEATS_WORK:?}/packages"
B=extra_eval/beacon3d/Beacon3D/data
mkdir -p "$W"
scenes() { $PY -c "import json,sys; d=json.load(open('$B/$1/$1_grounding.json')); q=json.load(open('$B/$1/$1_qa.json'));
print(json.dumps(sorted({x['scan_id'] for x in d}|{x['scene_id'] for x in q})))" > "$W/$1_scenes.json"; }
scenes 3rscan; scenes multiscan
$PY extra_eval/feats/prepare_scenes.py --domain 3rscan --scenes "$W/3rscan_scenes.json" --out_dir "$W" \
    --mask3d_meta "${MASK3D_META:?}" --rscan_raw "${RSCAN_RAW:?}" --rscan_sceneverse "${RSCAN_SCAN_DATA:?}"
$PY extra_eval/feats/prepare_scenes.py --domain multiscan --scenes "$W/multiscan_scenes.json" --out_dir "$W" \
    --mask3d_meta "$MASK3D_META" --mscan_raw "${MSCAN_RAW:?}" --mscan_sceneverse "${MULTISCAN_SCAN_DATA:?}"
if [ "${SCANNET_CHECK:-0}" = 1 ]; then
  scenes scannet
  $PY extra_eval/feats/prepare_scenes.py --domain scannet --scenes "$W/scannet_scenes.json" --out_dir "$W" \
      --mask3d_meta "$MASK3D_META" --scannet_scans "${SCANNET_SCANS:?}" --scannet_sens "${SCANNET_SENS:?}"
fi
echo PREPARE_DONE
