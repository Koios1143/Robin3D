#!/bin/bash
# Score an extra_eval/run_extra_eval.sh output dir (CPU only; needs Java for pycocoevalcap).
#   PREDS_DIR=outputs/extra_eval_<ts> bash extra_eval/score.sh
# RESULT_DIR defaults to extra_eval/results; BENCHES limits which benchmarks are (re)scored.
# Beacon3D QA is GPT-judged (QA_MODEL, default gpt-4o-mini-2024-07-18) and needs OPENAI_API_KEY.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${ROBIN3D_ENV:-.venv-cu128}/bin/python"
$PY extra_eval/score.py --pred_dir "${PREDS_DIR:?set PREDS_DIR}" --result_dir "${RESULT_DIR:-extra_eval/results}" \
    --benches "${BENCHES:-scanrefer,multi3dref,scan2cap,scanqa,sqa3d_test,real3dqa,beacon3d_scannet}" \
    --released_dir "${ROBIN3D_DATA:?set ROBIN3D_DATA}/ckpt_for_robin3d_final_performance" --qa_model "${QA_MODEL:-gpt-4o-mini-2024-07-18}" \
    --scan2cap_corpus "${SCAN2CAP_CORPUS:?set SCAN2CAP_CORPUS}" --scanrefer_lookup "${SCANREFER_LOOKUP:?set SCANREFER_LOOKUP}"
