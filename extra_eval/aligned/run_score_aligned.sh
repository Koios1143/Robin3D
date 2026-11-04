#!/bin/bash
# Re-score the saved predictions with the common normalisation (extra_eval/aligned/aligned_eval.py).
# CPU only, no GPU; PTB / METEOR need Java (this repo's deps/jre11 unless JAVA_HOME is set).
#   bash extra_eval/aligned/run_score_aligned.sh [--only scan2cap,scanqa,sqa3d,xrcap,xrqa]
# Writes extra_eval/results/aligned/<bench>__<variant>.json and summary.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
export JAVA_HOME="${JAVA_HOME:-deps/jre11}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH=".:${PYTHONPATH:-}"
PY="${ROBIN3D_ENV:-deps/venv}/bin/python"
java -version 2>&1 | head -1
"$PY" extra_eval/aligned/score_aligned.py "$@"
