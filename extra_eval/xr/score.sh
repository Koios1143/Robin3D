#!/bin/bash
# Score every XR prediction file in $OUT (preds_epoch*_step*_xr*.json) with score.py and
# print the overall scores (CPU only; needs Java for METEOR / PTB).
#   OUT=outputs/xr_eval_<ts> bash extra_eval/xr/score.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
ENV="${ROBIN3D_ENV:-.venv-cu128}"
ANNO=annotations/xr
for f in "${OUT:?}"/preds_epoch*_step*_xr*.json; do
  case "$f" in *_scores.json) continue ;; esac
  task=$(basename "$f" .json | sed -E 's/^preds_epoch-?[0-9]+_step[0-9]+_//')
  base=$(echo "$task" | sed -E 's/_(rpos|no2d)$//')
  split=$(echo "$base" | sed -E 's/^(xrqa|xrcap)_([a-z]+).*/\1_\2/')
  echo "== $(basename "$f")"
  $ENV/bin/python extra_eval/xr/score.py --preds "$f" --anno "$ANNO/$base.json" \
      --scene_map "$ANNO/${split}_scene_map.json" > /dev/null || { echo "score failed: $f"; continue; }
  $ENV/bin/python -c "
import json, sys
s = json.load(open(sys.argv[1]))
for n in ('robin3d', 'raw'):
    for k, v in s[n].items():
        print(f'  {n:8s} {k:6s}', {a: round(b, 2) for a, b in v.items()})
print('  mean_pred_words', round(s['mean_pred_words'], 2))" "${f%.json}_scores.json"
done
