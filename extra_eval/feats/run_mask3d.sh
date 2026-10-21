#!/bin/bash
# Mask3D proposals + query features for one processed scannet200 dir (one GPU).
#   DATA_DIR=<processed scannet200 dir> OUT_DIR=<out> bash extra_eval/feats/run_mask3d.sh
# Paths from env.sh: MASK3D_ENV (python with Mask3D's deps: MinkowskiEngine, hydra 1.0,
# pytorch-lightning 1.x), MASK3D_ROOT (clone of JonasSchult/Mask3D), MASK3D_CKPT
# (checkpoints/scannet200/scannet200_val.ckpt from the Mask3D README).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
export OMP_NUM_THREADS=3
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # Lightning checkpoint (holds its config); torch >= 2.6
"${MASK3D_ENV:?}/bin/python" extra_eval/feats/mask3d_infer.py --mask3d_root "${MASK3D_ROOT:?}" \
    --ckpt "${MASK3D_CKPT:?}" --data_dir "${DATA_DIR:?}" --out_dir "${OUT_DIR:?}"
