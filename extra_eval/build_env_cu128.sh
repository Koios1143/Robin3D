#!/bin/bash
# Robin3D runtime env (same recipe as Chat-Scene's extra_eval/build_env_cu128.sh) for Blackwell GPUs (sm_100/sm_120), which the upstream pins
# (torch 2.2.1+cu121, flash-attn 2.5.6) predate: python 3.10, torch 2.7.1+cu128, the
# flash-attn 2.8.3 prebuilt wheel, and upstream's requirements.txt otherwise
# (transformers 4.39.3, peft 0.9.0, pycocoevalcap 1.2).
#   bash extra_eval/build_env_cu128.sh
# Paths come from env.sh (see env.sh.example):
#   ROBIN3D_ENV   where the venv goes            (default .venv-cu128, under the repo root)
#   UV              the uv binary                  (default: uv on PATH)
#   UV_CACHE_DIR / TMPDIR   point these at a disk with room if $HOME is full
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
UV="${UV:-uv}"
ENV="${ROBIN3D_ENV:-.venv-cu128}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
[ -e "$ENV" ] && { echo "$ENV exists; not touching it (pick another ROBIN3D_ENV)"; exit 1; }
$UV python install 3.10
$UV venv "$ENV" -p 3.10 --seed
PIP="$UV pip install --python $ENV/bin/python"
$PIP torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
grep -v flash_attn requirements.txt > "${TMPDIR:-/tmp}/req_$$.txt"
$PIP -r "${TMPDIR:-/tmp}/req_$$.txt" pyyaml wandb tqdm timm plyfile openai "numpy<2"
$PIP "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
$ENV/bin/python -c "
import torch, transformers, peft, flash_attn
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__, 'peft', peft.__version__, 'flash_attn', flash_attn.__version__)
import models.chat3d, dataset, tasks.shared_utils; print('robin3d imports ok')
"
echo ENV_DONE
