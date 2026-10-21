#!/bin/bash
# Robin3D inference on ScanQA (val), Real-3DQA (debiased_test + 4 rotations) and Beacon3D
# ScanNet (grounding + QA) with the released checkpoint, evaluate-only (torchrun over all
# visible GPUs). ScanRefer (val) and SQA3D (test) also run, as checks against the authors'
# released predictions. Then score with extra_eval/score.sh (CPU only).
#   bash extra_eval/run_extra_eval.sh
#   VAL_TAG="scanqa#beacon3d_scannet_qa" bash extra_eval/run_extra_eval.sh   # a subset
# Paths come from env.sh (see env.sh.example); relative ones are under the repo root:
#   ROBIN3D_ENV    venv from extra_eval/build_env_cu128.sh            (default .venv-cu128)
#   ROBIN3D_DATA   Google Drive download (extra_eval/download.sh)
#   LLAMA          Vicuna-7B v1.5, HF id or local path                (default lmsys/vicuna-7b-v1.5)
#   CKPT           released checkpoint      (default $ROBIN3D_DATA/ckpt_for_robin3d_final_performance/ckpt_01.pth)
#   REAL3DQA_DIR   Real-3DQA <split>.jsonl files (HF Oliver-Ma/Real-3DQA)
#   FEATS_WORK     extra_eval/feats work dir; when set, the Beacon3D sets on rebuilt features
#                  are registered too (VAL_TAG="beacon3d_3rscan_grounding#..." to run them)
#   OUT            output dir                                         (default outputs/extra_eval_<timestamp>)
#   NGPU           GPUs for torchrun                                  (default: all visible)
#   BATCH_SIZE     per-GPU batch size                                 (default 8)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
ENV="${ROBIN3D_ENV:-.venv-cu128}"
export PYTHONPATH=.:${PYTHONPATH:-}
export MASTER_PORT=$((54000 + RANDOM % 10000)) MASTER_ADDR=localhost
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
export LLAMA="${LLAMA:-lmsys/vicuna-7b-v1.5}"   # also read by dataset/base_dataset.py
DATA="${ROBIN3D_DATA:?set ROBIN3D_DATA}"
CKPT="${CKPT:-$DATA/ckpt_for_robin3d_final_performance/ckpt_01.pth}"
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }   # train.py would silently use random weights
REAL3DQA="real3dqa_debiased_test#real3dqa_rotation_0#real3dqa_rotation_90#real3dqa_rotation_180#real3dqa_rotation_270"
VAL_TAG="${VAL_TAG:-check_scanrefer#scanqa#sqa3d_test#$REAL3DQA#beacon3d_scannet_grounding#beacon3d_scannet_qa}"
REBUILT_ARGS=()
if [ -n "${FEATS_WORK:-}" ]; then  # Beacon3D on features rebuilt by extra_eval/feats
  REBUILT_ARGS=(--rebuilt_dir "$FEATS_WORK/robin3d_inputs" --packages_dir "$FEATS_WORK/packages")
fi
OUT="${OUT:-outputs/extra_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
$ENV/bin/python extra_eval/build_annos.py --robin3d_data "$DATA" --llama "$LLAMA" \
    --real3dqa_dir "${REAL3DQA_DIR:?set REAL3DQA_DIR}" "${REBUILT_ARGS[@]}" || exit 1
echo "output_dir=$OUT val_tag=$VAL_TAG ckpt=$CKPT ngpu=$NGPU"
# Model options = scripts/eval.sh (the released checkpoint's stage-2 run).
$ENV/bin/torchrun --nproc_per_node="$NGPU" --master_port=$MASTER_PORT tasks/train.py scripts/extra_eval_config.py \
    output_dir "$OUT" evaluate True pretrained_path "$CKPT" val_tag "$VAL_TAG" train_tag "" \
    wandb.enable False gpu_num "$NGPU" do_save False batch_size "${BATCH_SIZE:-8}" num_workers 4 \
    scheduler.epochs 3 optimizer.lr 1e-5 seed 42 resume False do_eval_during_train False \
    model.input_dim 1024 model.img_input_dim 1536 model.input_mask3d_dim 128 model.max_obj_num 150 \
    lora.lora_r 16 lora.lora_alpha 16 optimizer.max_grad_norm 0.01 model.max_txt_len 64 \
    model.add_scene_token False model.add_pos_emb False model.add_box_emb False \
    model.add_sid_eid False model.add_objlabel False model.add_layer_norm False \
    model.add_mask_token False model.add_mask3d_token True
echo "ROBIN3D_EXIT=$?"
ls "$OUT"
echo ROBIN3D_DONE
