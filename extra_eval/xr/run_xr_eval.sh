#!/bin/bash
# Robin3D (released checkpoint, or CKPT=) on XR-QA / XR-SceneCaption val, then LSceneLLM-style
# scoring (score.py: raw text and Robin3D's own ScanQA normalisation). Needs run_xr_feats.sh.
#   bash extra_eval/xr/run_xr_eval.sh
#   VAL_TAG="xrqa_val_short" bash extra_eval/xr/run_xr_eval.sh
# Sets (build_xr.py): xrqa_val_{short,raw} (QA prompt + Robin3D's ScanQA suffix / bare question),
#   xrcap_val; suffix _rpos = Mask3D positions in each region's own frame, _no2d = no 2D features.
# Knobs: CKPT, OUT, NGPU, BATCH_SIZE (8), MAX_TXT_LEN (256: also the generation length;
#   XR captions are ~120 tokens), PREFIX_CACHE / GEN_BATCH (as run_extra_eval.sh), SKIP_BUILD=1,
#   BUILD_ARGS (extra build_xr.py args, e.g. --allow_missing for a debug subset).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
ENV="${ROBIN3D_ENV:-.venv-cu128}"
export PYTHONPATH=.:${PYTHONPATH:-}
export MASTER_PORT=$((54000 + RANDOM % 10000)) MASTER_ADDR=localhost
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
export LLAMA="${LLAMA:-lmsys/vicuna-7b-v1.5}"   # also read by dataset/base_dataset.py
CKPT="${CKPT:-${ROBIN3D_DATA:?}/ckpt_for_robin3d_final_performance/ckpt_01.pth}"
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
VAL_TAG="${VAL_TAG:-xrqa_val_short#xrqa_val_raw#xrcap_val#xrqa_val_short_rpos}"
ANNO=annotations/xr
CACHE_ARGS=()
[ "${PREFIX_CACHE:-0}" = 1 ] && CACHE_ARGS=(eval_prompt_cache True eval_gen_batch "${GEN_BATCH:-32}")
BATCH_SIZE="${BATCH_SIZE:-$([ "${PREFIX_CACHE:-0}" = 1 ] && echo 32 || echo 8)}"
OUT="${OUT:-outputs/xr_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  $ENV/bin/python extra_eval/xr/build_xr.py --xr_dir "${XR_DIR:?}" --scan_data "${XR_SCAN_DATA:?}" \
      --m3d_dir "${XR_M3D_DIR:?}" --feats_dir "${XR_FEATS_DIR:?}" --llama "$LLAMA" --label_db "${XR_MASK3D_DATA:?}/label_database.yaml" --anno_dir "$ANNO" \
      --splits ${SPLITS:-val} ${BUILD_ARGS:-} || exit 1
fi
echo "output_dir=$OUT val_tag=$VAL_TAG ckpt=$CKPT ngpu=$NGPU batch_size=$BATCH_SIZE prefix_cache=${PREFIX_CACHE:-0}"
# Model options = scripts/eval.sh (the released checkpoint's stage-2 run), max_txt_len aside.
$ENV/bin/torchrun --nproc_per_node="$NGPU" --master_port=$MASTER_PORT tasks/train.py scripts/xr_config.py \
    output_dir "$OUT" evaluate True pretrained_path "$CKPT" val_tag "$VAL_TAG" train_tag "" \
    wandb.enable False gpu_num "$NGPU" do_save False batch_size "$BATCH_SIZE" num_workers 4 \
    scheduler.epochs 3 optimizer.lr 1e-5 seed 42 resume False do_eval_during_train False \
    model.input_dim 1024 model.img_input_dim 1536 model.input_mask3d_dim 128 model.max_obj_num 150 \
    lora.lora_r 16 lora.lora_alpha 16 optimizer.max_grad_norm 0.01 model.max_txt_len "${MAX_TXT_LEN:-256}" \
    model.add_scene_token False model.add_pos_emb False model.add_box_emb False \
    model.add_sid_eid False model.add_objlabel False model.add_layer_norm False \
    model.add_mask_token False model.add_mask3d_token True "${CACHE_ARGS[@]}"
echo "ROBIN3D_EXIT=$?"
SCORE_ONLY=1 OUT="$OUT" bash extra_eval/xr/score.sh
echo XR_EVAL_DONE
