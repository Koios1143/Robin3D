#!/bin/bash
# Finetune Robin3D's released checkpoint on XR-QA (short prompt) + XR-SceneCaption train only,
# with Robin3D's stage-2 recipe (scripts/run_stage2.sh: LoRA r16, lr 1e-5 x per-GPU batch x GPUs,
# cosine, warmup 0.1 epoch, max_grad_norm 0.01, seed 42), 3 epochs, evaluating on val after
# every epoch; then score (score.sh). Needs run_xr_feats.sh.
#   bash extra_eval/xr/run_xr_finetune.sh
# Knobs: EPOCH (3), LR (1e-5), BATCH_SIZE (8 per GPU, as stage 2), MAX_TXT_LEN (256), NGPU,
#   OUT, CKPT (start point), VAL_TAG, PREFIX_CACHE / GEN_BATCH (eval speed-up).
# Resumable: rerun with the same OUT (e.g. a Slurm requeue after preemption) and training continues
# after the last saved epoch (ckpt_<k>.pth holds model, optimizer, scheduler and epoch: Robin3D's
# own resume path), evaluating that epoch first if its predictions are missing. At most the
# interrupted epoch is lost.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
ENV="${ROBIN3D_ENV:-.venv-cu128}"
export PYTHONPATH=.:${PYTHONPATH:-}
export MASTER_PORT=$((54000 + RANDOM % 10000)) MASTER_ADDR=localhost
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
export LLAMA="${LLAMA:-lmsys/vicuna-7b-v1.5}"
CKPT="${CKPT:-${ROBIN3D_DATA:?}/ckpt_for_robin3d_final_performance/ckpt_01.pth}"
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
TRAIN_TAG="xrqa_train_short#xrcap_train"
VAL_TAG="${VAL_TAG:-xrqa_val_short#xrcap_val}"
CACHE_ARGS=()
[ "${PREFIX_CACHE:-0}" = 1 ] && CACHE_ARGS=(eval_prompt_cache True eval_gen_batch "${GEN_BATCH:-32}")
# under Slurm the job id names OUT, so a requeued job (same id) finds its checkpoints
OUT="${OUT:-outputs/xr_ft_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
mkdir -p "$OUT"
$ENV/bin/python extra_eval/xr/build_xr.py --xr_dir "${XR_DIR:?}" --scan_data "${XR_SCAN_DATA:?}" \
    --m3d_dir "${XR_M3D_DIR:?}" --feats_dir "${XR_FEATS_DIR:?}" --llama "$LLAMA" --label_db "${XR_MASK3D_DATA:?}/label_database.yaml" --anno_dir annotations/xr \
    --splits train val || exit 1
MODEL_ARGS=(model.input_dim 1024 model.img_input_dim 1536 model.input_mask3d_dim 128 model.max_obj_num 150
    lora.lora_r 16 lora.lora_alpha 16 optimizer.max_grad_norm 0.01 model.max_txt_len "${MAX_TXT_LEN:-256}"
    model.add_scene_token False model.add_pos_emb False model.add_box_emb False
    model.add_sid_eid False model.add_objlabel False model.add_layer_norm False
    model.add_mask_token False model.add_mask3d_token True "${CACHE_ARGS[@]}")
COMMON=(output_dir "$OUT" val_tag "$VAL_TAG" wandb.enable False gpu_num "$NGPU" batch_size "${BATCH_SIZE:-8}"
    num_workers 4 seed 42)
run() { $ENV/bin/torchrun --nproc_per_node="$NGPU" --master_port=$MASTER_PORT tasks/train.py scripts/xr_config.py \
    "${COMMON[@]}" "$@" "${MODEL_ARGS[@]}"; }
RESUME=False
NEXT=0
LAST=$(ls "$OUT"/ckpt_*.pth 2>/dev/null | sort | tail -1)
if [ -n "$LAST" ]; then
    k=$((10#$(basename "$LAST" .pth | sed 's/ckpt_//')))
    if ! ls "$OUT"/preds_epoch${k}_step*_xrcap_val.json >/dev/null 2>&1; then
        echo "epoch $k was saved but not evaluated: evaluating $LAST"
        # resume True: start_epoch = k + 1, so the predictions are named epoch k as during training
        run evaluate True pretrained_path "$LAST" resume True train_tag "" ||
            echo "WARNING: evaluating epoch $k failed; training continues"
    fi
    CKPT=$LAST RESUME=True NEXT=$((k + 1))
fi
echo "output_dir=$OUT train_tag=$TRAIN_TAG val_tag=$VAL_TAG ckpt=$CKPT resume=$RESUME next_epoch=$NEXT ngpu=$NGPU epoch=${EPOCH:-3} lr=${LR:-1e-5}"
rc=0
if [ "$NEXT" -lt "${EPOCH:-3}" ]; then
    run evaluate False pretrained_path "$CKPT" resume $RESUME train_tag "$TRAIN_TAG" do_save True \
        scheduler.epochs "${EPOCH:-3}" optimizer.lr "${LR:-1e-5}" do_eval_during_train True
    rc=$?
fi
echo "ROBIN3D_EXIT=$rc"
OUT="$OUT" bash extra_eval/xr/score.sh
echo XR_FT_DONE
