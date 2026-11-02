#!/bin/bash
# Robin3D inputs for XR-QA / XR-SceneCaption scenes (train + val), on all visible GPUs:
#   1. Mask3D proposals + query features per HM3D region     (mask3d_xr.py infer, MASK3D_ENV)
#   2. per-scene 150 best proposals + scene-frame positions  (mask3d_xr.py scenes)
#   3. Uni3D-g features + boxes of the selected proposals    (uni3d_xr.py, FEATS_ENV)
#      and DINOv2-giant features from rendered views         (videofeats_xr.py, RENDER_ENV)
# Every step resumes where it stopped.
#   bash extra_eval/xr/run_xr_feats.sh
#   REGIONS="00006-HkseAnWCgqk_sub002 ..." bash extra_eval/xr/run_xr_feats.sh   # debug subset
# Paths from env.sh (see env.sh.example): MASK3D_ENV MASK3D_ROOT MASK3D_CKPT FEATS_ENV UNI3D_CKPT
#   DINOV2, and XR_DIR (XR-QA / XR-SceneCaption), XR_SCAN_DATA (SceneVerse HM3D scan_data),
#   HM3D_GLB (HM3D v0.2 glb dir), RENDER_ENV (pyrender + transformers venv), XR_MASK3D_DATA
#   (the regions in Mask3D's processed scannet200 layout, e.g. from LEO's
#   extra_eval/xr/mask3d/prepare_regions.py), XR_M3D_DIR / XR_FEATS_DIR (outputs).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export OMP_NUM_THREADS=3
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # Lightning checkpoint (holds its config); torch >= 2.6
N="${NGPU:-$(nvidia-smi -L | wc -l)}"
M3D="${XR_M3D_DIR:?}"
FD="${XR_FEATS_DIR:?}"
SEL=()
[ -n "${REGIONS:-}" ] && SEL=(--regions $REGIONS)
M3D_ARGS=(--mask3d_root "${MASK3D_ROOT:?}" --ckpt "${MASK3D_CKPT:?}" --data_dir "${XR_MASK3D_DATA:?}" --out_dir "$M3D")
mkdir -p "$M3D" "$FD"
run_shards() {  # <log prefix> <command...>; one process per GPU, "{i}" -> shard index
  local name=$1; shift
  local pids=() rc=0
  for i in $(seq 0 $((N - 1))); do
    CUDA_VISIBLE_DEVICES=$i EGL_DEVICE_ID=$i "${@//\{i\}/$i}" > "$M3D/$name.$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  tail -n 2 "$M3D/$name".*.log
  return $rc
}
echo "== mask3d $(date)"
run_shards mask3d "${MASK3D_ENV:?}/bin/python" -u extra_eval/xr/mask3d_xr.py infer "${M3D_ARGS[@]}" \
    --shard "{i}" --num_shards "$N" "${SEL[@]}" || exit 1
echo "== scenes $(date)"
CUDA_VISIBLE_DEVICES=0 "$MASK3D_ENV/bin/python" extra_eval/xr/mask3d_xr.py scenes "${M3D_ARGS[@]}" \
    --xr_dir "${XR_DIR:?}" || exit 1
echo "== uni3d + 2D $(date)"
run_shards uni3d "${FEATS_ENV:?}/bin/python" -u extra_eval/xr/uni3d_xr.py --scan_data "${XR_SCAN_DATA:?}" \
    --m3d_dir "$M3D" --uni3d_ckpt "${UNI3D_CKPT:?}" --out_dir "$FD" --shard "{i}/$N" "${SEL[@]}" &
U=$!
PYOPENGL_PLATFORM=egl run_shards video "${RENDER_ENV:?}/bin/python" -u extra_eval/xr/videofeats_xr.py \
    --scan_data "$XR_SCAN_DATA" --glb_root "${HM3D_GLB:?}" --m3d_dir "$M3D" --dinov2 "${DINOV2:?}" \
    --out_dir "$FD" --shard "{i}/$N" "${SEL[@]}" ${DUMP_DIR:+--dump_dir "$DUMP_DIR"}
V=$?
wait $U || exit 1
[ $V = 0 ] || exit 1
echo "XR_FEATS_DONE $(date)"
