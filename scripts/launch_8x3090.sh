#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "usage: $0 DOMAIN CONFIG CORPUS OUTPUT_DIR [RESUME]" >&2
  exit 2
fi

DOMAIN="$1"
CONFIG="$2"
CORPUS="$3"
OUTPUT_DIR="$4"
RESUME_PATH="${5:-}"
ENV_DIR="${HRI_TTR_ENV_DIR:-/data/autovla/envs/hri-ttr}"
MIN_FREE_MIB="${HRI_TTR_MIN_FREE_MIB:-16000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mapfile -t FREE_MEMORY < <(
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
)
if [[ "${#FREE_MEMORY[@]}" -ne 8 ]]; then
  echo "expected exactly 8 visible physical GPUs" >&2
  exit 3
fi
for INDEX in "${!FREE_MEMORY[@]}"; do
  if (( FREE_MEMORY[INDEX] < MIN_FREE_MIB )); then
    echo "GPU ${INDEX} has ${FREE_MEMORY[INDEX]} MiB free; need ${MIN_FREE_MIB} MiB" >&2
    exit 3
  fi
done

COMMAND=(
  "${ENV_DIR}/bin/torchrun" --standalone --nproc-per-node=8
  --module hri_ttr.cli train "${DOMAIN}"
  --config "${CONFIG}" --corpus "${CORPUS}" --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${RESUME_PATH}" ]]; then
  COMMAND+=(--resume "${RESUME_PATH}")
fi
exec "${COMMAND[@]}"
