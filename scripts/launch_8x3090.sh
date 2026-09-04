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
NPROC_PER_NODE="${HRI_TTR_NPROC_PER_NODE:-8}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
fi
export CUDA_VISIBLE_DEVICES
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=',' read -r -a SELECTED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#SELECTED_GPUS[@]}" -ne "${NPROC_PER_NODE}" ]]; then
  echo "CUDA_VISIBLE_DEVICES count must equal HRI_TTR_NPROC_PER_NODE" >&2
  exit 3
fi
for GPU in "${SELECTED_GPUS[@]}"; do
  FREE_MEMORY="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  if (( FREE_MEMORY < MIN_FREE_MIB )); then
    echo "GPU ${GPU} has ${FREE_MEMORY} MiB free; need ${MIN_FREE_MIB} MiB" >&2
    exit 3
  fi
done

COMMAND=(
  "${ENV_DIR}/bin/torchrun" --standalone
)
if [[ "${NPROC_PER_NODE}" == 8 ]]; then
  COMMAND+=(--nproc-per-node=8)
else
  COMMAND+=("--nproc-per-node=${NPROC_PER_NODE}")
fi
COMMAND+=(
  --module hri_ttr.cli train "${DOMAIN}"
  --config "${CONFIG}" --corpus "${CORPUS}" --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${RESUME_PATH}" ]]; then
  COMMAND+=(--resume "${RESUME_PATH}")
fi
exec "${COMMAND[@]}"
