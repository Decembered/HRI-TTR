#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HRI_TTR_PROJECT_ROOT:-/data/autovla/projects/HRI-TTR-train}"
CORPUS_ROOT="${HRI_TTR_CORPUS_ROOT:-/data/users/autovla/datasets/hri_ttr_same_motion_20hz_v1}"
PYTHON_BIN="${HRI_TTR_PYTHON:-/data/autovla/envs/hri-ttr/bin/python}"
LOG_ROOT="${PROJECT_ROOT}/runs/launcher_logs"
mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

launch_domain() {
  local domain="$1"
  local devices="$2"
  local port="$3"
  local config="$4"
  local output="$5"
  local resume_args=()
  if [[ -f "${output}/last.pt" ]]; then
    resume_args=(--resume "${output}/last.pt")
  elif [[ -f "${output}/interrupted.pt" ]]; then
    resume_args=(--resume "${output}/interrupted.pt")
  fi
  nohup env \
    -u ALL_PROXY \
    -u HTTPS_PROXY \
    -u HTTP_PROXY \
    -u all_proxy \
    -u https_proxy \
    -u http_proxy \
    CUDA_VISIBLE_DEVICES="${devices}" \
    WANDB_MODE=online \
    PYTHONPATH="${PROJECT_ROOT}/src" \
    "${PYTHON_BIN}" -m torch.distributed.run \
      --nproc-per-node=4 \
      --master-port="${port}" \
      -m hri_ttr.cli train "${domain}-vq" \
      --config "${config}" \
      --corpus "${CORPUS_ROOT}" \
      "${resume_args[@]}" \
      >"${LOG_ROOT}/${domain}.log" 2>&1 &
  printf '%s\n' "$!" >"${LOG_ROOT}/${domain}.pid"
}

launch_domain \
  human \
  0,1,2,3 \
  29611 \
  configs/human_vq/causal_scratch_4x3090_long.json \
  runs/20260903_human_causal_full_b128
launch_domain \
  g1 \
  4,5,6,7 \
  29612 \
  configs/g1_vq/causal_scratch_4x3090_long.json \
  runs/20260903_g1_causal_full_b256

printf 'human pid: %s\n' "$(<"${LOG_ROOT}/human.pid")"
printf 'g1 pid: %s\n' "$(<"${LOG_ROOT}/g1.pid")"
