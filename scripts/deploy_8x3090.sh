#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${HRI_TTR_REMOTE_HOST:-10.24.116.79}"
REMOTE_PORT="${HRI_TTR_REMOTE_PORT:-1172}"
REMOTE_USER="${HRI_TTR_REMOTE_USER:-autovla}"
REMOTE_PROJECT="${HRI_TTR_REMOTE_PROJECT:-/data/autovla/projects/HRI-TTR}"
LOCAL_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

ssh -p "${REMOTE_PORT}" "${REMOTE}" "mkdir -p '${REMOTE_PROJECT}'"
rsync -az --info=stats2 \
  -e "ssh -p ${REMOTE_PORT}" \
  --exclude='.git/' \
  --exclude='.debug-journal.md' \
  --exclude='.python-version' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='runs/***' \
  --exclude='checkpoints/**/*.pt' \
  --exclude='checkpoints/**/*.pth' \
  --exclude='checkpoints/**/*.ckpt' \
  --exclude='checkpoints/**/*.safetensors' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  --exclude='__pycache__/' \
  "${LOCAL_PROJECT}/" "${REMOTE}:${REMOTE_PROJECT}/"
ssh -p "${REMOTE_PORT}" "${REMOTE}" \
  "cd '${REMOTE_PROJECT}' && test -d .git || git init"
ssh -p "${REMOTE_PORT}" "${REMOTE}" \
  "cd '${REMOTE_PROJECT}' && bash scripts/bootstrap_8x3090.sh"
