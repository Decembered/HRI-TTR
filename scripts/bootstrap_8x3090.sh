#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${HRI_TTR_PROJECT_DIR:-/data/autovla/projects/HRI-TTR}"
ENV_DIR="${HRI_TTR_ENV_DIR:-/data/autovla/envs/hri-ttr}"
BIN_DIR="${HRI_TTR_BIN_DIR:-/data/autovla/bin}"
PYTHON_DIR="${HRI_TTR_PYTHON_DIR:-/data/autovla/python}"
CACHE_DIR="${HRI_TTR_UV_CACHE_DIR:-/data/autovla/cache/uv-hri-ttr}"
UV_BIN="${BIN_DIR}/uv"

mkdir -p "${BIN_DIR}" "${PYTHON_DIR}" "${CACHE_DIR}" "$(dirname "${ENV_DIR}")"
if [[ ! -x "${UV_BIN}" ]]; then
  curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="${BIN_DIR}" sh
fi
export UV_CACHE_DIR="${CACHE_DIR}"
export UV_PYTHON_INSTALL_DIR="${PYTHON_DIR}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
"${UV_BIN}" python install 3.11
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.11 "${ENV_DIR}"
fi
cd "${PROJECT_DIR}"
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY \
  -u https_proxy -u http_proxy -u all_proxy \
  UV_PROJECT_ENVIRONMENT="${ENV_DIR}" \
  "${UV_BIN}" sync --frozen --python 3.11
"${ENV_DIR}/bin/python" -c \
  'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)'
