# Source this file once per shell before running uv commands from the project root:
#   . scripts/use-nonhidden-uv-env.sh
# A non-dot environment avoids macOS marking editable-install `.pth` files hidden.
UV_PROJECT_ENVIRONMENT=venv
export UV_PROJECT_ENVIRONMENT
