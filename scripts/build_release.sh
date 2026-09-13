#!/usr/bin/env bash
set -e

# Determine target
TARGET=$(rustc -Vv | grep "host:" | awk '{print $2}')
echo "==> Detected Host Target: $TARGET"

# 1. Build Python sidecar binary
echo "==> Building Python Sidecar binary with PyInstaller..."
mkdir -p src-tauri/binaries
.venv/bin/pyinstaller --noconfirm --clean --onefile \
  --name "mllm-backend-${TARGET}" \
  --distpath src-tauri/binaries \
  backend/ipc/server.py

# 2. Build Tauri Desktop Application Bundle
echo "==> Building Tauri Native Bundle..."
npx -y @tauri-apps/cli build

echo "==> Build Complete! Bundles created in src-tauri/target/release/bundle/"
