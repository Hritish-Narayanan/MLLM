# Release Packaging & Cross-Platform Builds

This document explains how releases are built for **Windows (`MLLM.exe` / `.msi` / `.exe` installer)**, **macOS (`.dmg` / `.app`)**, and **Linux (`.AppImage` / `.deb`)**.

---

## 1. Automated GitHub Actions CI/CD (Recommended)

The repository includes a ready-to-use GitHub Actions workflow: [`.github/workflows/release.yml`](.github/workflows/release.yml).

Whenever you push a tag (e.g. `git tag v0.1.0 && git push origin v0.1.0`) or click **Run workflow** in the GitHub Actions tab, it triggers a matrix build:

| Platform | Target Triple | Artifact Produced | Format |
|---|---|---|---|
| **Windows** | `x86_64-pc-windows-msvc` | `MLLM-Windows-x86_64` | `MLLM.exe`, NSIS installer (`.exe`), WiX MSI (`.msi`) |
| **macOS** | `aarch64-apple-darwin` / `x86_64` | `MLLM-macOS-arm64` | Standalone `.app` bundle, Disk Image (`.dmg`) |
| **Linux** | `x86_64-unknown-linux-gnu` | `MLLM-Linux-x86_64` | Standalone `.AppImage`, Debian package (`.deb`) |

Each job automatically:
1. Installs Python 3.11 and dependencies.
2. Compiles the private Python JSON-RPC server into a standalone single-binary sidecar using PyInstaller (`mllm-backend-<target>`).
3. Embeds the sidecar inside the Tauri 2 bundle.
4. Produces the native desktop installer and binaries.

---

## 2. Building Locally

### On macOS (Produces `.app` and `.dmg`)

```bash
# 1. Ensure Python 3.11 environment is active
source .venv/bin/activate
pip install pyinstaller

# 2. Run the build script
chmod +x scripts/build_release.sh
./scripts/build_release.sh
```
The output will be generated in:
- `src-tauri/target/release/bundle/macos/MLLM.app`
- `src-tauri/target/release/bundle/dmg/MLLM_0.1.0_aarch64.dmg`

### On Windows (Produces `MLLM.exe` and `.msi`)

From PowerShell on a Windows host:
```powershell
# 1. Set up Python venv
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
pip install pyinstaller

# 2. Build Python sidecar
mkdir src-tauri\binaries
pyinstaller --noconfirm --clean --onefile `
  --name "mllm-backend-x86_64-pc-windows-msvc" `
  --distpath src-tauri\binaries `
  backend\ipc\server.py

# 3. Build Tauri Windows release
npx @tauri-apps/cli build
```
The output will be in:
- `src-tauri\target\release\MLLM.exe`
- `src-tauri\target\release\bundle\nsis\MLLM_0.1.0_x64-setup.exe`
- `src-tauri\target\release\bundle\msi\MLLM_0.1.0_x64_en-US.msi`

### On Linux (Produces `.AppImage` and `.deb`)

```bash
# 1. Install prerequisites
sudo apt-get install -y libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf libjavascriptcoregtk-4.1-dev

# 2. Set up Python venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt pyinstaller

# 3. Build sidecar
mkdir -p src-tauri/binaries
pyinstaller --noconfirm --clean --onefile \
  --name "mllm-backend-x86_64-unknown-linux-gnu" \
  --distpath src-tauri/binaries \
  backend/ipc/server.py

# 4. Build Tauri Linux release
npx @tauri-apps/cli build
```
The output will be in:
- `src-tauri/target/release/bundle/appimage/mllm_0.1.0_amd64.AppImage`
- `src-tauri/target/release/bundle/deb/mllm_0.1.0_amd64.deb`
