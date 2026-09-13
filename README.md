# MLLM — Local Multi-Agent Model Platform

A cross-platform desktop application for running multiple instances of local LLMs and orchestrating them to collaborate, critique, and improve each other's outputs.

## Core Hypothesis

> Multiple independent instances of a small model, combined with structured collaboration, criticism, verification, or voting, may outperform a single instance of the same model — and potentially compete with larger models at lower computational cost.

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** — install from [ollama.com](https://ollama.com) or `brew install ollama`

### Install

```bash
# Clone and enter the project
cd MLLM

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r backend/requirements.txt
```

### Run

#### Option 1: Tauri 2 Desktop Application (Production Shell)
```bash
# Launch via Tauri 2 desktop shell
npx tauri dev
```

#### Option 2: Lightweight Desktop Window (pywebview)
```bash
source .venv/bin/activate
python main.py
```

The application opens directly as a native desktop window. No browser navigation or localhost URL required.

### Standalone Releases (Windows, macOS, Linux)

To package standalone installers and binaries without requiring Python or Node on the end-user's machine:
- **macOS**: `MLLM.app` and `MLLM_0.1.0_aarch64.dmg` in `src-tauri/target/release/bundle/`
- **Windows**: `MLLM.exe`, NSIS installer, and WiX `.msi`
- **Linux**: `.AppImage` and `.deb` package

See [RELEASES.md](RELEASES.md) and the automated GitHub Actions CI/CD in [`.github/workflows/release.yml`](.github/workflows/release.yml).

To build the release locally:
```bash
npm run build:release
```

### First Time Setup

1. **System Check** — detects hardware (Apple Silicon Metal GPU / CUDA) and runtimes
2. **Select Runtime** — choose Ollama (native with Metal GPU on macOS)
3. **Load Model** — enter Hugging Face ID (`google/gemma-4-E4B`), Ollama tag (`gemma3:4b`), or local directory
4. **Chat** — enter a prompt, get a real local response
5. **Multi-Agent** — select 2 agents with Independent, Solver → Critic, or Debate strategy

## Architecture

```text
               +----------------------------------+
               |        Tauri 2 Desktop App       |
               |  (Rust Shell + Webview Frontend) |
               +-----------------+----------------+
                                 |
                          IPC (JSON-RPC)
                                 |
               +-----------------v----------------+
               |       Python 3.11+ Backend       |
               |                                  |
               |      AppController & Registry    |
               |                 |                |
               |         Runtime Interface        |
               |          /             \         |
               |     Ollama            AirLLM     |
               |   (Native/Metal)    (Container)  |
               |          \             /         |
               |           Agent Manager          |
               |                 |                |
               |            Orchestrator          |
               |                 |                |
               |   +-------------+------------+   |
               |   |             |            |   |
               | Single     Independent  Debate/  |
               |                          Critic  |
               |                 |                |
               |             Benchmark            |
               +----------------------------------+
```
        │
    ┌───┴────────────────┐
    │                    │
AgentManager        RuntimeRegistry
    │                    │
    │             ┌──────┴──────┐
    │             │             │
    │        OllamaRuntime  AirLLMRuntime
    │             │          (planned)
    │             │
Orchestrator  Model via Ollama API
    │
┌───┼───┐
│   │   │
S   I   D   (Single / Independent / Debate)
```

**Key design principle:** The orchestrator and agents NEVER import backend-specific code. They work through the `RuntimeBase` abstraction.

## Project Structure

```
MLLM/
├── main.py              # Desktop app entry point
├── src/                  # Frontend (HTML/CSS/JS)
│   ├── index.html
│   ├── styles/main.css
│   └── js/app.js
├── backend/
│   ├── app.py            # Application controller
│   ├── runtime/          # Runtime abstraction
│   │   ├── base.py       # Abstract RuntimeBase
│   │   ├── registry.py   # Runtime discovery
│   │   ├── ollama/       # Ollama adapter
│   │   └── airllm/       # AirLLM adapter (planned)
│   ├── agent/            # Agent & AgentManager
│   ├── orchestrator/     # Strategies (Single, Independent, ...)
│   ├── hardware/         # Hardware detection
│   └── benchmark/        # Benchmarking (planned)
└── tests/                # Unit tests
```

## Supported Runtimes

| Runtime | Status | GPU Support |
|---------|--------|-------------|
| Ollama  | ✓ Working | NVIDIA (CUDA), Apple Silicon (Metal) |
| AirLLM  | Planned | NVIDIA (CUDA) |

## License

MIT
