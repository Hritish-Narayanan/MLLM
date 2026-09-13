"""
MLLM Desktop Application — Entry Point.

Launches a native desktop window using pywebview.
The Python backend runs in the same process and exposes methods
to the frontend via pywebview's JavaScript-to-Python bridge.

No browser, no localhost URL, no manual server start.
The user launches this and gets a desktop window.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path

import webview

from backend.app import AppController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("mllm")

# Path to frontend files
FRONTEND_DIR = Path(__file__).parent / "src"


class JSBridge:
    """
    Bridge between JavaScript frontend and Python backend.

    pywebview exposes this object to JavaScript. The frontend calls
    methods on window.pywebview.api.* which map to methods here.

    Each method runs the async controller method in the event loop
    and returns JSON-serializable results.
    """

    def __init__(self):
        self._controller = AppController()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro):
        """Run an async coroutine in the background event loop and wait for result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=600)  # 10 min timeout for large model operations

    # ----- System -----

    def detect_system(self):
        """Detect hardware and return system info."""
        return self._controller.detect_system()

    def get_available_runtimes(self):
        """Return list of available runtimes."""
        return self._controller.get_available_runtimes_list()

    def get_strategies(self):
        """Return list of available orchestration strategies."""
        return self._controller.get_strategies()

    # ----- Runtime -----

    def select_runtime(self, runtime_name: str):
        """Select and initialize a runtime."""
        return self._run_async(self._controller.select_runtime(runtime_name))

    # ----- Model -----

    def load_model(self, model_id: str):
        """Load a model."""
        return self._run_async(self._controller.load_model(model_id))

    def get_model_info(self, model_id: str):
        """Get model metadata."""
        return self._run_async(self._controller.get_model_info(model_id))

    # ----- Agents -----

    def setup_agents(self, count: int, strategy: str = "single"):
        """Create agents for a strategy."""
        return self._controller.setup_agents(count, strategy)

    # ----- Inference -----

    def run_prompt(self, prompt: str, strategy: str = None):
        """Run a prompt through the orchestrator."""
        return self._run_async(self._controller.run_prompt(prompt, strategy))

    # ----- Benchmark -----

    def run_benchmark(self, prompt: str, strategy: str = None):
        """Run a benchmark: execute + measure."""
        return self._run_async(self._controller.run_benchmark(prompt, strategy))

    def get_benchmark_history(self):
        """Get stored benchmark results."""
        return self._controller.get_benchmark_history()

    # ----- Lifecycle -----

    def shutdown(self):
        """Clean shutdown."""
        self._run_async(self._controller.shutdown())
        self._loop.call_soon_threadsafe(self._loop.stop)


def main():
    """Launch the MLLM desktop application."""
    bridge = JSBridge()

    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        logger.error("Frontend not found at %s", index_path)
        sys.exit(1)

    window = webview.create_window(
        title="MLLM — Local Multi-Agent Platform",
        url=str(index_path),
        js_api=bridge,
        width=1200,
        height=800,
        min_size=(900, 600),
        resizable=True,
        text_select=True,
    )

    def on_closing():
        try:
            bridge.shutdown()
        except Exception:
            pass
        return True

    window.events.closing += on_closing

    webview.start(debug=True)  # debug=True enables dev tools during development


if __name__ == "__main__":
    main()
