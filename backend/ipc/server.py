"""
Sidecar JSON-RPC server for Tauri 2.

Reads JSON-RPC 2.0 requests line-by-line from stdin,
dispatches to AppController, and writes responses to stdout.
Completely isolated from any browser or network exposure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in python path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app import AppController

logger = logging.getLogger("mllm_sidecar")


class JsonRpcServer:
    def __init__(self):
        self.controller = AppController()

    async def handle_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        try:
            if method == "detect_system":
                result = self.controller.detect_system()
            elif method == "get_available_runtimes":
                result = self.controller.get_available_runtimes_list()
            elif method == "get_strategies":
                result = self.controller.get_strategies()
            elif method == "select_runtime":
                runtime_name = params.get("runtime_name", "ollama")
                result = await self.controller.select_runtime(runtime_name)
            elif method == "load_model":
                model_id = params.get("model_id")
                result = await self.controller.load_model(model_id)
            elif method == "get_model_info":
                model_id = params.get("model_id")
                result = await self.controller.get_model_info(model_id)
            elif method == "setup_agents":
                count = params.get("count", 1)
                strategy = params.get("strategy", "single")
                result = self.controller.setup_agents(count, strategy)
            elif method == "run_prompt":
                prompt = params.get("prompt", "")
                strategy = params.get("strategy")
                result = await self.controller.run_prompt(prompt, strategy)
            elif method == "run_benchmark":
                prompt = params.get("prompt", "")
                strategy = params.get("strategy")
                result = await self.controller.run_benchmark(prompt, strategy)
            elif method == "get_benchmark_history":
                result = self.controller.get_benchmark_history()
            elif method == "shutdown":
                await self.controller.shutdown()
                result = {"status": "shutdown"}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }
        except Exception as e:
            logger.exception("Error handling RPC %s", method)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }

    async def run(self):
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                break
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue

            try:
                req = json.loads(line_str)
                resp = await self.handle_request(req)
                resp_json = json.dumps(resp) + "\n"
                sys.stdout.write(resp_json)
                sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = JsonRpcServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
