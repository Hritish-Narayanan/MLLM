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
            elif method == "get_hardware_monitor":
                result = self.controller.get_hardware_monitor()
            elif method == "get_available_runtimes":
                result = self.controller.get_available_runtimes_list()
            elif method == "get_strategies":
                result = self.controller.get_strategies()
            elif method == "select_runtime":
                runtime_name = params.get("runtime_name", "ollama")
                result = await self.controller.select_runtime(runtime_name)
            elif method == "list_models":
                result = self.controller.list_models()
            elif method == "add_model_to_library":
                model_id = params.get("model_id", "")
                result = await self.controller.add_model_to_library(model_id)
            elif method == "remove_model_from_library":
                model_id = params.get("model_id", "")
                delete_files = params.get("delete_files", False)
                result = self.controller.remove_model_from_library(model_id, delete_files)
            elif method == "analyze_model":
                model_id = params.get("model_id", "")
                result = self.controller.analyze_model(model_id)
            elif method == "check_model_compatibility":
                model_id = params.get("model_id", "")
                runtime_name = params.get("runtime_name", "ollama")
                result = self.controller.check_model_compatibility(model_id, runtime_name)
            elif method == "load_model":
                model_id = params.get("model_id")
                result = await self.controller.load_model(model_id)
            elif method == "setup_agents":
                count = params.get("count", 1)
                strategy = params.get("strategy", "single")
                result = self.controller.setup_agents(count, strategy)
            elif method == "run_prompt":
                prompt = params.get("prompt", "")
                strategy = params.get("strategy")
                agent_count = params.get("agent_count")
                temperature = params.get("temperature")
                context_window = params.get("context_window")
                result = await self.controller.run_prompt(
                    prompt, strategy, agent_count, temperature, context_window
                )
            elif method == "run_baseline":
                prompt = params.get("prompt", "")
                result = await self.controller.run_baseline(prompt)
            elif method == "save_experiment":
                data = params.get("experiment", {})
                result = self.controller.save_experiment(data)
            elif method == "list_experiments":
                result = self.controller.list_experiments()
            elif method == "get_experiment":
                exp_id = params.get("id", "")
                result = self.controller.get_experiment(exp_id)
            elif method == "delete_experiment":
                exp_id = params.get("id", "")
                result = self.controller.delete_experiment(exp_id)
            elif method == "get_available_datasets":
                result = self.controller.get_available_datasets()
            elif method == "run_benchmark_suite":
                model_id = params.get("model_id", "google/gemma-4-E4B")
                configs = params.get("configurations", ["single", "independent", "debate"])
                dataset_id = params.get("dataset_id", "coding")
                runs = params.get("runs_per_config", 1)
                result = await self.controller.run_benchmark_suite(model_id, configs, dataset_id, runs)
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
