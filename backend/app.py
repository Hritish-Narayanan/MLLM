"""
Application controller — central coordinator for the MLLM backend.

This is the single entry point for all UI operations. The frontend
sends commands through IPC, and this controller dispatches them to
the appropriate subsystem (runtime, agent manager, orchestrator, etc.).

The controller NEVER contains backend-specific logic.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Optional

from backend.agent.agent import Agent, AgentRole
from backend.agent.manager import AgentManager
from backend.hardware.detector import SystemInfo, detect_hardware, format_bytes
from backend.orchestrator.base import OrchestrationResult, StrategyBase
from backend.orchestrator.single import SingleStrategy
from backend.orchestrator.independent import IndependentStrategy
from backend.orchestrator.solver_critic import SolverCriticStrategy
from backend.orchestrator.debate import DebateStrategy
from backend.runtime.base import GenerateParams, RuntimeBase
from backend.runtime.registry import (
    create_runtime,
    get_available_runtimes,
    get_or_create_runtime,
    shutdown_all,
)

logger = logging.getLogger(__name__)


# Strategy registry
STRATEGIES: dict[str, type[StrategyBase]] = {
    "single": SingleStrategy,
    "independent": IndependentStrategy,
    "solver_critic": SolverCriticStrategy,
    "debate": DebateStrategy,
}


class AppController:
    """
    Main application controller.

    Manages the lifecycle: hardware detection → runtime selection →
    model loading → agent creation → orchestration → results.
    """

    def __init__(self):
        self._system_info: Optional[SystemInfo] = None
        self._runtime: Optional[RuntimeBase] = None
        self._agent_manager: Optional[AgentManager] = None
        self._current_model: Optional[str] = None
        self._current_strategy: str = "single"

    # ----- System -----

    def detect_system(self) -> dict:
        """Detect hardware and return system information."""
        self._system_info = detect_hardware()
        info = self._system_info
        return {
            "os": f"{info.os_name} {info.os_version}",
            "architecture": info.architecture,
            "cpu": info.cpu_name,
            "cpu_cores": info.cpu_cores,
            "cpu_threads": info.cpu_threads,
            "ram_total": format_bytes(info.ram_total_bytes),
            "ram_available": format_bytes(info.ram_available_bytes),
            "disk_free": format_bytes(info.disk_free_bytes),
            "gpu_vendor": info.gpu.vendor,
            "gpu_name": info.gpu.name,
            "gpu_vram": format_bytes(info.gpu.vram_bytes) if info.gpu.vram_bytes else "Unknown",
            "gpu_cuda": info.gpu.cuda_available,
            "gpu_metal": info.gpu.metal_available,
            "docker_available": info.docker_available,
            "docker_version": info.docker_version or "Not installed",
            "ollama_installed": info.ollama_installed,
            "ollama_path": info.ollama_path or "Not installed",
        }

    def get_available_runtimes_list(self) -> list[dict]:
        """Return available runtimes with availability status."""
        runtimes = get_available_runtimes()
        result = []
        for name in runtimes:
            available = True
            note = ""
            if name == "airllm":
                available = False
                note = "Not yet implemented"
            elif name == "ollama":
                if self._system_info and not self._system_info.ollama_installed:
                    note = "Ollama not installed — install from ollama.com"
                else:
                    note = "Ready"
            result.append({"name": name, "available": available, "note": note})
        return result

    # ----- Runtime -----

    async def select_runtime(self, runtime_name: str) -> dict:
        """Select and initialize a runtime."""
        try:
            self._runtime = await get_or_create_runtime(runtime_name)
            self._agent_manager = AgentManager(self._runtime)
            caps = await self._runtime.get_capabilities()
            status = await self._runtime.get_status()
            return {
                "success": True,
                "runtime": runtime_name,
                "status": status.message,
                "capabilities": {
                    "concurrent_sessions": caps.supports_concurrent_sessions,
                    "streaming": caps.supports_streaming,
                    "gpu": caps.supports_gpu,
                    "gpu_vendor": caps.gpu_vendor,
                },
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # ----- Model -----

    async def load_model(self, model_id: str) -> dict:
        """Load a model in the current runtime."""
        if not self._agent_manager:
            return {"success": False, "error": "No runtime selected. Select a runtime first."}

        try:
            handle = await self._agent_manager.load_model(model_id)
            self._current_model = model_id
            return {
                "success": True,
                "model_id": model_id,
                "message": f"Model '{model_id}' loaded successfully.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_model_info(self, model_id: str) -> dict:
        """Get model metadata."""
        if not self._runtime:
            return {"error": "No runtime selected."}
        try:
            info = await self._runtime.get_model_info(model_id)
            return {
                "model_id": info.model_id,
                "name": info.name,
                "architecture": info.architecture or "Unknown",
                "parameter_count": info.parameter_count,
                "quantization": info.quantization or "Unknown",
                "format": info.format or "Unknown",
                "context_length": info.context_length,
                "supported": info.supported_by_runtime,
                "note": info.compatibility_note,
            }
        except Exception as e:
            return {"error": str(e)}

    # ----- Agents -----

    def setup_agents(self, count: int, strategy: str = "single") -> dict:
        """Create agents for the selected strategy."""
        if not self._agent_manager:
            return {"success": False, "error": "No runtime selected."}

        self._agent_manager.clear_agents()
        self._current_strategy = strategy

        for i in range(count):
            name = f"Agent {chr(65 + i)}" if count > 1 else "Agent"
            self._agent_manager.create_agent(name=name)

        return {
            "success": True,
            "agents": [a.to_dict() for a in self._agent_manager.agents],
            "strategy": strategy,
        }

    # ----- Orchestration -----

    async def run_prompt(self, prompt: str, strategy: Optional[str] = None) -> dict:
        """
        Run a prompt through the current strategy.
        This is the main inference entry point for the UI.
        """
        if not self._agent_manager:
            return {"success": False, "error": "No runtime selected."}
        if not self._agent_manager.model_loaded:
            return {"success": False, "error": "No model loaded."}

        strat_name = strategy or self._current_strategy
        if strat_name not in STRATEGIES:
            return {"success": False, "error": f"Unknown strategy: {strat_name}"}

        # Ensure we have agents
        agents = self._agent_manager.agents
        if not agents:
            # Auto-create based on strategy
            count = 1 if strat_name == "single" else 2
            self.setup_agents(count, strat_name)

        strategy_impl = STRATEGIES[strat_name]()

        try:
            result = await strategy_impl.execute(
                self._agent_manager, prompt
            )
            return self._format_orchestration_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _format_orchestration_result(self, result: OrchestrationResult) -> dict:
        """Format an orchestration result for the frontend."""
        return {
            "success": True,
            "strategy": result.strategy,
            "final_answer": result.final_answer,
            "total_time_seconds": round(result.total_time_seconds, 3),
            "total_model_calls": result.total_model_calls,
            "rounds": result.rounds,
            "agents": [
                {
                    "agent_id": ar.agent_id,
                    "agent_name": ar.agent_name,
                    "text": ar.text,
                    "prompt_tokens": ar.prompt_tokens,
                    "completion_tokens": ar.completion_tokens,
                    "tokens_per_second": round(ar.tokens_per_second, 2) if ar.tokens_per_second else None,
                    "generation_time": round(ar.generation_time_seconds, 3),
                    "metrics_measured": ar.metrics_measured,
                }
                for ar in result.agent_results
            ],
        }

    # ----- Lifecycle -----

    async def shutdown(self) -> None:
        """Clean shutdown of all runtimes."""
        await shutdown_all()
        logger.info("Application controller shut down")

    def get_strategies(self) -> list[dict]:
        """Return available strategies."""
        result = []
        for name, cls in STRATEGIES.items():
            s = cls()
            result.append({
                "name": s.name,
                "description": s.description,
                "min_agents": s.min_agents,
                "max_agents": s.max_agents,
            })
        return result

    # ----- Benchmarking -----

    async def run_benchmark(self, prompt: str, strategy: Optional[str] = None) -> dict:
        """Run a benchmark: execute the prompt and record metrics."""
        if not self._agent_manager or not self._agent_manager.model_loaded:
            return {"success": False, "error": "No model loaded."}

        from backend.benchmark.runner import BenchmarkRunner

        strat_name = strategy or self._current_strategy
        if strat_name not in STRATEGIES:
            return {"success": False, "error": f"Unknown strategy: {strat_name}"}

        strategy_impl = STRATEGIES[strat_name]()
        runner = BenchmarkRunner()

        try:
            record = await runner.run(self._agent_manager, strategy_impl, prompt)
            return {
                "success": True,
                "record": record.to_dict(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_benchmark_history(self) -> list[dict]:
        """Get stored benchmark results."""
        from backend.benchmark.store import BenchmarkStore
        store = BenchmarkStore()
        return store.load_all()

