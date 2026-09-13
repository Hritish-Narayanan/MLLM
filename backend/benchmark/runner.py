"""
Benchmark runner.

Executes structured benchmark runs: given a prompt, model, runtime,
and strategy configuration, runs the orchestrator and collects
real measurements.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from backend.agent.manager import AgentManager
from backend.benchmark.metrics import BenchmarkRecord, InferenceMetrics, measure_memory
from backend.benchmark.store import BenchmarkStore
from backend.orchestrator.base import OrchestrationResult, StrategyBase
from backend.runtime.base import GenerateParams

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmarks and records real measurements."""

    def __init__(self, store: Optional[BenchmarkStore] = None):
        self._store = store or BenchmarkStore()

    async def run(
        self,
        manager: AgentManager,
        strategy: StrategyBase,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> BenchmarkRecord:
        """
        Execute a benchmark run.
        Returns a BenchmarkRecord with all real measurements.
        """
        model_id = ""
        if manager._model_handle:
            model_id = manager._model_handle.model_id

        runtime_name = manager.runtime.name

        # Measure memory before
        mem_before = measure_memory()

        # Execute
        start_time = time.monotonic()
        result = await strategy.execute(manager, prompt, params)
        elapsed = time.monotonic() - start_time

        # Measure memory after
        mem_after = measure_memory()

        # Aggregate token metrics from all agent results
        total_prompt_tokens = 0
        total_completion_tokens = 0
        all_measured = True
        all_tps: list[float] = []

        for ar in result.agent_results:
            if ar.prompt_tokens is not None:
                total_prompt_tokens += ar.prompt_tokens
            if ar.completion_tokens is not None:
                total_completion_tokens += ar.completion_tokens
            if ar.tokens_per_second is not None:
                all_tps.append(ar.tokens_per_second)
            if not ar.metrics_measured:
                all_measured = False

        avg_tps = sum(all_tps) / len(all_tps) if all_tps else None

        metrics = InferenceMetrics(
            execution_time_seconds=elapsed,
            prompt_tokens=total_prompt_tokens or None,
            completion_tokens=total_completion_tokens or None,
            total_tokens=(total_prompt_tokens + total_completion_tokens) or None,
            tokens_per_second=avg_tps,
            memory_usage_bytes=mem_after - mem_before if mem_after > mem_before else None,
            model_calls=result.total_model_calls,
            is_measured=all_measured,
        )

        record = BenchmarkRecord(
            timestamp=time.time(),
            model_id=model_id,
            runtime=runtime_name,
            strategy=strategy.name,
            agent_count=len(manager.agents),
            prompt=prompt,
            metrics=metrics,
        )

        # Persist
        filepath = self._store.save(record)
        logger.info("Benchmark saved: %s", filepath)

        return record

    def get_history(self) -> list[dict]:
        """Get all stored benchmark results."""
        return self._store.load_all()

    def clear_history(self) -> int:
        """Clear all stored benchmarks."""
        return self._store.clear()
