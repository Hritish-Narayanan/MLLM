"""
Single agent strategy — baseline.

One agent generates the answer. This is the simplest strategy and serves
as the control/baseline for benchmarking.
"""

from __future__ import annotations

import time
from typing import Optional

from backend.agent.manager import AgentManager
from backend.orchestrator.base import OrchestrationResult, StrategyBase
from backend.runtime.base import GenerateParams


class SingleStrategy(StrategyBase):

    @property
    def name(self) -> str:
        return "single"

    @property
    def description(self) -> str:
        return "Single agent produces the answer (baseline)"

    @property
    def min_agents(self) -> int:
        return 1

    @property
    def max_agents(self) -> int:
        return 1

    async def execute(
        self,
        manager: AgentManager,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> OrchestrationResult:
        agents = manager.agents
        if not agents:
            raise RuntimeError("No agents available. Create at least one agent.")

        agent = agents[0]
        start = time.monotonic()
        result = await manager.run_agent(agent, prompt, params)
        elapsed = time.monotonic() - start

        return OrchestrationResult(
            strategy=self.name,
            agent_results=[result],
            final_answer=result.text,
            total_time_seconds=elapsed,
            total_model_calls=1,
            rounds=1,
        )
