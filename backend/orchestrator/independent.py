"""
Independent strategy — multiple agents answer independently.

Each agent receives the same prompt and generates its own answer.
No collaboration or critique. The user sees all outputs side-by-side.
"""

from __future__ import annotations

import time
from typing import Optional

from backend.agent.manager import AgentManager
from backend.orchestrator.base import OrchestrationResult, StrategyBase
from backend.runtime.base import GenerateParams


class IndependentStrategy(StrategyBase):

    @property
    def name(self) -> str:
        return "independent"

    @property
    def description(self) -> str:
        return "Multiple agents independently answer the same prompt"

    @property
    def min_agents(self) -> int:
        return 2

    @property
    def max_agents(self) -> int:
        return 0  # unlimited

    async def execute(
        self,
        manager: AgentManager,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> OrchestrationResult:
        agents = manager.agents
        if len(agents) < 2:
            raise RuntimeError(
                f"Independent strategy requires at least 2 agents, got {len(agents)}."
            )

        start = time.monotonic()
        results = await manager.run_agents_independent(agents, prompt, params)
        elapsed = time.monotonic() - start

        return OrchestrationResult(
            strategy=self.name,
            agent_results=list(results),
            final_answer=None,  # No single final answer — user compares outputs
            total_time_seconds=elapsed,
            total_model_calls=len(agents),
            rounds=1,
        )
