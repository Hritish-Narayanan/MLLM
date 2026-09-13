"""
Orchestrator base — abstract strategy interface.

Each orchestration strategy (Single, Independent, Debate, SolverCritic)
implements this interface. The orchestrator does NOT know which runtime
or backend is being used — it works purely through the AgentManager.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from backend.agent.manager import AgentManager, AgentResult
from backend.runtime.base import GenerateParams


@dataclass
class OrchestrationResult:
    """Result from an orchestration run."""
    strategy: str
    agent_results: list[AgentResult]
    final_answer: Optional[str] = None
    total_time_seconds: float = 0.0
    total_model_calls: int = 0
    rounds: int = 1
    metadata: dict = field(default_factory=dict)


class StrategyBase(ABC):
    """Abstract base for orchestration strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (e.g. 'single', 'independent', 'debate')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    @abstractmethod
    def min_agents(self) -> int:
        """Minimum number of agents required."""
        ...

    @property
    @abstractmethod
    def max_agents(self) -> int:
        """Maximum number of agents supported (0 = unlimited)."""
        ...

    @abstractmethod
    async def execute(
        self,
        manager: AgentManager,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> OrchestrationResult:
        """
        Execute the strategy.
        The manager provides agents and runtime access.
        Returns an OrchestrationResult with all individual outputs.
        """
        ...
