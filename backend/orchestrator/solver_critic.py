"""
Solver → Critic strategy.

One agent (Solver) generates a solution.
Another agent (Critic) reviews it for correctness, hallucinations,
missing requirements, bugs, edge cases, and logical errors.
The Solver then revises based on the critique.

Configurable number of revision rounds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from backend.agent.agent import AgentRole
from backend.agent.manager import AgentManager, AgentResult
from backend.orchestrator.base import OrchestrationResult, StrategyBase
from backend.runtime.base import GenerateParams


SOLVER_SYSTEM_PROMPT = (
    "You are a problem solver. Provide clear, thorough, and accurate answers. "
    "When you receive critique, carefully consider each point and revise your answer accordingly."
)

CRITIC_SYSTEM_PROMPT = (
    "You are a critical reviewer. Your job is to examine the solution provided and identify:\n"
    "- Factual errors or hallucinations\n"
    "- Missing information or requirements\n"
    "- Logical errors or inconsistencies\n"
    "- Edge cases not considered\n"
    "- Bugs (if code is involved)\n\n"
    "Be specific and constructive. Point out what is correct and what needs improvement."
)


class SolverCriticStrategy(StrategyBase):

    def __init__(self, rounds: int = 1):
        self._rounds = max(1, rounds)

    @property
    def name(self) -> str:
        return "solver_critic"

    @property
    def description(self) -> str:
        return "Solver generates a solution, Critic reviews it, Solver revises"

    @property
    def min_agents(self) -> int:
        return 2

    @property
    def max_agents(self) -> int:
        return 2

    async def execute(
        self,
        manager: AgentManager,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> OrchestrationResult:
        agents = manager.agents
        if len(agents) < 2:
            raise RuntimeError("Solver→Critic strategy requires at least 2 agents.")

        solver = agents[0]
        critic = agents[1]

        # Assign roles and system prompts
        solver.role = AgentRole.SOLVER
        solver.name = "Solver"
        solver.system_prompt = SOLVER_SYSTEM_PROMPT

        critic.role = AgentRole.CRITIC
        critic.name = "Critic"
        critic.system_prompt = CRITIC_SYSTEM_PROMPT

        all_results: list[AgentResult] = []
        total_calls = 0
        start = time.monotonic()

        # Initial solution
        solver_result = await manager.run_agent(solver, prompt, params)
        all_results.append(solver_result)
        total_calls += 1

        current_solution = solver_result.text

        for round_num in range(self._rounds):
            # Critic reviews the solution
            critique_prompt = (
                f"The following solution was provided for the question: \"{prompt}\"\n\n"
                f"--- Solution ---\n{current_solution}\n--- End Solution ---\n\n"
                f"Please review this solution critically."
            )
            critic_result = await manager.run_agent(critic, critique_prompt, params)
            all_results.append(critic_result)
            total_calls += 1

            # Solver revises based on critique
            revision_prompt = (
                f"Original question: \"{prompt}\"\n\n"
                f"Your previous answer:\n{current_solution}\n\n"
                f"A reviewer provided this critique:\n{critic_result.text}\n\n"
                f"Please revise your answer, addressing the critique."
            )
            revision_result = await manager.run_agent(solver, revision_prompt, params)
            all_results.append(revision_result)
            total_calls += 1

            current_solution = revision_result.text

        elapsed = time.monotonic() - start

        return OrchestrationResult(
            strategy=self.name,
            agent_results=all_results,
            final_answer=current_solution,
            total_time_seconds=elapsed,
            total_model_calls=total_calls,
            rounds=self._rounds,
            metadata={"revision_rounds": self._rounds},
        )
