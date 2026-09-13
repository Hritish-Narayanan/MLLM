"""
Judge strategy.

Multiple candidate agents produce solutions independently.
A designated Judge agent evaluates each solution with explicit criteria
(correctness, completeness, reasoning quality), selects the winning answer
or synthesizes a superior merged answer, and explains the judgment.
"""

from __future__ import annotations

import time
from typing import Optional

from backend.agent.agent import AgentRole
from backend.agent.manager import AgentManager, AgentResult
from backend.orchestrator.base import OrchestrationResult, StrategyBase
from backend.runtime.base import GenerateParams

CANDIDATE_SYSTEM_PROMPT = (
    "You are a skilled problem solver. Provide a complete, rigorous, and well-reasoned answer."
)

JUDGE_EVALUATION_PROMPT = (
    "You are an expert impartial Judge and evaluator. Your role is to:\n"
    "1. Rigorously evaluate each candidate solution against the original problem.\n"
    "2. Score or critique each candidate on accuracy, clarity, and edge cases.\n"
    "3. Select the best solution (or combine the best elements if advantageous).\n"
    "4. Present the rationale and the final selected solution clearly."
)


class JudgeStrategy(StrategyBase):
    """
    Multiple candidate agents evaluated by an impartial judge.
    """

    def __init__(self, judge_model: Optional[str] = None):
        self._judge_model = judge_model

    @property
    def name(self) -> str:
        return "judge"

    @property
    def description(self) -> str:
        return "Multiple candidate agents produce solutions, evaluated and selected by a Judge"

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
            raise RuntimeError("Judge strategy requires at least 2 candidate agents.")

        for i, agent in enumerate(agents):
            agent.role = AgentRole.GENERAL
            agent.name = f"Candidate {chr(65 + i)}"
            agent.system_prompt = CANDIDATE_SYSTEM_PROMPT

        start = time.monotonic()
        all_results: list[AgentResult] = []
        total_calls = 0

        # Step 1: Run candidate agents in parallel / independently
        candidate_results = await manager.run_agents_independent(agents, prompt, params)
        for res in candidate_results:
            all_results.append(res)
            total_calls += 1

        # Step 2: Format candidates for evaluation
        eval_body = []
        for i, res in enumerate(candidate_results):
            eval_body.append(
                f"=== Candidate {chr(65 + i)} ===\n{res.text}\n=== End Candidate {chr(65 + i)} ==="
            )

        judge_input = (
            f"Problem Statement:\n\"{prompt}\"\n\n"
            f"Candidate Solutions:\n\n"
            + "\n\n".join(eval_body) + "\n\n"
            f"Evaluation Instructions:\n"
            f"- Evaluate each candidate solution thoroughly.\n"
            f"- Identify flaws, omissions, or superior insights in each.\n"
            f"- Select the winning solution, explain why it won, and output the final verified answer."
        )

        # Step 3: Run Judge
        judge_agent = manager.create_agent(
            name="Judge",
            role=AgentRole.JUDGE,
            system_prompt=JUDGE_EVALUATION_PROMPT,
        )
        judge_result = await manager.run_agent(judge_agent, judge_input, params)
        all_results.append(judge_result)
        total_calls += 1

        elapsed = time.monotonic() - start

        return OrchestrationResult(
            strategy=self.name,
            agent_results=all_results,
            final_answer=judge_result.text,
            total_time_seconds=elapsed,
            total_model_calls=total_calls,
            rounds=2,
            metadata={
                "candidate_count": len(candidate_results),
                "judge_model": self._judge_model or "same_model",
            },
        )
