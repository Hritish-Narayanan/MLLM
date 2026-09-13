"""
Debate strategy.

Multiple agents produce solutions and then critique each other's work
over multiple rounds. The final answer is typically the most refined
solution after all debate rounds.

Flow:
  Round 0: Each agent independently answers the prompt
  Round 1..N: Each agent sees all other agents' answers and critiques,
              then revises their own answer
  Final: The last revision from Agent A (or a merged answer)
"""

from __future__ import annotations

import time
from typing import Optional

from backend.agent.agent import AgentRole
from backend.agent.manager import AgentManager, AgentResult
from backend.orchestrator.base import OrchestrationResult, StrategyBase
from backend.runtime.base import GenerateParams


DEBATE_SYSTEM_PROMPT = (
    "You are participating in a structured debate to find the best answer. "
    "When you see other participants' answers, carefully consider their reasoning. "
    "If their points are valid, incorporate them. If you disagree, explain why with evidence. "
    "Always aim for the most accurate and complete answer."
)


class DebateStrategy(StrategyBase):

    def __init__(self, rounds: int = 2):
        self._rounds = max(1, rounds)

    @property
    def name(self) -> str:
        return "debate"

    @property
    def description(self) -> str:
        return "Agents debate and critique each other over multiple rounds"

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
            raise RuntimeError("Debate strategy requires at least 2 agents.")

        # Assign system prompts
        for i, agent in enumerate(agents):
            agent.role = AgentRole.GENERAL
            agent.name = f"Debater {chr(65 + i)}"
            agent.system_prompt = DEBATE_SYSTEM_PROMPT

        all_results: list[AgentResult] = []
        total_calls = 0
        start = time.monotonic()

        # Round 0: Independent initial answers
        current_answers: dict[str, str] = {}
        initial_results = await manager.run_agents_independent(agents, prompt, params)
        for result in initial_results:
            all_results.append(result)
            current_answers[result.agent_id] = result.text
            total_calls += 1

        # Debate rounds
        for round_num in range(self._rounds):
            new_answers: dict[str, str] = {}

            for agent in agents:
                # Build a prompt showing other agents' answers
                other_answers = []
                for other_agent in agents:
                    if other_agent.id != agent.id and other_agent.id in current_answers:
                        other_answers.append(
                            f"--- {other_agent.name}'s answer ---\n"
                            f"{current_answers[other_agent.id]}\n"
                            f"--- end ---"
                        )

                debate_prompt = (
                    f"Original question: \"{prompt}\"\n\n"
                    f"Your previous answer:\n{current_answers.get(agent.id, '')}\n\n"
                    f"Other participants' answers:\n\n"
                    + "\n\n".join(other_answers) + "\n\n"
                    f"Round {round_num + 1} of {self._rounds}: "
                    f"Consider the other answers, critique them where appropriate, "
                    f"and provide your revised answer."
                )

                result = await manager.run_agent(agent, debate_prompt, params)
                all_results.append(result)
                new_answers[agent.id] = result.text
                total_calls += 1

            current_answers = new_answers

        elapsed = time.monotonic() - start

        # Final answer is the last response from the first agent
        final_answer = current_answers.get(agents[0].id, "")

        return OrchestrationResult(
            strategy=self.name,
            agent_results=all_results,
            final_answer=final_answer,
            total_time_seconds=elapsed,
            total_model_calls=total_calls,
            rounds=self._rounds + 1,  # +1 for initial round
            metadata={"debate_rounds": self._rounds},
        )
