"""
Majority Vote strategy.

Multiple agents receive the same prompt independently.
A voting/synthesizing judge agent compares all answers, extracts commonalities/consensus,
and determines the majority or most coherent answer.

Best suited for tasks with verifiable or comparable answers.
"""

from __future__ import annotations

import time
from typing import Optional

from backend.agent.agent import AgentRole
from backend.agent.manager import AgentManager, AgentResult
from backend.orchestrator.base import OrchestrationResult, StrategyBase
from backend.runtime.base import GenerateParams

VOTER_SYSTEM_PROMPT = (
    "You are an independent reasoning agent. Analyze the question carefully "
    "and provide your clearest, most accurate answer with rationale."
)

JUDGE_VOTE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating multiple candidate answers to the same question. "
    "Analyze where the candidates agree and disagree. Determine the majority consensus or "
    "synthesize the most correct answer based on logical consistency, correctness, and evidence. "
    "State the consensus clearly, note any dissenting views, and output the final validated answer."
)


class MajorityVoteStrategy(StrategyBase):
    """
    Multiple independent answers synthesized/voted by a judge.
    """

    def __init__(self, agent_count: int = 3):
        self._agent_count = max(2, agent_count)

    @property
    def name(self) -> str:
        return "majority_vote"

    @property
    def description(self) -> str:
        return "Multiple independent agents generate candidate answers, voted/synthesized by consensus"

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
            raise RuntimeError("Majority Vote strategy requires at least 2 candidate agents.")

        for i, agent in enumerate(agents):
            agent.role = AgentRole.GENERAL
            agent.name = f"Agent {chr(65 + i)}"
            agent.system_prompt = VOTER_SYSTEM_PROMPT

        start = time.monotonic()
        all_results: list[AgentResult] = []
        total_calls = 0

        # Run candidate agents independently
        candidate_results = await manager.run_agents_independent(agents, prompt, params)
        for res in candidate_results:
            all_results.append(res)
            total_calls += 1

        # Format candidates for voting / synthesis
        candidates_text = []
        for i, res in enumerate(candidate_results):
            candidates_text.append(f"### Candidate {chr(65 + i)} ({res.agent_name})\n{res.text}\n")

        judge_prompt = (
            f"Original Question:\n\"{prompt}\"\n\n"
            f"Candidate Answers:\n\n"
            + "\n".join(candidates_text) + "\n\n"
            f"Task:\n"
            f"1. Compare all candidate answers above.\n"
            f"2. Identify the majority agreement / consensus among them.\n"
            f"3. Note any minority errors or discrepancies.\n"
            f"4. Provide the final, verified consensus answer."
        )

        # Create or assign Judge agent
        judge_agent = manager.create_agent(
            name="Judge (Consensus)",
            role=AgentRole.JUDGE,
            system_prompt=JUDGE_VOTE_SYSTEM_PROMPT,
        )
        judge_result = await manager.run_agent(judge_agent, judge_prompt, params)
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
                "voting_method": "consensus_judge",
            },
        )
