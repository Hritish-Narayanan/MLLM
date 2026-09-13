"""
Tests for orchestration strategies.
Uses a mock runtime to test without actual inference.
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.runtime.base import (
    RuntimeBase,
    RuntimeCapabilities,
    RuntimeStatus,
    RuntimeStatusCode,
    ModelHandle,
    ModelInfo,
    GenerateParams,
    GenerateResult,
    StreamChunk,
)
from backend.agent.manager import AgentManager
from backend.orchestrator.single import SingleStrategy
from backend.orchestrator.independent import IndependentStrategy


class MockRuntime(RuntimeBase):
    """A fake runtime for testing orchestration logic."""

    def __init__(self):
        self._call_count = 0

    @property
    def name(self) -> str:
        return "Mock"

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def health_check(self) -> bool:
        return True

    async def get_status(self):
        return RuntimeStatus(code=RuntimeStatusCode.READY)

    async def get_capabilities(self):
        return RuntimeCapabilities(
            supports_concurrent_sessions=True,
            supports_streaming=False,
        )

    async def load_model(self, model_id, options=None):
        return ModelHandle(runtime_name="Mock", model_id=model_id)

    async def unload_model(self, handle):
        pass

    async def generate(self, handle, prompt, params=None):
        self._call_count += 1
        return GenerateResult(
            text=f"Mock response #{self._call_count} for: {prompt[:30]}",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            tokens_per_second=100.0,
            generation_time_seconds=0.1,
            model_id=handle.model_id,
            metrics_measured=False,  # Mock data is not real
        )

    async def stream(self, handle, prompt, params=None):
        yield StreamChunk(text="mock", done=True)

    async def get_model_info(self, model_id):
        return ModelInfo(model_id=model_id, supported_by_runtime=True)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_single_strategy():
    runtime = MockRuntime()
    manager = AgentManager(runtime)
    run(manager.load_model("test-model"))
    manager.create_agent(name="Solo")

    strategy = SingleStrategy()
    result = run(strategy.execute(manager, "What is 2+2?"))

    assert result.strategy == "single"
    assert len(result.agent_results) == 1
    assert result.total_model_calls == 1
    assert result.final_answer is not None
    assert "Mock response" in result.final_answer


def test_independent_strategy():
    runtime = MockRuntime()
    manager = AgentManager(runtime)
    run(manager.load_model("test-model"))
    manager.create_agent(name="Agent A")
    manager.create_agent(name="Agent B")

    strategy = IndependentStrategy()
    result = run(strategy.execute(manager, "What is 2+2?"))

    assert result.strategy == "independent"
    assert len(result.agent_results) == 2
    assert result.total_model_calls == 2
    assert result.final_answer is None  # Independent has no single answer
    # Each agent should have a different response (different call count)
def test_independent_requires_two_agents():
    runtime = MockRuntime()
    manager = AgentManager(runtime)
    run(manager.load_model("test-model"))
    manager.create_agent(name="Solo")

    strategy = IndependentStrategy()
    try:
        run(strategy.execute(manager, "test"))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "at least 2 agents" in str(e)


def test_debate_strategy():
    from backend.orchestrator.debate import DebateStrategy
    runtime = MockRuntime()
    manager = AgentManager(runtime)
    run(manager.load_model("test-model"))
    manager.create_agent(name="Debater 1")
    manager.create_agent(name="Debater 2")

    strategy = DebateStrategy(rounds=1)
    result = run(strategy.execute(manager, "Explain quantum entanglement"))

    assert result.strategy == "debate"
    assert result.final_answer is not None
    # 2 initial + 2 debate + 1 judge = 5 model calls
    assert result.total_model_calls == 5


def test_solver_critic_strategy():
    from backend.orchestrator.solver_critic import SolverCriticStrategy
    runtime = MockRuntime()
    manager = AgentManager(runtime)
    run(manager.load_model("test-model"))
    manager.create_agent(name="Solver")
    manager.create_agent(name="Critic")

    strategy = SolverCriticStrategy(rounds=1)
    result = run(strategy.execute(manager, "Write quicksort in Python"))

    assert result.strategy == "solver_critic"
    assert result.final_answer is not None
    # 1 solver + 1 critic + 1 solver revision = 3 model calls
    assert result.total_model_calls == 3


def test_majority_vote_strategy():
    from backend.orchestrator.majority_vote import MajorityVoteStrategy
    runtime = MockRuntime()
    manager = AgentManager(runtime)
    run(manager.load_model("test-model"))
    manager.create_agent(name="Agent A")
    manager.create_agent(name="Agent B")
    manager.create_agent(name="Agent C")

    strategy = MajorityVoteStrategy()
    result = run(strategy.execute(manager, "What is the capital of France?"))

    assert result.strategy == "majority_vote"
    assert result.final_answer is not None
    # 3 candidates + 1 judge = 4 calls
    assert result.total_model_calls == 4


def test_judge_strategy():
    from backend.orchestrator.judge import JudgeStrategy
    runtime = MockRuntime()
    manager = AgentManager(runtime)
    run(manager.load_model("test-model"))
    manager.create_agent(name="Candidate 1")
    manager.create_agent(name="Candidate 2")

    strategy = JudgeStrategy()
    result = run(strategy.execute(manager, "Compare PostgreSQL vs MySQL"))

    assert result.strategy == "judge"
    assert result.final_answer is not None
    # 2 candidates + 1 judge = 3 calls
    assert result.total_model_calls == 3


if __name__ == "__main__":
    test_single_strategy()
    test_independent_strategy()
    test_independent_requires_two_agents()
    test_debate_strategy()
    test_solver_critic_strategy()
    test_majority_vote_strategy()
    test_judge_strategy()
    print("All orchestrator tests passed ✓")

