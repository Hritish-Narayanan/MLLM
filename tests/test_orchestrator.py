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
    assert result.agent_results[0].text != result.agent_results[1].text


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


if __name__ == "__main__":
    test_single_strategy()
    test_independent_strategy()
    test_independent_requires_two_agents()
    print("All orchestrator tests passed ✓")
