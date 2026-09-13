"""
AgentManager — creates and manages agents, sharing model resources.

The manager is responsible for:
1. Creating agents with independent state
2. Providing a shared ModelHandle when the runtime supports concurrent sessions
3. Running inference for agents through the runtime abstraction
4. Never containing any backend-specific logic
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.agent.agent import Agent, AgentRole
from backend.runtime.base import (
    GenerateParams,
    GenerateResult,
    ModelHandle,
    RuntimeBase,
    RuntimeCapabilities,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from a single agent's generation."""
    agent_id: str
    agent_name: str
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    generation_time_seconds: float = 0.0
    metrics_measured: bool = True


class AgentManager:
    """
    Manages a pool of agents that share a runtime and model.

    Key design principle: if the runtime supports concurrent sessions
    (e.g., Ollama), all agents share one ModelHandle. If not (e.g., AirLLM),
    the manager serializes agent requests through the single handle.
    """

    def __init__(self, runtime: RuntimeBase):
        self._runtime = runtime
        self._agents: dict[str, Agent] = {}
        self._model_handle: Optional[ModelHandle] = None
        self._capabilities: Optional[RuntimeCapabilities] = None
        self._lock = asyncio.Lock()  # For serializing when needed

    @property
    def runtime(self) -> RuntimeBase:
        return self._runtime

    @property
    def agents(self) -> list[Agent]:
        return list(self._agents.values())

    @property
    def model_loaded(self) -> bool:
        return self._model_handle is not None

    async def load_model(self, model_id: str) -> ModelHandle:
        """Load a model into the runtime. All agents will share this handle."""
        self._capabilities = await self._runtime.get_capabilities()
        self._model_handle = await self._runtime.load_model(model_id)
        # Update all existing agents
        for agent in self._agents.values():
            agent.model_id = model_id
        logger.info(
            "Model loaded: %s (concurrent sessions: %s)",
            model_id,
            self._capabilities.supports_concurrent_sessions,
        )
        return self._model_handle

    async def unload_model(self) -> None:
        """Unload the current model."""
        if self._model_handle:
            await self._runtime.unload_model(self._model_handle)
            self._model_handle = None

    def create_agent(
        self,
        name: str = "Agent",
        role: AgentRole = AgentRole.GENERAL,
        system_prompt: str = "",
    ) -> Agent:
        """Create a new agent. It shares the loaded model with all other agents."""
        agent = Agent(
            name=name,
            role=role,
            system_prompt=system_prompt,
            model_id=self._model_handle.model_id if self._model_handle else "",
        )
        self._agents[agent.id] = agent
        logger.info("Created agent: %s (%s)", agent.name, agent.id)
        return agent

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent."""
        self._agents.pop(agent_id, None)

    def clear_agents(self) -> None:
        """Remove all agents."""
        self._agents.clear()

    async def run_agent(
        self,
        agent: Agent,
        user_input: str,
        params: Optional[GenerateParams] = None,
    ) -> AgentResult:
        """
        Run a single agent: build prompt, generate response, record in history.
        If the runtime doesn't support concurrent sessions, acquire the lock.
        """
        if not self._model_handle:
            raise RuntimeError("No model loaded. Load a model before running agents.")

        prompt = agent.build_prompt(user_input)
        p = params or GenerateParams()
        if agent.system_prompt and not p.system_prompt:
            p.system_prompt = agent.system_prompt

        # Serialize access if needed
        if self._capabilities and not self._capabilities.supports_concurrent_sessions:
            async with self._lock:
                result = await self._runtime.generate(self._model_handle, prompt, p)
        else:
            result = await self._runtime.generate(self._model_handle, prompt, p)

        # Record in agent conversation history
        agent.add_user_message(user_input)
        agent.add_assistant_message(result.text)

        return AgentResult(
            agent_id=agent.id,
            agent_name=agent.name,
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            tokens_per_second=result.tokens_per_second,
            generation_time_seconds=result.generation_time_seconds,
            metrics_measured=result.metrics_measured,
        )

    async def run_agents_independent(
        self,
        agents: list[Agent],
        user_input: str,
        params: Optional[GenerateParams] = None,
    ) -> list[AgentResult]:
        """
        Run multiple agents independently on the same prompt.
        If concurrent sessions are supported, runs in parallel.
        Otherwise, runs sequentially.
        """
        if self._capabilities and self._capabilities.supports_concurrent_sessions:
            # Parallel execution
            tasks = [self.run_agent(agent, user_input, params) for agent in agents]
            return await asyncio.gather(*tasks)
        else:
            # Sequential execution (lock handled inside run_agent)
            results = []
            for agent in agents:
                result = await self.run_agent(agent, user_input, params)
                results.append(result)
            return results
