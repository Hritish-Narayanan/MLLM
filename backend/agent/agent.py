"""
Agent — represents a single reasoning entity with its own conversation state.

An agent is NOT a copy of the model weights. Multiple agents can share
the same ModelHandle through the runtime, each maintaining independent
conversation state and system prompts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AgentRole(Enum):
    SOLVER = "solver"
    CRITIC = "critic"
    JUDGE = "judge"
    GENERAL = "general"


@dataclass
class Message:
    """A single message in a conversation."""
    role: str        # "system", "user", "assistant"
    content: str
    timestamp: float = 0.0


@dataclass
class Agent:
    """
    A reasoning agent with independent conversation state.

    The agent does not own or load model weights. It holds a reference
    to a model_id and relies on the AgentManager to provide the
    runtime and model handle for inference.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Agent"
    role: AgentRole = AgentRole.GENERAL
    system_prompt: str = ""
    conversation: list[Message] = field(default_factory=list)
    model_id: str = ""

    def add_user_message(self, content: str) -> None:
        """Record a user message in conversation history."""
        import time
        self.conversation.append(Message(role="user", content=content, timestamp=time.time()))

    def add_assistant_message(self, content: str) -> None:
        """Record an assistant response in conversation history."""
        import time
        self.conversation.append(Message(role="assistant", content=content, timestamp=time.time()))

    def build_prompt(self, user_input: str) -> str:
        """
        Build the full prompt including system prompt and conversation history.
        For now, simple concatenation. Can be made template-aware later.
        """
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        # Include recent conversation for context (last 10 turns)
        recent = self.conversation[-10:]
        for msg in recent:
            if msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        parts.append(f"User: {user_input}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def clear_conversation(self) -> None:
        """Reset conversation history."""
        self.conversation.clear()

    def to_dict(self) -> dict:
        """Serialize for IPC."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "system_prompt": self.system_prompt,
            "model_id": self.model_id,
            "message_count": len(self.conversation),
        }
