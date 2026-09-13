"""
Tests for the Agent and AgentManager.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.agent import Agent, AgentRole, Message


def test_agent_creation():
    agent = Agent(name="Test Agent", role=AgentRole.SOLVER)
    assert agent.name == "Test Agent"
    assert agent.role == AgentRole.SOLVER
    assert len(agent.conversation) == 0


def test_agent_conversation():
    agent = Agent(name="A")
    agent.add_user_message("Hello")
    agent.add_assistant_message("Hi there")
    assert len(agent.conversation) == 2
    assert agent.conversation[0].role == "user"
    assert agent.conversation[1].role == "assistant"


def test_agent_build_prompt():
    agent = Agent(name="A", system_prompt="You are a helpful assistant.")
    prompt = agent.build_prompt("What is 2+2?")
    assert "You are a helpful assistant." in prompt
    assert "What is 2+2?" in prompt
    assert "Assistant:" in prompt


def test_agent_build_prompt_with_history():
    agent = Agent(name="A")
    agent.add_user_message("Hello")
    agent.add_assistant_message("Hi")
    prompt = agent.build_prompt("How are you?")
    assert "Hello" in prompt
    assert "Hi" in prompt
    assert "How are you?" in prompt


def test_agent_clear_conversation():
    agent = Agent(name="A")
    agent.add_user_message("Hello")
    agent.clear_conversation()
    assert len(agent.conversation) == 0


def test_agent_to_dict():
    agent = Agent(name="Agent A", role=AgentRole.CRITIC, model_id="gemma:4b")
    d = agent.to_dict()
    assert d["name"] == "Agent A"
    assert d["role"] == "critic"
    assert d["model_id"] == "gemma:4b"


def test_agent_unique_ids():
    a = Agent(name="A")
    b = Agent(name="B")
    assert a.id != b.id


if __name__ == "__main__":
    test_agent_creation()
    test_agent_conversation()
    test_agent_build_prompt()
    test_agent_build_prompt_with_history()
    test_agent_clear_conversation()
    test_agent_to_dict()
    test_agent_unique_ids()
    print("All agent tests passed ✓")
