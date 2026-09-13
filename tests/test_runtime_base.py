"""
Tests for the runtime abstraction layer.
Verifies that RuntimeBase defines the correct interface and that
data classes serialize properly.
"""

import sys
import os
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
from backend.runtime.registry import get_available_runtimes, create_runtime


def test_runtime_base_is_abstract():
    """RuntimeBase cannot be instantiated directly."""
    try:
        RuntimeBase()
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_generate_params_defaults():
    p = GenerateParams()
    assert p.temperature == 0.7
    assert p.top_p == 0.9
    assert p.max_tokens == 2048
    assert p.system_prompt is None


def test_generate_result_metrics():
    r = GenerateResult(
        text="Hello",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        tokens_per_second=12.5,
        generation_time_seconds=0.4,
        model_id="test",
        metrics_measured=True,
    )
    assert r.text == "Hello"
    assert r.total_tokens == 15
    assert r.metrics_measured is True


def test_model_handle():
    h = ModelHandle(runtime_name="test", model_id="gemma:4b")
    assert h.runtime_name == "test"
    assert h.model_id == "gemma:4b"
    assert h.metadata == {}


def test_runtime_capabilities():
    caps = RuntimeCapabilities(
        supports_concurrent_sessions=True,
        supports_streaming=True,
        supports_gpu=True,
        gpu_vendor="nvidia",
    )
    assert caps.supports_concurrent_sessions is True
    assert caps.gpu_vendor == "nvidia"


def test_available_runtimes():
    runtimes = get_available_runtimes()
    assert "ollama" in runtimes
    assert "airllm" in runtimes


def test_create_runtime_ollama():
    runtime = create_runtime("ollama")
    assert runtime.name == "Ollama"


def test_create_runtime_airllm():
    runtime = create_runtime("airllm")
    assert runtime.name == "AirLLM"


def test_create_runtime_unknown():
    try:
        create_runtime("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


if __name__ == "__main__":
    test_runtime_base_is_abstract()
    test_generate_params_defaults()
    test_generate_result_metrics()
    test_model_handle()
    test_runtime_capabilities()
    test_available_runtimes()
    test_create_runtime_ollama()
    test_create_runtime_airllm()
    test_create_runtime_unknown()
    print("All runtime tests passed ✓")
