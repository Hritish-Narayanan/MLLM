"""
Runtime registry — discovers and provides runtime implementations.

This is the single point where backend-specific imports happen.
The rest of the application asks the registry for a runtime by name
and receives a RuntimeBase instance.
"""

from __future__ import annotations

from typing import Optional

from backend.runtime.base import RuntimeBase


# Registry of available runtime classes (populated lazily)
_RUNTIME_FACTORIES: dict[str, type[RuntimeBase]] = {}
_ACTIVE_RUNTIMES: dict[str, RuntimeBase] = {}


def register_runtime(name: str, cls: type[RuntimeBase]) -> None:
    """Register a runtime implementation by name."""
    _RUNTIME_FACTORIES[name.lower()] = cls


def get_available_runtimes() -> list[str]:
    """Return names of all registered runtimes."""
    _ensure_registered()
    return list(_RUNTIME_FACTORIES.keys())


def create_runtime(name: str) -> RuntimeBase:
    """
    Create a new runtime instance by name.
    Raises KeyError if the runtime is not registered.
    """
    _ensure_registered()
    key = name.lower()
    if key not in _RUNTIME_FACTORIES:
        available = ", ".join(_RUNTIME_FACTORIES.keys()) or "(none)"
        raise KeyError(
            f"Unknown runtime '{name}'. Available runtimes: {available}"
        )
    return _RUNTIME_FACTORIES[key]()


async def get_or_create_runtime(name: str) -> RuntimeBase:
    """
    Get an existing initialized runtime or create + initialize a new one.
    Only one instance per runtime name is kept alive.
    """
    key = name.lower()
    if key in _ACTIVE_RUNTIMES:
        return _ACTIVE_RUNTIMES[key]

    runtime = create_runtime(key)
    await runtime.initialize()
    _ACTIVE_RUNTIMES[key] = runtime
    return runtime


async def shutdown_all() -> None:
    """Shutdown all active runtimes. Called during application exit."""
    for runtime in _ACTIVE_RUNTIMES.values():
        try:
            await runtime.shutdown()
        except Exception:
            pass  # Best-effort shutdown
    _ACTIVE_RUNTIMES.clear()


def _ensure_registered() -> None:
    """Lazily import and register all known runtimes."""
    if _RUNTIME_FACTORIES:
        return

    # Import each runtime adapter — these are the ONLY places where
    # backend-specific code is imported.
    try:
        from backend.runtime.ollama.runtime import OllamaRuntime
        register_runtime("ollama", OllamaRuntime)
    except ImportError:
        pass  # Ollama dependencies not available

    try:
        from backend.runtime.airllm.runtime import AirLLMRuntime
        register_runtime("airllm", AirLLMRuntime)
    except ImportError:
        pass  # AirLLM dependencies not available
