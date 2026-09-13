"""
Abstract base class for all inference runtimes.

Every backend (Ollama, AirLLM, future engines) implements RuntimeBase.
The rest of the application — agents, orchestrator, UI — never imports
backend-specific code directly. They only talk through this interface.

Design decisions:
  - Async throughout: inference can be slow; blocking the event loop is unacceptable.
  - ModelHandle is an opaque token returned by load_model. The runtime decides
    what it actually is (a container reference, an internal ID, etc.).
  - RuntimeCapabilities tells the agent manager whether the runtime supports
    concurrent sessions on a single loaded model, so we can avoid duplicate weight loading.
  - GenerateResult includes token counts and timing so benchmarks can collect
    real measurements without backend-specific introspection.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional


# ---------------------------------------------------------------------------
# Data classes — shared across all runtimes
# ---------------------------------------------------------------------------

class RuntimeStatusCode(Enum):
    NOT_INITIALIZED = "not_initialized"
    STARTING = "starting"
    READY = "ready"
    LOADING_MODEL = "loading_model"
    MODEL_LOADED = "model_loaded"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class RuntimeStatus:
    code: RuntimeStatusCode
    message: str = ""
    detail: Optional[str] = None  # Technical detail for debugging


@dataclass
class RuntimeCapabilities:
    """What this runtime can do. Queried once after initialization."""
    supports_concurrent_sessions: bool = False
    supports_streaming: bool = False
    supports_gpu: bool = False
    supports_quantization: bool = False
    max_context_length: Optional[int] = None
    supported_model_formats: list[str] = field(default_factory=list)
    gpu_vendor: Optional[str] = None  # "nvidia", "apple", "amd", None


@dataclass
class ModelHandle:
    """
    Opaque reference to a loaded model.
    The runtime fills in whatever it needs. Other layers treat this as a token.
    """
    runtime_name: str
    model_id: str
    internal_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Metadata about a model, as reported by the runtime."""
    model_id: str
    name: str = ""
    architecture: str = ""
    parameter_count: Optional[int] = None
    quantization: Optional[str] = None
    format: str = ""
    context_length: Optional[int] = None
    size_bytes: Optional[int] = None
    supported_by_runtime: bool = False
    compatibility_note: str = ""


@dataclass
class GenerateParams:
    """Parameters for a single generation call."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    stop_sequences: list[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    seed: Optional[int] = None


@dataclass
class GenerateResult:
    """Result from a single generation call, including real metrics."""
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    generation_time_seconds: float = 0.0
    model_id: str = ""
    # Whether the metric values are actual measurements (True) or estimates (False)
    metrics_measured: bool = True


@dataclass
class StreamChunk:
    """A single chunk from a streaming generation."""
    text: str
    done: bool = False
    # Final chunk may include cumulative metrics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None


# ---------------------------------------------------------------------------
# Abstract Runtime
# ---------------------------------------------------------------------------

class RuntimeBase(ABC):
    """
    Abstract interface for inference backends.

    Lifecycle:
        1. __init__() — lightweight, no side effects
        2. initialize() — start containers, connect to services, etc.
        3. load_model() → ModelHandle
        4. generate() / stream() — using the handle
        5. unload_model()
        6. shutdown() — clean up containers, connections
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable runtime name, e.g. 'Ollama', 'AirLLM'."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """
        Start the runtime (e.g. launch container, connect to service).
        Called once before any model operations.
        Raises RuntimeError with a user-friendly message on failure.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Stop the runtime and release all resources."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the runtime is responsive and operational."""
        ...

    @abstractmethod
    async def get_status(self) -> RuntimeStatus:
        """Current runtime status."""
        ...

    @abstractmethod
    async def get_capabilities(self) -> RuntimeCapabilities:
        """Query runtime capabilities. Safe to call after initialize()."""
        ...

    @abstractmethod
    async def load_model(self, model_id: str, options: Optional[dict] = None) -> ModelHandle:
        """
        Load a model and return an opaque handle.
        The model_id format depends on the runtime:
          - Ollama: "gemma3:4b" or similar Ollama model tags
          - AirLLM: HuggingFace model ID like "google/gemma-4-E4B"
        Raises RuntimeError with user-friendly message on failure.
        """
        ...

    @abstractmethod
    async def unload_model(self, handle: ModelHandle) -> None:
        """Unload a previously loaded model."""
        ...

    @abstractmethod
    async def generate(
        self,
        handle: ModelHandle,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> GenerateResult:
        """
        Generate a complete response (non-streaming).
        Returns GenerateResult with text and real metrics.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        handle: ModelHandle,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Generate a response as a stream of chunks.
        The final chunk has done=True and may include cumulative metrics.
        """
        ...
        # Yield is required to make this an async generator in the type system.
        # Concrete implementations override entirely.
        yield  # pragma: no cover

    @abstractmethod
    async def get_model_info(self, model_id: str) -> ModelInfo:
        """
        Inspect a model's metadata without necessarily loading it.
        Returns ModelInfo with compatibility information.
        """
        ...
