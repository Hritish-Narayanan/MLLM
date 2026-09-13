"""
AirLLMRuntime — RuntimeBase implementation for AirLLM.

AirLLM enables running large models on limited VRAM by loading
model layers one at a time. It's useful for models that don't fit
entirely in GPU memory.

Current status: STUB — Milestone 1 focuses on Ollama.
This adapter defines the interface but will raise NotImplementedError
for operations that are not yet implemented.

AirLLM does NOT support concurrent sessions — requests must be
serialized at the session layer.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from backend.runtime.base import (
    GenerateParams,
    GenerateResult,
    ModelHandle,
    ModelInfo,
    RuntimeBase,
    RuntimeCapabilities,
    RuntimeStatus,
    RuntimeStatusCode,
    StreamChunk,
)

logger = logging.getLogger(__name__)

# Feature flag: set to True once AirLLM is fully implemented
AIRLLM_AVAILABLE = False


class AirLLMRuntime(RuntimeBase):
    """AirLLM inference runtime (layer-by-layer model loading)."""

    def __init__(self):
        self._status = RuntimeStatus(code=RuntimeStatusCode.NOT_INITIALIZED)

    @property
    def name(self) -> str:
        return "AirLLM"

    async def initialize(self) -> None:
        if not AIRLLM_AVAILABLE:
            self._status = RuntimeStatus(
                code=RuntimeStatusCode.ERROR,
                message="AirLLM backend is not yet available.",
                detail="AirLLM support is under development. Use Ollama for now.",
            )
            raise RuntimeError(
                "AirLLM backend is not yet available.\n\n"
                "AirLLM support is planned but not yet implemented.\n"
                "Please select Ollama as your inference backend."
            )

    async def shutdown(self) -> None:
        self._status = RuntimeStatus(code=RuntimeStatusCode.STOPPED)

    async def health_check(self) -> bool:
        return AIRLLM_AVAILABLE

    async def get_status(self) -> RuntimeStatus:
        return self._status

    async def get_capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_concurrent_sessions=False,  # AirLLM must serialize
            supports_streaming=False,  # TBD
            supports_gpu=True,
            supports_quantization=False,
            supported_model_formats=["safetensors", "pytorch"],
        )

    async def load_model(self, model_id: str, options: Optional[dict] = None) -> ModelHandle:
        raise NotImplementedError("AirLLM model loading is not yet implemented")

    async def unload_model(self, handle: ModelHandle) -> None:
        raise NotImplementedError("AirLLM model unloading is not yet implemented")

    async def generate(
        self,
        handle: ModelHandle,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> GenerateResult:
        raise NotImplementedError("AirLLM generation is not yet implemented")

    async def stream(
        self,
        handle: ModelHandle,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError("AirLLM streaming is not yet implemented")
        yield  # type: ignore  # Required for async generator type

    async def get_model_info(self, model_id: str) -> ModelInfo:
        return ModelInfo(
            model_id=model_id,
            supported_by_runtime=False,
            compatibility_note="AirLLM backend is not yet available.",
        )
