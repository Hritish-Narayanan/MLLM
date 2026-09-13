"""MLLM Runtime package — abstract interface for inference backends."""
from backend.runtime.base import (
    RuntimeBase,
    RuntimeCapabilities,
    RuntimeStatus,
    ModelHandle,
    ModelInfo,
    GenerateParams,
    GenerateResult,
    StreamChunk,
)

__all__ = [
    "RuntimeBase",
    "RuntimeCapabilities",
    "RuntimeStatus",
    "ModelHandle",
    "ModelInfo",
    "GenerateParams",
    "GenerateResult",
    "StreamChunk",
]
