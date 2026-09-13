"""
Benchmark metric collection.

Collects real measurements during inference runs.
All values are actual measured data — NEVER fabricated.
When a metric cannot be measured, it is recorded as None.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil


@dataclass
class InferenceMetrics:
    """Metrics from a single inference run."""
    execution_time_seconds: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    memory_usage_bytes: Optional[int] = None
    vram_usage_bytes: Optional[int] = None  # Only if measurable
    model_calls: int = 0
    is_measured: bool = True  # False if any value is estimated

    def to_dict(self) -> dict:
        return {
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2) if self.tokens_per_second else None,
            "memory_usage_bytes": self.memory_usage_bytes,
            "memory_usage_human": _format_bytes(self.memory_usage_bytes) if self.memory_usage_bytes else None,
            "vram_usage_bytes": self.vram_usage_bytes,
            "vram_usage_human": _format_bytes(self.vram_usage_bytes) if self.vram_usage_bytes else None,
            "model_calls": self.model_calls,
            "is_measured": self.is_measured,
        }


@dataclass
class BenchmarkRecord:
    """A complete benchmark record for one run."""
    timestamp: float = 0.0
    model_id: str = ""
    runtime: str = ""
    strategy: str = ""
    agent_count: int = 0
    prompt: str = ""
    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "runtime": self.runtime,
            "strategy": self.strategy,
            "agent_count": self.agent_count,
            "prompt_preview": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "metrics": self.metrics.to_dict(),
        }


def measure_memory() -> int:
    """Measure current process memory usage in bytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def measure_system_memory() -> dict:
    """Measure system-wide memory usage."""
    mem = psutil.virtual_memory()
    return {
        "total": mem.total,
        "available": mem.available,
        "used": mem.used,
        "percent": mem.percent,
    }


def _format_bytes(size_bytes: Optional[int]) -> str:
    if not size_bytes or size_bytes <= 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes = size_bytes / 1024  # type: ignore
    return f"{size_bytes:.1f} PB"
