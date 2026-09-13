"""
Benchmark result storage.

Persists benchmark records to JSON files so configurations can be
compared across runs. Results are stored locally, never uploaded.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from backend.benchmark.metrics import BenchmarkRecord


DEFAULT_STORE_DIR = os.path.expanduser("~/.mllm/benchmarks")


class BenchmarkStore:
    """Stores and retrieves benchmark results."""

    def __init__(self, store_dir: str = DEFAULT_STORE_DIR):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: BenchmarkRecord) -> str:
        """Save a benchmark record. Returns the filename."""
        timestamp = int(record.timestamp or time.time())
        filename = f"benchmark_{timestamp}_{record.strategy}_{record.agent_count}agents.json"
        filepath = self._dir / filename

        with open(filepath, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

        return str(filepath)

    def load_all(self) -> list[dict]:
        """Load all stored benchmark results."""
        results = []
        for filepath in sorted(self._dir.glob("benchmark_*.json")):
            try:
                with open(filepath) as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue
        return results

    def clear(self) -> int:
        """Delete all stored benchmarks. Returns count deleted."""
        count = 0
        for filepath in self._dir.glob("benchmark_*.json"):
            filepath.unlink()
            count += 1
        return count
