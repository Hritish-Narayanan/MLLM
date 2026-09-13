"""
Experiment history storage.

Saves full experiment runs including:
- model, runtime, strategy, agent count, parameters
- prompt
- individual agent responses and transcripts
- final answer
- execution metrics (time, calls, tokens, RAM/VRAM)
- baseline comparison (if run)
- timestamps
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENTS_DIR = Path.home() / ".mllm" / "experiments"


class ExperimentStore:
    def __init__(self, store_dir: Optional[Path] = None):
        self._dir = store_dir or DEFAULT_EXPERIMENTS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, data: dict) -> str:
        timestamp = int(data.get("timestamp") or time.time())
        exp_id = f"exp_{timestamp}_{data.get('strategy', 'unknown')}"
        data["id"] = exp_id
        data["timestamp"] = timestamp
        
        filepath = self._dir / f"{exp_id}.json"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return exp_id
        except Exception as e:
            logger.error("Failed to save experiment: %s", e)
            raise

    def list_experiments(self) -> list[dict]:
        experiments = []
        for p in sorted(self._dir.glob("exp_*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    experiments.append(json.load(f))
            except Exception:
                continue
        return experiments

    def get_experiment(self, exp_id: str) -> Optional[dict]:
        filepath = self._dir / f"{exp_id}.json"
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def delete_experiment(self, exp_id: str) -> bool:
        filepath = self._dir / f"{exp_id}.json"
        if filepath.exists():
            try:
                filepath.unlink()
                return True
            except Exception:
                return False
        return False
