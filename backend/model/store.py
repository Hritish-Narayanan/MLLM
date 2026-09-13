"""
Persistent model library store.

Tracks imported models, their metadata, runtime mappings, and status.
Ensures models persist across app launches without requiring terminal commands.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path.home() / ".mllm"
MODELS_FILE = DEFAULT_MODELS_DIR / "models.json"


@dataclass
class ModelMetadata:
    id: str
    name: str
    runtime: str
    status: str = "Ready"  # "Ready", "Downloading", "Error", "Available"
    architecture: str = "Unknown"
    parameter_count: Optional[int] = None
    parameter_str: str = "Unknown"
    context_length: Optional[int] = None
    format: str = "Unknown"
    quantization: str = "Unknown"
    source_type: str = "huggingface"
    storage_size: str = "Unknown"
    memory_estimate: str = "Unknown"
    expected_perf: str = "Fast"
    supported_runtimes: list[str] = None
    created_at: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.supported_runtimes is None:
            d["supported_runtimes"] = ["ollama", "airllm"]
        return d


class ModelStore:
    def __init__(self, store_path: Optional[Path] = None):
        self._path = store_path or MODELS_FILE
        self._models: dict[str, ModelMetadata] = {}
        self._init_store()

    def _init_store(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for m in data:
                        meta = ModelMetadata(**m)
                        self._models[meta.id] = meta
            else:
                # Seed with default first target: Gemma 4 E4B
                default_gemma = ModelMetadata(
                    id="google/gemma-4-E4B",
                    name="Gemma 4 E4B",
                    runtime="ollama",
                    status="Ready",
                    architecture="Gemma",
                    parameter_count=4500000000,
                    parameter_str="4.5B effective",
                    context_length=131072,
                    format="Safetensors",
                    quantization="FP16 / Q4_K_M",
                    source_type="huggingface",
                    storage_size="9.4 GB",
                    memory_estimate="~6.8 GB RAM / Metal Unified",
                    expected_perf="High (Metal Accelerated)",
                    supported_runtimes=["ollama", "airllm"],
                )
                self._models[default_gemma.id] = default_gemma
                self._save()
        except Exception as e:
            logger.warning("Failed to load models store: %s", e)

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump([m.to_dict() for m in self._models.values()], f, indent=2)
        except Exception as e:
            logger.error("Failed to save models store: %s", e)

    def list_all(self) -> list[dict]:
        return [m.to_dict() for m in self._models.values()]

    def get(self, model_id: str) -> Optional[ModelMetadata]:
        return self._models.get(model_id)

    def add_or_update(self, meta: ModelMetadata) -> dict:
        self._models[meta.id] = meta
        self._save()
        return meta.to_dict()

    def remove(self, model_id: str, delete_files: bool = False) -> bool:
        if model_id in self._models:
            del self._models[model_id]
            self._save()
            return True
        return False
