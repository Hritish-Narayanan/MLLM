"""
Model source loading and ingestion.

Supports:
1. Hugging Face repository IDs (e.g. google/gemma-4-E4B, with optional HF token/auth)
2. Local directories containing weights/GGUF/safetensors
3. Existing runtime models/tags (e.g. gemma3:4b)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class ModelSourceType(str, Enum):
    RUNTIME_TAG = "runtime_tag"
    HUGGINGFACE = "huggingface"
    LOCAL_FOLDER = "local_folder"


@dataclass
class ModelSource:
    source_type: ModelSourceType
    identifier: str
    hf_token: Optional[str] = None
    local_path: Optional[str] = None

    @classmethod
    def parse(cls, input_str: str, hf_token: Optional[str] = None) -> ModelSource:
        input_str = input_str.strip()

        # Check if local folder or file
        path = Path(os.path.expanduser(input_str))
        if path.exists():
            return cls(
                source_type=ModelSourceType.LOCAL_FOLDER,
                identifier=path.name,
                local_path=str(path.resolve()),
            )

        # Check if HuggingFace repository ID format: owner/repo
        if "/" in input_str and not input_str.startswith(("http://", "https://", "./", "../")):
            # Known HF ID
            return cls(
                source_type=ModelSourceType.HUGGINGFACE,
                identifier=input_str,
                hf_token=hf_token or os.environ.get("HF_TOKEN"),
            )

        # Default to runtime model tag (e.g. gemma3:4b, llama3.2)
        return cls(
            source_type=ModelSourceType.RUNTIME_TAG,
            identifier=input_str,
        )
