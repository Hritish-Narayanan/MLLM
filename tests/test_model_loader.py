"""
Tests for ModelSource parsing and resolution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model.loader import ModelSource, ModelSourceType


def test_runtime_tag():
    src = ModelSource.parse("gemma3:4b")
    assert src.source_type == ModelSourceType.RUNTIME_TAG
    assert src.identifier == "gemma3:4b"


def test_huggingface_repo():
    src = ModelSource.parse("google/gemma-4-E4B", hf_token="hf_test_123")
    assert src.source_type == ModelSourceType.HUGGINGFACE
    assert src.identifier == "google/gemma-4-E4B"
    assert src.hf_token == "hf_test_123"


def test_local_folder(tmp_path):
    d = tmp_path / "my_model"
    d.mkdir()
    src = ModelSource.parse(str(d))
    assert src.source_type == ModelSourceType.LOCAL_FOLDER
    assert src.identifier == "my_model"
    assert src.local_path == str(d.resolve())


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_runtime_tag()
    test_huggingface_repo()
    with tempfile.TemporaryDirectory() as td:
        test_local_folder(Path(td))
    print("All model loader tests passed ✓")
