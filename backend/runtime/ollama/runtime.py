"""
OllamaRuntime — RuntimeBase implementation for Ollama.

Ollama natively supports concurrent requests to the same loaded model,
so multiple agents can share a single ModelHandle without loading
duplicate weights.

On macOS/Apple Silicon, the native Ollama installation uses Metal for
GPU acceleration. Docker-based Ollama on macOS only gets CPU.
This runtime detects which is available and warns accordingly.
"""

from __future__ import annotations

import logging
import shutil
import time
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
from backend.runtime.ollama.client import OllamaClient, DEFAULT_OLLAMA_HOST

logger = logging.getLogger(__name__)


class OllamaRuntime(RuntimeBase):
    """Ollama inference runtime."""

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST):
        self._host = host
        self._client: Optional[OllamaClient] = None
        self._status = RuntimeStatus(code=RuntimeStatusCode.NOT_INITIALIZED)
        self._loaded_models: dict[str, ModelHandle] = {}

    # ----- RuntimeBase properties -----

    @property
    def name(self) -> str:
        return "Ollama"

    # ----- Lifecycle -----

    async def initialize(self) -> None:
        """Connect to Ollama. Expects Ollama to be already running."""
        self._status = RuntimeStatus(code=RuntimeStatusCode.STARTING, message="Connecting to Ollama...")
        self._client = OllamaClient(self._host)

        if not await self._client.is_running():
            # Check if ollama binary exists
            ollama_path = shutil.which("ollama")
            if ollama_path:
                self._status = RuntimeStatus(
                    code=RuntimeStatusCode.ERROR,
                    message="Ollama is installed but not running.",
                    detail=f"Found at: {ollama_path}. Start it with: ollama serve",
                )
                raise RuntimeError(
                    "Ollama is installed but not running. "
                    "Please start it with 'ollama serve' or launch the Ollama application."
                )
            else:
                self._status = RuntimeStatus(
                    code=RuntimeStatusCode.ERROR,
                    message="Ollama is not installed.",
                    detail="Install from https://ollama.com or via 'brew install ollama'",
                )
                raise RuntimeError(
                    "Ollama is not installed. "
                    "Install it from https://ollama.com or run 'brew install ollama'."
                )

        self._status = RuntimeStatus(code=RuntimeStatusCode.READY, message="Connected to Ollama")
        logger.info("OllamaRuntime initialized: %s", self._host)

    async def shutdown(self) -> None:
        """Close the HTTP client. Does not stop the Ollama service."""
        if self._client:
            await self._client.close()
            self._client = None
        self._loaded_models.clear()
        self._status = RuntimeStatus(code=RuntimeStatusCode.STOPPED)
        logger.info("OllamaRuntime shut down")

    # ----- Health & Status -----

    async def health_check(self) -> bool:
        if not self._client:
            return False
        return await self._client.is_running()

    async def get_status(self) -> RuntimeStatus:
        return self._status

    async def get_capabilities(self) -> RuntimeCapabilities:
        import platform
        import subprocess

        gpu_vendor = None
        supports_gpu = False

        system = platform.system()
        if system == "Darwin":
            # macOS — check if running natively (Metal) or in Docker (CPU only)
            machine = platform.machine()
            if machine == "arm64":
                gpu_vendor = "apple"
                supports_gpu = True  # Native Ollama on Apple Silicon uses Metal
        elif system == "Linux":
            # Check for NVIDIA GPU
            try:
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_vendor = "nvidia"
                    supports_gpu = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        return RuntimeCapabilities(
            supports_concurrent_sessions=True,  # Ollama handles this natively
            supports_streaming=True,
            supports_gpu=supports_gpu,
            supports_quantization=True,
            supported_model_formats=["gguf"],
            gpu_vendor=gpu_vendor,
        )

    # ----- Model Operations -----

    async def load_model(self, model_id: str, options: Optional[dict] = None) -> ModelHandle:
        """
        'Load' a model in Ollama.
        Ollama loads models on first use and keeps them in memory.
        We trigger a generation with an empty prompt to force loading.
        """
        assert self._client is not None, "Runtime not initialized"

        self._status = RuntimeStatus(
            code=RuntimeStatusCode.LOADING_MODEL,
            message=f"Loading {model_id}...",
        )

        # Resolve model source (HF ID, local folder/file, or runtime tag)
        from backend.model.loader import ModelSource, ModelSourceType

        source = ModelSource.parse(model_id, hf_token=options.get("hf_token") if options else None)
        resolved_name = model_id

        # Check if user requested HF ID or Gemma specifically
        if source.source_type == ModelSourceType.HUGGINGFACE:
            if "gemma" in source.identifier.lower():
                # Map google/gemma-4-E4B / similar to Ollama's gemma3:4b tag
                resolved_name = "gemma3:4b"
                logger.info("Mapped HuggingFace ID %s to Ollama model tag %s", source.identifier, resolved_name)
            else:
                # Format as hf.co/org/model
                resolved_name = f"hf.co/{source.identifier}"
        elif source.source_type == ModelSourceType.LOCAL_FOLDER:
            resolved_name = source.identifier

        # Check if the model exists locally
        try:
            models = await self._client.list_models()
            local_names = [m.get("name", "") for m in models]

            found = any(
                resolved_name == n
                or f"{resolved_name}:latest" == n
                or resolved_name == n.replace(":latest", "")
                for n in local_names
            )

            if not found:
                logger.info("Model %s not found locally, pulling...", resolved_name)
                self._status = RuntimeStatus(
                    code=RuntimeStatusCode.LOADING_MODEL,
                    message=f"Pulling {resolved_name}...",
                )
                async for progress in self._client.pull_model(resolved_name):
                    status_msg = progress.get("status", "")
                    if "pulling" in status_msg or "downloading" in status_msg:
                        completed = progress.get("completed", 0)
                        total = progress.get("total", 0)
                        if total > 0:
                            pct = int(completed / total * 100)
                            self._status = RuntimeStatus(
                                code=RuntimeStatusCode.LOADING_MODEL,
                                message=f"Pulling {resolved_name}: {pct}%",
                            )
                model_id = resolved_name
            else:
                model_id = resolved_name
        except Exception as e:
            self._status = RuntimeStatus(
                code=RuntimeStatusCode.ERROR,
                message=f"Failed to load model: {model_id}",
                detail=str(e),
            )
            raise RuntimeError(
                f"Could not load model '{model_id}' in Ollama.\n\n"
                f"Possible causes:\n"
                f"• The model name may be incorrect\n"
                f"• Ollama may not support this model\n"
                f"• Network issues if the model needs to be downloaded\n\n"
                f"Technical detail: {e}"
            ) from e

        # Preload model in Ollama without generating dummy text
        try:
            await self._client.preload_model(model_id)
        except Exception as e:
            logger.warning("Preload failed (non-fatal): %s", e)

        handle = ModelHandle(
            runtime_name=self.name,
            model_id=model_id,
            internal_id=model_id,  # Ollama uses the model name directly
        )
        self._loaded_models[model_id] = handle
        self._status = RuntimeStatus(
            code=RuntimeStatusCode.MODEL_LOADED,
            message=f"Model loaded: {model_id}",
        )
        logger.info("Model loaded: %s", model_id)
        return handle

    async def unload_model(self, handle: ModelHandle) -> None:
        """Remove model from our tracking. Ollama manages its own unloading."""
        self._loaded_models.pop(handle.model_id, None)
        self._status = RuntimeStatus(code=RuntimeStatusCode.READY)
        logger.info("Model unloaded (tracking removed): %s", handle.model_id)

    # ----- Generation -----

    async def generate(
        self,
        handle: ModelHandle,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> GenerateResult:
        assert self._client is not None, "Runtime not initialized"
        p = params or GenerateParams()

        start_time = time.monotonic()

        try:
            resp = await self._client.generate(
                model=handle.model_id,
                prompt=prompt,
                system=p.system_prompt,
                temperature=p.temperature,
                top_p=p.top_p,
                top_k=p.top_k,
                num_predict=p.max_tokens,
                seed=p.seed,
                stop=p.stop_sequences or None,
            )
        except Exception as e:
            raise RuntimeError(
                f"Generation failed for model '{handle.model_id}'.\n\n"
                f"This could be caused by:\n"
                f"• Insufficient memory\n"
                f"• Model not loaded properly\n"
                f"• Ollama service crashed\n\n"
                f"Technical detail: {e}"
            ) from e

        elapsed = time.monotonic() - start_time

        return GenerateResult(
            text=resp.response,
            prompt_tokens=resp.prompt_eval_count,
            completion_tokens=resp.eval_count,
            total_tokens=(
                (resp.prompt_eval_count or 0) + (resp.eval_count or 0)
                if resp.prompt_eval_count is not None or resp.eval_count is not None
                else None
            ),
            tokens_per_second=resp.tokens_per_second,
            generation_time_seconds=elapsed,
            model_id=handle.model_id,
            metrics_measured=True,  # Ollama provides real token counts
        )

    async def stream(
        self,
        handle: ModelHandle,
        prompt: str,
        params: Optional[GenerateParams] = None,
    ) -> AsyncIterator[StreamChunk]:
        assert self._client is not None, "Runtime not initialized"
        p = params or GenerateParams()

        try:
            async for resp in self._client.generate_stream(
                model=handle.model_id,
                prompt=prompt,
                system=p.system_prompt,
                temperature=p.temperature,
                top_p=p.top_p,
                top_k=p.top_k,
                num_predict=p.max_tokens,
                seed=p.seed,
                stop=p.stop_sequences or None,
            ):
                yield StreamChunk(
                    text=resp.response,
                    done=resp.done,
                    prompt_tokens=resp.prompt_eval_count if resp.done else None,
                    completion_tokens=resp.eval_count if resp.done else None,
                    tokens_per_second=resp.tokens_per_second if resp.done else None,
                )
        except Exception as e:
            raise RuntimeError(
                f"Streaming generation failed for model '{handle.model_id}'.\n\n"
                f"Technical detail: {e}"
            ) from e

    # ----- Model Info -----

    async def get_model_info(self, model_id: str) -> ModelInfo:
        assert self._client is not None, "Runtime not initialized"

        try:
            info = await self._client.show_model(model_id)
        except Exception:
            # Model may not be locally available
            return ModelInfo(
                model_id=model_id,
                supported_by_runtime=False,
                compatibility_note="Model not found in Ollama. It may need to be pulled first.",
            )

        # Parse modelfile details
        details = info.get("details", {})
        model_info = info.get("model_info", {})

        # Try to extract context length from model_info
        context_length = None
        for key, value in model_info.items():
            if "context_length" in key and isinstance(value, (int, float)):
                context_length = int(value)
                break

        param_count = None
        param_size = details.get("parameter_size", "")
        if param_size:
            # Parse strings like "4B", "7B", "13B"
            param_size_clean = param_size.upper().strip()
            if param_size_clean.endswith("B"):
                try:
                    param_count = int(float(param_size_clean[:-1]) * 1e9)
                except ValueError:
                    pass

        return ModelInfo(
            model_id=model_id,
            name=model_id,
            architecture=details.get("family", ""),
            parameter_count=param_count,
            quantization=details.get("quantization_level", None),
            format=details.get("format", ""),
            context_length=context_length,
            supported_by_runtime=True,
        )
