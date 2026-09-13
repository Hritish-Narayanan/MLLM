"""
HTTP client for the Ollama API.

This module handles all direct communication with the Ollama REST API.
It is ONLY imported by the OllamaRuntime adapter — no other part of
the application should use this module.

Ollama API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 300.0  # 5 minutes — model loading can be slow


@dataclass
class OllamaGenerateResponse:
    """Parsed response from Ollama /api/generate endpoint."""
    response: str
    done: bool
    model: str = ""
    total_duration: Optional[int] = None      # nanoseconds
    load_duration: Optional[int] = None       # nanoseconds
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None  # nanoseconds
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None       # nanoseconds

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.eval_count and self.eval_duration and self.eval_duration > 0:
            return self.eval_count / (self.eval_duration / 1e9)
        return None


class OllamaClient:
    """Low-level async HTTP client for Ollama."""

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST):
        self.host = host.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=10.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ----- Health -----

    async def is_running(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = await self._client.get("/", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ----- Models -----

    async def list_models(self) -> list[dict]:
        """List locally available models."""
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return data.get("models", [])

    async def show_model(self, model: str) -> dict:
        """Get detailed information about a model."""
        resp = await self._client.post("/api/show", json={"name": model})
        resp.raise_for_status()
        return resp.json()

    async def pull_model(self, model: str) -> AsyncIterator[dict]:
        """Pull a model from the Ollama registry (streaming progress)."""
        async with self._client.stream(
            "POST",
            "/api/pull",
            json={"name": model, "stream": True},
            timeout=httpx.Timeout(3600.0, connect=30.0),  # 1 hour for large models
        ) as resp:
            async for line in resp.aiter_lines():
                if line.strip():
                    yield json.loads(line)

    async def preload_model(self, model: str) -> None:
        """Preload model into memory."""
        await self._client.post("/api/generate", json={"model": model, "keep_alive": -1})

    # ----- Generation -----

    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: int = 2048,
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> OllamaGenerateResponse:
        """Non-streaming generation."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": num_predict,
            },
        }
        if system:
            payload["system"] = system
        if seed is not None:
            payload["options"]["seed"] = seed
        if stop:
            payload["options"]["stop"] = stop

        resp = await self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return OllamaGenerateResponse(
            response=data.get("response", ""),
            done=data.get("done", True),
            model=data.get("model", model),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            prompt_eval_duration=data.get("prompt_eval_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration"),
        )

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: int = 2048,
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> AsyncIterator[OllamaGenerateResponse]:
        """Streaming generation — yields partial responses."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": num_predict,
            },
        }
        if system:
            payload["system"] = system
        if seed is not None:
            payload["options"]["seed"] = seed
        if stop:
            payload["options"]["stop"] = stop

        async with self._client.stream(
            "POST", "/api/generate", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                yield OllamaGenerateResponse(
                    response=data.get("response", ""),
                    done=data.get("done", False),
                    model=data.get("model", model),
                    total_duration=data.get("total_duration"),
                    load_duration=data.get("load_duration"),
                    prompt_eval_count=data.get("prompt_eval_count"),
                    prompt_eval_duration=data.get("prompt_eval_duration"),
                    eval_count=data.get("eval_count"),
                    eval_duration=data.get("eval_duration"),
                )
