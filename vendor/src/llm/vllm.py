"""Client for a local or private vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from .base import LLMClient, LLMResponse


class VLLMClient(LLMClient):
    """Call one model served by the vLLM chat-completions API."""

    provider = "vllm-openai-compatible"

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        seed: int = 42,
        top_p: float = 0.95,
        base_url: str | None = None,
        timeout: int | None = None,
        disable_thinking: bool = True,
    ) -> None:
        if not model_name.strip():
            raise ValueError("vLLM model_name must be non-empty")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.top_p = top_p
        self.base_url = (
            base_url
            or os.environ.get("VLLM_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        ).rstrip("/")
        self.timeout = timeout or int(os.environ.get("VLLM_TIMEOUT", "900"))
        self.disable_thinking = disable_thinking
        self.api_key = os.environ.get("VLLM_API_KEY", "")

    def call(self, user_message: str) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "top_p": self.top_p,
            # Qwen3 enables thinking by default. The benchmark compares final
            # JSON formulations, so every assigned model uses non-thinking mode.
            "chat_template_kwargs": {
                "enable_thinking": not self.disable_thinking,
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to vLLM at {self.base_url}; start and health-check "
                "the assigned server first."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"vLLM request exceeded {self.timeout} seconds"
            ) from exc
        except (requests.exceptions.HTTPError, ValueError, KeyError) as exc:
            detail = getattr(response, "text", "")[:500] if "response" in locals() else ""
            raise RuntimeError(f"Invalid vLLM response: {detail}") from exc
        latency = time.perf_counter() - started

        try:
            text = body["choices"][0]["message"]["content"] or ""
            usage = body.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("vLLM response is missing content or token usage") from exc
        if not isinstance(text, str):
            raise RuntimeError("vLLM response content must be a string")

        return LLMResponse(
            raw_text=text,
            latency_s=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=self.model_name,
            provider=self.provider,
        )
