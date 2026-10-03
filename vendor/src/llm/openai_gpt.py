"""
OpenAI GPT Client — OpenAI API.

Dùng cho GPT-5 (paper gốc) hoặc gpt-4o / gpt-4o-mini (fallback).

Setup:
    pip install openai python-dotenv
    export OPENAI_API_KEY="sk-..."   # or set in .env
"""

import os
import time

from .base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """
    Calls OpenAI Chat Completions API.
    Paper dùng: GPT-5 (model_name="gpt-5")
    Fallback: gpt-4o-mini (rẻ hơn, vẫn mạnh)
    """

    provider = "openai"

    def __init__(
        self,
        model_name: str = "gpt-5",
        temperature: float = 0.1,
        max_tokens: int = 800,
    ):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError("Cài thư viện: pip install openai python-dotenv")

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY chưa được set. "
                "Thêm vào .env hoặc export OPENAI_API_KEY=sk-..."
            )

        self._client   = _OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def call(self, user_message: str) -> LLMResponse:
        t0 = time.time()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_message}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.time() - t0

        text       = response.choices[0].message.content or ""
        in_tokens  = response.usage.prompt_tokens
        out_tokens = response.usage.completion_tokens

        return LLMResponse(
            raw_text=text,
            latency_s=latency,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model_name=self.model_name,
            provider=self.provider,
        )
