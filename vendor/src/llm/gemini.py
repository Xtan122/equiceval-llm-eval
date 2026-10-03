"""
Gemini LLM Client — Google Gemini API (free tier: gemini-1.5-flash).

Setup:
    pip install google-generativeai python-dotenv
    export GEMINI_API_KEY="your_key"   # or set in .env
"""

import os
import time

from .base import LLMClient, LLMResponse


class GeminiClient(LLMClient):
    """
    Calls Google Gemini API.
    Default model: gemini-1.5-flash (free tier, 15 RPM).
    Paper equivalent: closest open model to GPT-5 reasoning quality.
    """

    provider = "gemini"

    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite",
        temperature: float = 0.1,
        max_tokens: int = 800,
    ):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Cài thư viện: pip install google-generativeai python-dotenv"
            )

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY chưa được set. "
                "Thêm vào .env hoặc export GEMINI_API_KEY=..."
            )

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

    def call(self, user_message: str) -> LLMResponse:
        t0 = time.time()
        response = self._model.generate_content(user_message)
        latency = time.time() - t0

        text = response.text or ""

        # Token counts từ usage_metadata (có trên gemini-1.5+)
        meta = getattr(response, "usage_metadata", None)
        in_tokens  = getattr(meta, "prompt_token_count",      len(user_message.split()))
        out_tokens = getattr(meta, "candidates_token_count",  len(text.split()))

        return LLMResponse(
            raw_text=text,
            latency_s=latency,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model_name=self.model_name,
            provider=self.provider,
        )
