"""
Ollama Local LLM Client — dùng cho LLaMA-3.1 và DeepSeek-Math.

Ollama chạy local, không cần API key.

Setup:
    1. Cài Ollama: https://ollama.com/download
    2. Pull models:
         ollama pull llama3.1        # LLaMA-3.1-8B-Instruct
         ollama pull deepseek-r1     # DeepSeek-Math-7B (hoặc deepseek-math)
    3. (Optional) set OLLAMA_BASE_URL trong .env nếu dùng remote host

Paper models:
    - LLaMA-3.1   → model_name="llama3.1"
    - DeepSeek    → model_name="deepseek-r1:7b"  hoặc "deepseek-math"
"""

import os
import time
from typing import Optional

import requests

from .base import LLMClient, LLMResponse


class OllamaClient(LLMClient):
    """
    Gọi Ollama REST API để chạy model local.
    Dùng cho LLaMA-3.1-8B-Instruct và DeepSeek-Math-7B.
    """

    provider = "ollama"

    def __init__(
        self,
        model_name: str = "llama3.1",
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 600,
    ):
        self.model_name  = model_name
        self.base_url    = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        # Ưu tiên: tham số __init__ → env OLLAMA_TIMEOUT → default 600s
        # (llama3.1:8b trên CPU có thể mất vài phút để sinh formulation dài)
        env_timeout = int(os.environ.get("OLLAMA_TIMEOUT", "600"))
        self.timeout     = timeout if timeout != 600 else env_timeout


    def call(self, user_message: str) -> LLMResponse:
        payload = {
            "model":  self.model_name,
            "prompt": user_message,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Không kết nối được Ollama tại {self.base_url}. "
                "Đảm bảo Ollama đang chạy: ollama serve"
            )
        latency = time.time() - t0

        data       = resp.json()
        text       = data.get("response", "")
        in_tokens  = data.get("prompt_eval_count", len(user_message.split()))
        out_tokens = data.get("eval_count",         len(text.split()))

        return LLMResponse(
            raw_text=text,
            latency_s=latency,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model_name=self.model_name,
            provider=self.provider,
        )
