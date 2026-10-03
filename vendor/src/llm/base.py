"""
Abstract base class and response dataclass for all LLM clients.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Unified response object returned by every LLM client."""
    raw_text: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    model_name: str
    provider: str       # 'gemini' | 'openai' | 'ollama' | 'mock'


class LLMClient(ABC):
    """Abstract interface that every LLM backend must implement."""

    provider: str = "base"
    model_name: str = "unknown"

    @abstractmethod
    def call(self, user_message: str) -> LLMResponse:
        """
        Send user_message to the model and return a unified LLMResponse.
        temperature and max_tokens are set in __init__, not here.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
