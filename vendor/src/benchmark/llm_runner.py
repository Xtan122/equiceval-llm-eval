"""
LLM Runner — Điều phối LLM call cho pipeline bài báo gốc.

Nhận (model_name, prompt_id, problem_name) → gọi đúng LLM client
→ parse output → trả về (ParsedFormulation | None, LLMResponse, metadata).

Các model được hỗ trợ:
  - GPT-5              → OpenAIClient   (OPENAI_API_KEY)
  - Gemini             → GeminiClient   (GEMINI_API_KEY)
  - LLaMA-3.1          → OllamaClient   (ollama pull llama3.1)
  - DeepSeek           → OllamaClient   (ollama pull deepseek-r1:7b)
  - mock               → MockClient     (offline, no key needed)
  - Bedrock-LLaMA      → BedrockClient  (meta.llama3-1-8b-instruct-v1:0)
  - Bedrock-GPT        → BedrockClient  (openai.gpt-5.6-terra)
  - Bedrock-DeepSeek   → BedrockClient  (deepseek.v3.2)
  - bedrock:<model-id> → BedrockClient  (any Bedrock model ID)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from src.llm.base import LLMClient, LLMResponse
from src.llm.mock import MockClient
from src.prompts.strategies import PROMPT_TEMPLATES
from src.prompts.problems import PROBLEM_DESCRIPTIONS
from src.benchmark.output_parser import FormulationParser, ParsedFormulation


@dataclass
class LLMGenerationConfig:
    """Config cho một lần LLM call."""
    model_name: str       # 'GPT-5' | 'Gemini' | 'LLaMA-3.1' | 'DeepSeek-Math'
    prompt_strategy: str  # 'P1' .. 'P6'
    problem_name: str     # Registry name or stable natural-source problem id
    temperature: float = 0.1
    problem_text: Optional[str] = None  # Explicit natural-language input; bypasses registry lookup
    source_id: Optional[str] = None     # Provenance id, e.g. "OptMATH:2"
    extra_instructions: Optional[str] = None  # Shared serialization contract, if needed
    max_output_tokens: int = 800
    model_revision: Optional[str] = None  # Immutable provider/Hugging Face revision
    generation_seed: int = 42
    top_p: float = 0.95

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not self.problem_name.strip():
            raise ValueError("problem_name must be non-empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.model_revision is not None and not self.model_revision.strip():
            raise ValueError("model_revision cannot be blank")
        if self.generation_seed < 0:
            raise ValueError("generation_seed must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


# ─── Factory: tạo LLM client đúng theo model_name ────────────────────────────
def _build_client(config: LLMGenerationConfig) -> LLMClient:
    name_lower = config.model_name.lower()

    if "mock" in name_lower or os.environ.get("LLM_USE_MOCK", "").lower() == "1":
        return MockClient(
            model_name=config.model_name,
            problem_name=config.problem_name,
            prompt_id=config.prompt_strategy,
        )

    if name_lower.startswith("vllm:"):
        from src.llm.vllm import VLLMClient

        served_model = config.model_name.split(":", 1)[1].strip()
        return VLLMClient(
            model_name=served_model,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            seed=config.generation_seed,
            top_p=config.top_p,
            disable_thinking=True,
        )

    # ── AWS Bedrock models (check FIRST — before gpt/llama/deepseek keywords) ───
    if "bedrock" in name_lower:
        from src.llm.bedrock import BedrockClient

        BEDROCK_ALIASES = {
            "bedrock-llama":    "meta.llama3-1-8b-instruct-v1:0",
            "bedrock-gpt":      "openai.gpt-oss-120b-1:0",   # GPT OSS 120B (accessible)
            "bedrock-gpt-20b":  "openai.gpt-oss-20b-1:0",   # GPT OSS 20B  (accessible)
            "bedrock-deepseek": "deepseek.v3.2",
        }

        if name_lower in BEDROCK_ALIASES:
            model_id = BEDROCK_ALIASES[name_lower]
        elif ":" in config.model_name and config.model_name.lower().startswith("bedrock:"):
            model_id = config.model_name.split(":", 1)[1]
        else:
            model_id = config.model_name

        return BedrockClient(
            model_id=model_id,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
        )

    if "gemini" in name_lower:
        from src.llm.gemini import GeminiClient
        # Cho phép chỉ định model cụ thể, e.g. 'gemini-1.5-pro'
        gemini_model = config.model_name if "/" in config.model_name or "-" in config.model_name.replace("Gemini", "") else "gemini-1.5-flash"
        return GeminiClient(
            model_name=gemini_model,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
        )

    if "gpt" in name_lower and "bedrock" not in name_lower:
        from src.llm.openai_gpt import OpenAIClient
        # paper dùng gpt-5; truyền thẳng nếu user chỉ định rõ, else fallback
        model_id = config.model_name if config.model_name not in ("GPT-5", "gpt-5") else "gpt-5"
        return OpenAIClient(
            model_name=model_id,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
        )

    if "llama" in name_lower or "phi" in name_lower or "mistral" in name_lower:
        from src.llm.ollama import OllamaClient
        # Truyền thẳng model_name từ config để user kiểm soát tag Ollama
        # Ví dụ: 'llama3.1:8b', 'llama3.1', 'phi3', 'mistral'
        return OllamaClient(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
        )

    if "deepseek" in name_lower and "bedrock" not in name_lower:
        from src.llm.ollama import OllamaClient
        return OllamaClient(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
        )

    # Fallback: thử dùng Ollama với model name nguyên vẹn
    # Cho phép dùng bất kỳ Ollama model nào (e.g. 'qwen2.5:7b')
    from src.llm.ollama import OllamaClient
    return OllamaClient(
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
    )



# ─── Runner chính ─────────────────────────────────────────────────────────────
class LLMRunner:
    """
    Điều phối toàn bộ Node 2–4 của pipeline:
      Node 2: lấy problem text + prompt template
      Node 3: gọi LLM API
      Node 4: parse output JSON → ParsedFormulation
    """

    @staticmethod
    def build_message(config: LLMGenerationConfig) -> Tuple[str, str]:
        """Build a validated request and report where the problem text came from."""
        prompt_text = PROMPT_TEMPLATES.get(config.prompt_strategy)
        if not prompt_text:
            raise ValueError(
                f"Unknown prompt strategy {config.prompt_strategy!r}; "
                f"available: {sorted(PROMPT_TEMPLATES)}"
            )

        if config.problem_text is not None:
            problem_text = config.problem_text.strip()
            problem_text_source = "explicit"
        else:
            problem_text = PROBLEM_DESCRIPTIONS.get(config.problem_name, "").strip()
            problem_text_source = "registry"
        if not problem_text:
            raise ValueError(
                f"No problem text for {config.problem_name!r}. Pass problem_text explicitly "
                "for a natural-source problem."
            )
        parts = [problem_text]
        if config.extra_instructions is not None:
            extra_instructions = config.extra_instructions.strip()
            if not extra_instructions:
                raise ValueError("extra_instructions cannot be blank")
            parts.append(extra_instructions)
        parts.append(prompt_text)
        return "\n\n".join(parts), problem_text_source

    def run(
        self,
        config: LLMGenerationConfig,
    ) -> Tuple[Optional[ParsedFormulation], LLMResponse, Dict[str, Any]]:
        """
        Returns:
            formulation: ParsedFormulation nếu parse thành công, else None
            response:    LLMResponse (latency, tokens, raw_text)
            metadata:    dict với model, prompt, problem, parse_success, ...
        """
        # Validate and compose the request before constructing a network client.
        full_message, problem_text_source = self.build_message(config)
        client = _build_client(config)

        # LLM call
        response = client.call(full_message)

        # Parse output
        formulation = FormulationParser.parse(response.raw_text)

        metadata: Dict[str, Any] = {
            "model_name":     config.model_name,
            "prompt_strategy": config.prompt_strategy,
            "problem_name":   config.problem_name,
            "source_id":      config.source_id,
            "problem_text_source": problem_text_source,
            "problem_text_sha256": hashlib.sha256(
                (config.problem_text or PROBLEM_DESCRIPTIONS[config.problem_name])
                .strip()
                .encode("utf-8")
            ).hexdigest(),
            "request_sha256": hashlib.sha256(full_message.encode("utf-8")).hexdigest(),
            "extra_instructions_sha256": (
                hashlib.sha256(config.extra_instructions.strip().encode("utf-8")).hexdigest()
                if config.extra_instructions is not None
                else None
            ),
            "max_output_tokens": config.max_output_tokens,
            "model_revision": config.model_revision,
            "generation_seed": config.generation_seed,
            "top_p": config.top_p,
            "provider":       response.provider,
            "parse_success":  formulation is not None,
        }

        return formulation, response, metadata
