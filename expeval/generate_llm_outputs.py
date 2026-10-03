"""Generate LLM candidate formulations for the four reference problems.

Produces the pair set consumed by ``expeval.label_oracle`` and
``expeval.run_comparison``. This is the only module that calls a real LLM
(AWS Bedrock / Ollama / OpenAI / Gemini / vLLM via the vendored runner).

Usage:
    PYTHONPATH=. python -m expeval.generate_llm_outputs \
        --model bedrock-gpt --problem Knapsack Diet --prompt P1 P2 \
        --output data/llm_outputs_raw.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.benchmark.llm_ir import parsed_to_canonical_ir
from src.benchmark.llm_runner import LLMGenerationConfig, LLMRunner
from src.benchmark.reference_ir import REFERENCE_IR_REGISTRY, get_reference_ir
from src.benchmark.verified_evaluation import ir_to_dict
from src.prompts.strategies import PROMPT_TEMPLATES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", default=["bedrock-gpt"])
    parser.add_argument("--problem", nargs="+", default=list(REFERENCE_IR_REGISTRY))
    parser.add_argument("--prompt", nargs="+", default=list(PROMPT_TEMPLATES))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output already exists; choose a new path")

    runner = LLMRunner()
    pairs, log = [], []
    for problem in args.problem:
        reference = get_reference_ir(problem)
        for model in args.model:
            for prompt in args.prompt:
                pair_id = f"llm_{model}_{problem}_{prompt}".replace("/", "_")
                formulation, response, _ = runner.run(LLMGenerationConfig(
                    model_name=model, prompt_strategy=prompt, problem_name=problem,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens))
                candidate = parsed_to_canonical_ir(formulation, problem)
                entry = {"pair_id": pair_id, "model": model, "problem": problem,
                         "prompt": prompt, "provider": response.provider,
                         "output_tokens": response.output_tokens,
                         "parse_ok": candidate is not None}
                log.append(entry)
                if candidate is not None:
                    pairs.append({
                        "pair_id": pair_id, "family": problem,
                        "reference_ir": ir_to_dict(reference),
                        "candidate_ir": ir_to_dict(candidate),
                        "label": None,
                    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation": log, "pairs": pairs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(pairs)} parsed pairs ({len(log)} calls) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
