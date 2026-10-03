# equiceval-llm-eval

Experiment repo comparing the **Refai & Ahmed** (RA) component-metric baseline
with the **EquiCEval** engine on **LLM-generated formulations**.

- Reference: the four ComplexOR problems (Knapsack, AircraftAssignment, Diet,
  AircraftLanding).
- Candidate: an LLM output parsed into the canonical IR.
- Label: produced independently of both methods (`expeval.label_oracle`) so that
  FPR / Recall are computed against a ground truth rather than against a method.

The method code is **vendored** (manual copy) from `../EquiCEval` and `../RA`
under `vendor/src`; the LLM layer is vendored from the monorepo. `VENDOR.lock`
pins the per-file hashes; `tests/test_vendor_manifest.py` detects drift.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Test

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

## Run

```bash
# 1. Generate candidates with a real LLM (Bedrock / Ollama / OpenAI / vLLM)
PYTHONPATH=. .venv/bin/python -m expeval.generate_llm_outputs \
  --model bedrock-gpt --problem Knapsack Diet --prompt P1 P2 \
  --output data/llm_outputs_raw.json

# 2. Label them independently (True / False / null)
PYTHONPATH=. .venv/bin/python -m expeval.label_oracle \
  --pairs data/llm_outputs_raw.json \
  --contract feasible_set_and_objective_affine \
  --output data/llm_outputs_labeled.json

# 3. Compare RA vs EquiCEval
PYTHONPATH=. .venv/bin/python -m expeval.run_comparison \
  --pairs data/llm_outputs_labeled.json \
  --contract feasible_set_and_objective_affine \
  --ra-rule primary \
  --output output/llm_eval_report.json
```

`data/sample_pairs.json` is a small labelled smoke set (not LLM output) used by
the tests and for an offline dry run.

## RA decision rule

RA has no native verdict. The default (`--ra-rule primary`) is generous to RA and
uses its most predictive signals:

```
faulty <=> Cons-R < 1 OR Cons-RMSE > tau OR optimality-gap > tau
```

`--ra-rule cons-precision` adds `Cons-P < 1` (sensitivity); `--ra-rule
solver-only` uses only the optimality gap.

## Layout

- `vendor/src/` — vendored EquiCEval engine, RA baseline, LLM layer.
- `expeval/adapters/` — `Pair`/`Decision` interface, EquiCEval and RA adapters.
- `expeval/label_oracle.py` — independent labelling.
- `expeval/run_comparison.py` — FPR / Recall / FAR / Coverage + clustered CI.
- `VENDOR.lock` — vendored file hashes.
