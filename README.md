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

`PYTHONPATH` must include both the experiment repo (`.`) and the vendored method
tree (`vendor`).

```bash
# 1. Generate candidates with a real LLM (Bedrock / Ollama / OpenAI / vLLM).
#    gpt-oss-120B lives in us-east-1 (not the default ap-southeast-1).
PYTHONPATH=.:vendor AWS_DEFAULT_REGION=us-east-1 .venv/bin/python -m expeval.generate_llm_outputs \
  --model bedrock-gpt --problem Knapsack AircraftAssignment Diet AircraftLanding \
  --prompt P1 P2 P3 P4 P5 P6 \
  --output data/llm_outputs_raw.json

# 2. Label them independently (True / False / null); prints per-family coverage
PYTHONPATH=.:vendor .venv/bin/python -m expeval.label_oracle \
  --pairs data/llm_outputs_raw.json \
  --contract feasible_set_and_objective_affine \
  --output data/llm_outputs_labeled.json

# 3. Compare RA vs EquiCEval
PYTHONPATH=.:vendor .venv/bin/python -m expeval.run_comparison \
  --pairs data/llm_outputs_labeled.json \
  --contract feasible_set_and_objective_affine \
  --ra-rule primary \
  --output output/llm_eval_report.json

# 4. (optional) human patch + Cohen kappa for residual None labels
PYTHONPATH=.:vendor .venv/bin/python -m expeval.human_patch export \
  --pairs data/llm_outputs_labeled.json --output data/human_patch_sheet.json
PYTHONPATH=.:vendor .venv/bin/python -m expeval.human_patch merge \
  --pairs data/llm_outputs_labeled.json --sheet data/human_patch_sheet.json \
  --output data/llm_outputs_labeled_patched.json
```

The machine oracle has four independent tiers: exact integer enumeration, exact
variable-renaming bijection, LP containment, and (new) mixed-integer containment
that enumerates discrete variables and solves the continuous part as an LP.
`None` means "not provable", never "not equivalent".

On the EquiCEval side, M1 now supports **existential projection** of auxiliary
variables (a candidate that encodes the same problem with fewer/more auxiliaries
than the reference). This lifts single-ordering-binary candidates into the
reference space when `order_link` rows become tautologies, so M4/M5 can run.
Proposal and results: `../EquiCEval/docs/PROJECTION_EXISTENTIAL_PROPOSAL.md`.

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
- `expeval/label_oracle.py` — independent labelling + per-family coverage summary.
- `expeval/run_comparison.py` — FPR / Recall (+ unresolved/unsupported rates).
- `expeval/human_patch.py` — residual-`None` annotation sheet + Cohen kappa merge.
- `VENDOR.lock` — vendored file hashes.
