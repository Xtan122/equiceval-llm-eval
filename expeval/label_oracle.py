"""Independently label a pair set with the vendored oracle.

The label is produced *without* the EquiCEval pipeline and is what FPR / Recall
are computed against. Unverifiable pairs get ``label = null`` and stay out of the
rate denominators.

Usage:
    PYTHONPATH=. python -m expeval.label_oracle \
        --pairs data/llm_outputs_raw.json --contract \
        feasible_set_and_objective_affine --output data/llm_outputs_labeled.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark.independent_oracle import independent_label
from src.benchmark.verified_evaluation import ir_from_dict
from src.equiceval.contracts import PRIMARY_EQUIVAFORMULATION_CONTRACT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--contract", default=PRIMARY_EQUIVAFORMULATION_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    raw = json.loads(args.pairs.read_text(encoding="utf-8"))
    records = raw["pairs"] if isinstance(raw, dict) else raw

    labeled = 0
    for record in records:
        reference = ir_from_dict(record["reference_ir"])
        candidate = ir_from_dict(record["candidate_ir"])
        label, evidence = independent_label(reference, candidate, args.contract)
        record["label"] = label
        record["label_evidence"] = evidence
        labeled += 1 if label is not None else 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if not isinstance(raw, dict) else {**raw, "pairs": records}
    if isinstance(raw, list):
        payload = records
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Labeled {labeled}/{len(records)} pairs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
