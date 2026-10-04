"""Human patch + Cohen kappa for the pairs the machine oracle could not certify.

Task 1 invariant: ``None`` labels stay out of the FPR/Recall denominators and are
counted separately (``n_unverified``). When the machine oracle (independent of
EquiCEval/RA) cannot prove equivalence either way, two human annotators label the
residual pairs; their agreement is reported as Cohen's kappa and disagreements are
adjudicated. This module never calls EquiCEval or RA.

Usage:
    # 1. Export the residual None pairs for two annotators
    PYTHONPATH=.:vendor python -m expeval.human_patch export \
        --pairs data/llm_outputs_labeled.json \
        --output data/human_patch_sheet.json

    # 2. Annotators fill ``annotator_1`` / ``annotator_2`` (true/false/null)

    # 3. Report kappa and merge adjudicated labels
    PYTHONPATH=.:vendor python -m expeval.human_patch merge \
        --pairs data/llm_outputs_labeled.json \
        --sheet data/human_patch_sheet.json \
        --output data/llm_outputs_labeled_patched.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_TRUE = {"true", "t", "1", "yes", "y", True}
_FALSE = {"false", "f", "0", "no", "n", False}


def _parse_label(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    raise ValueError(f"Unparseable label: {value!r}")


def cohen_kappa(a: List[Optional[bool]], b: List[Optional[bool]]) -> Dict[str, Any]:
    """Cohen's kappa on the pairs both annotators labelled (non-null).

    Only ``True``/``False`` are agreed categories; ``None`` is treated as missing
    and excluded pairwise (it is a "cannot decide", not a class). Returns kappa,
    observed agreement, expected agreement and the confusion counts.
    """
    a = [_parse_label(x) for x in a]
    b = [_parse_label(x) for x in b]
    both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(both)
    if n == 0:
        return {"n": 0, "kappa": None, "observed_agreement": None,
                "expected_agreement": None, "counts": {}}
    cats = (True, False)
    counts = {str(x): {str(y): 0 for y in cats} for x in cats}
    for x, y in both:
        counts[str(x)][str(y)] += 1
    agree = sum(counts[str(c)][str(c)] for c in cats)
    observed = agree / n
    row = {c: counts[str(c)][str(True)] + counts[str(c)][str(False)] for c in cats}
    col = {c: counts[str(True)][str(c)] + counts[str(False)][str(c)] for c in cats}
    expected = sum((row[c] / n) * (col[c] / n) for c in cats)
    kappa = ((observed - expected) / (1 - expected)) if expected < 1 else 1.0
    return {"n": n, "kappa": kappa, "observed_agreement": observed,
            "expected_agreement": expected, "counts": counts}


def export_sheet(pairs: List[dict]) -> Dict[str, Any]:
    """Build the annotation sheet for every pair whose machine label is None."""
    items = []
    for record in pairs:
        if record.get("label") is not None:
            continue
        items.append({
            "pair_id": record["pair_id"],
            "family": record.get("family") or record.get("base_problem_name"),
            "model": record.get("model"),
            "prompt": record.get("prompt"),
            "machine_evidence": record.get("label_evidence"),
            "reference_ir": record.get("reference_ir"),
            "candidate_ir": record.get("candidate_ir"),
            "annotator_1": None,
            "annotator_2": None,
            "adjudicated": None,
            "note": "",
        })
    return {
        "instructions": (
            "Label each (reference, candidate) pair: true = the two formulations are "
            "equivalent under the contract feasible_set_and_objective_affine "
            "(same feasible set and objective equal up to positive scaling); "
            "false = they are not; null = cannot decide. Annotators must work "
            "independently and not consult EquiCEval or RA outputs."
        ),
        "contract": "feasible_set_and_objective_affine",
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="write the residual-None annotation sheet")
    p_export.add_argument("--pairs", type=Path, required=True)
    p_export.add_argument("--output", type=Path, required=True)

    p_merge = sub.add_parser("merge", help="report kappa and merge adjudicated labels")
    p_merge.add_argument("--pairs", type=Path, required=True)
    p_merge.add_argument("--sheet", type=Path, required=True)
    p_merge.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "export":
        raw = json.loads(args.pairs.read_text(encoding="utf-8"))
        records = raw["pairs"] if isinstance(raw, dict) else raw
        sheet = export_sheet(records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(sheet, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Exported {len(sheet['items'])} residual-None pairs -> {args.output}")
        return 0

    raw = json.loads(args.pairs.read_text(encoding="utf-8"))
    records = raw["pairs"] if isinstance(raw, dict) else raw
    sheet = json.loads(args.sheet.read_text(encoding="utf-8"))
    items = {item["pair_id"]: item for item in sheet["items"]}

    kappa = cohen_kappa(
        [item.get("annotator_1") for item in sheet["items"]],
        [item.get("annotator_2") for item in sheet["items"]],
    )
    patched = 0
    for record in records:
        item = items.get(record["pair_id"])
        if item is None:
            continue
        resolved = _parse_label(item.get("adjudicated"))
        if resolved is None:
            resolved = _parse_label(item.get("annotator_1"))
        if resolved is None:
            resolved = _parse_label(item.get("annotator_2"))
        if resolved is not None:
            record["label_machine"] = record.get("label")
            record["label"] = resolved
            record["label_source"] = "human"
            record["annotators"] = {
                "annotator_1": item.get("annotator_1"),
                "annotator_2": item.get("annotator_2"),
                "adjudicated": item.get("adjudicated"),
            }
            patched += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(raw, dict):
        payload = {**raw, "pairs": records, "human_patch": {"kappa": kappa, "n_patched": patched}}
    else:
        payload = records
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Patched {patched} pairs -> {args.output}")
    print(f"Cohen kappa = {kappa['kappa']} (n={kappa['n']}, "
          f"observed={kappa['observed_agreement']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
