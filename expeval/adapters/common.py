"""Common comparison interface shared by both experiment repos.

``Pair`` is one (reference, candidate, verified label) unit. Each method adapter
reduces a method's native output to a ``Decision`` with a 5-value outcome:
``equivalent`` / ``faulty`` / ``tolerance`` / ``unresolved`` / ``unsupported``.
The rate helpers then compute class-conditional FPR / Recall and the
supplementary rates the target document asks for (unresolved / unsupported).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Pair:
    pair_id: str
    family: Optional[str]
    reference_ir: Dict[str, Any]
    candidate_ir: Dict[str, Any]
    label: Optional[bool]
    contract: str


@dataclass
class Decision:
    pair_id: str
    method: str
    outcome: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)


def load_pairs(path, default_contract: str) -> List[Pair]:
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = raw["pairs"] if isinstance(raw, dict) else raw
    pairs = []
    for record in records:
        pairs.append(Pair(
            pair_id=record["pair_id"],
            family=record.get("family") or record.get("base_problem_name"),
            reference_ir=record["reference_ir"],
            candidate_ir=record["candidate_ir"],
            label=record.get("label", record.get("is_semantically_equivalent")),
            contract=record.get("contract", default_contract),
        ))
    return pairs


def _rate(numerator: int, denominator: int) -> Dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator,
            "value": (numerator / denominator) if denominator else None}


def rates(labels: Sequence[Optional[bool]], outcomes: Sequence[str],
          accepted=("equivalent",)) -> Dict[str, Any]:
    if len(labels) != len(outcomes):
        raise ValueError("labels and outcomes must have equal length")
    eq = [i for i, label in enumerate(labels) if label is True]
    bad = [i for i, label in enumerate(labels) if label is False]
    accepted = set(accepted)
    return {
        "n_total": len(labels),
        "n_equivalent": len(eq),
        "n_faulty": len(bad),
        "n_unverified": sum(label is None for label in labels),
        "false_alarm_rate": _rate(sum(outcomes[i] == "faulty" for i in eq), len(eq)),
        "error_recall": _rate(sum(outcomes[i] == "faulty" for i in bad), len(bad)),
        "equivalent_confirmation_rate": _rate(sum(outcomes[i] in accepted for i in eq), len(eq)),
        "unresolved_rate": _rate(sum(o == "unresolved" for o in outcomes), len(outcomes)),
        "unsupported_rate": _rate(sum(o == "unsupported" for o in outcomes), len(outcomes)),
        "outcome_counts": dict(Counter(outcomes)),
    }
