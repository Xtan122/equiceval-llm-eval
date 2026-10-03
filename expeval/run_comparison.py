"""Compare RA and EquiCEval on a frozen pair set and report FPR / Recall.

Usage:
    PYTHONPATH=. python -m expeval.run_comparison \
        --pairs data/sample_pairs.json \
        --contract feasible_set_and_objective_affine \
        --output output/llm_eval_report.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.equiceval.contracts import PRIMARY_EQUIVAFORMULATION_CONTRACT

from expeval.adapters import EquiCEvalAdapter, RAAdapter
from expeval.adapters.common import load_pairs, rate_ci, rates


def _fmt(metric):
    value = metric.get("value") if isinstance(metric, dict) else metric
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--contract", default=PRIMARY_EQUIVAFORMULATION_CONTRACT)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--mode", default="certificates",
                        choices=["direct_only", "certificates", "early_stop"])
    parser.add_argument("--ra-rule", default="primary",
                        choices=["primary", "cons-precision", "solver-only"],
                        help="RA binary decision rule (sensitivity analysis)")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON file; existing files are never overwritten")
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    if not args.pairs.exists():
        parser.error(f"Pairs file not found: {args.pairs}")

    pairs = load_pairs(args.pairs, args.contract)
    adapters = {
        "equiceval": EquiCEvalAdapter(seconds=args.seconds, mode=args.mode),
        "ra": RAAdapter(use_cons_precision=args.ra_rule == "cons-precision",
                        solver_only=args.ra_rule == "solver-only"),
    }

    labels = [p.label for p in pairs]
    families = [p.family or "unknown" for p in pairs]
    report = {
        "config": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pairs": str(args.pairs),
            "n_pairs": len(pairs),
            "contract": args.contract,
            "equiceval_mode": args.mode,
            "equiceval_seconds": args.seconds,
            "ra_rule": args.ra_rule,
        },
        "methods": {},
    }

    for name, adapter in adapters.items():
        decisions = [adapter.score(pair) for pair in pairs]
        outcomes = [d.outcome for d in decisions]
        method_metrics = rates(labels, outcomes)
        method_metrics["false_alarm_ci"] = rate_ci(labels, outcomes, families, "faulty", True)
        method_metrics["error_recall_ci"] = rate_ci(labels, outcomes, families, "faulty", False)
        report["methods"][name] = {
            "metrics": method_metrics,
            "decisions": [d.__dict__ for d in decisions],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved -> {args.output}  (n={len(pairs)})")
    for name, entry in report["methods"].items():
        m = entry["metrics"]
        print(f"  {name:10s} FPR={_fmt(m['false_alarm_rate'])} "
              f"recall={_fmt(m['error_recall'])} "
              f"unresolved={_fmt(m['unresolved_rate'])} "
              f"unsupported={_fmt(m['unsupported_rate'])} "
              f"unverified={m['n_unverified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
