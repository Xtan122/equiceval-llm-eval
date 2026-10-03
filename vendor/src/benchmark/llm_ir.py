"""Convert a parsed LLM formulation into the canonical IR used by the verified
evaluator (Module 0-5). Kept free of evaluator imports so it can be reused by
generation/benchmark tooling.
"""
from __future__ import annotations

from typing import Optional

from src.equiceval.canonical_ir import (
    CanonicalIR,
    CanonicalConstraint,
    CanonicalVariable,
)
from src.benchmark.output_parser import ParsedFormulation


def parsed_to_canonical_ir(
    pf: Optional[ParsedFormulation],
    problem_name: str,
) -> Optional[CanonicalIR]:
    """Map ``ParsedFormulation`` -> ``CanonicalIR`` in canonical ``<= 0`` form.

    Convention used by the rest of the codebase: a constraint is stored as
    ``sum(a_i * x_i) + constant <= 0`` (or ``== 0``), so the solver right-hand
    side is ``-constant``. ``>=`` rows are flipped to ``<=``.
    """
    if pf is None:
        return None
    try:
        variables = {
            name: CanonicalVariable(
                name=name,
                var_type=vtype,
                lower_bound=lb,
                upper_bound=ub,
            )
            for name, (vtype, lb, ub) in pf.variable_specs.items()
        }
        constraints = []
        for i, (coeffs, sense, rhs) in enumerate(pf.constraints_for_solver()):
            coeffs = {k: float(v) for k, v in coeffs.items()}
            if sense == ">=":
                # a >= b  <=>  -a + b <= 0
                constraints.append(CanonicalConstraint(
                    f"c{i}", {k: -v for k, v in coeffs.items()}, float(rhs), "<="))
            elif sense == "==":
                constraints.append(CanonicalConstraint(
                    f"c{i}", coeffs, -float(rhs), "=="))
            else:
                constraints.append(CanonicalConstraint(
                    f"c{i}", coeffs, -float(rhs), "<="))
        return CanonicalIR(
            problem_name=f"{problem_name}_LLM",
            variables=variables,
            constraints=constraints,
            objective_sense=pf.objective_sense,
            objective_coeffs={k: float(v) for k, v in pf.objective_coeffs.items()},
            objective_constant=0.0,
            metadata={"source": "llm_output"},
        )
    except Exception:
        return None
