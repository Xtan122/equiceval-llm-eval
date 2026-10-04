"""Adapter for the Refai & Ahmed component-metric baseline (vendored ``src.baseline``).

RA has no native verdict, so its metric vector is reduced to a binary decision by
an explicit rule (see ``ra-eval`` notes). The default is generous to RA and uses
its own most predictive signals:

    faulty  <=>  Cons-R < 1  OR  Cons-RMSE > tau  OR  optimality-gap > tau

``Cons-P`` is excluded from the default because RA's own analysis shows harmless
redundant constraints do not change solver quality; it is available as a
sensitivity via ``use_cons_precision=True``.

Candidate variables are renamed to the ground-truth variable names before
scoring, so Cons-RMSE samples the GT domain with matching keys. Without this,
formulations that name ``x1`` (compact) instead of ``x_1`` (ground truth) would
be scored against empty samples and flagged as faulty.
"""
from __future__ import annotations

import re
from time import perf_counter

from src.baseline.evaluator import BaselineEvaluator
from src.baseline.ground_truth import PROBLEMS_REGISTRY
from src.benchmark.verified_evaluation import ir_from_dict

from expeval.adapters.common import Decision, Pair


def _normalize(name: str) -> str:
    """Case- and separator-insensitive key that also collapses index markers.

    ``x_1`` / ``x1`` → ``x1`` and ``x_A1_R1`` / ``x_1_1`` / ``x11`` → ``x11``:
    the leading alpha prefix plus the trailing digit sequence identifies the
    variable regardless of whether the LLM wrote ``A1``/``R1`` labels or plain
    indices. Names without any digit (e.g. ``x_apple``) keep the full stripped
    form so distinct word-based names do not collide.
    """
    lowered = name.lower()
    digits = re.findall(r"\d+", lowered)
    if not digits:
        return re.sub(r"[^a-z0-9]", "", lowered)
    prefix = re.match(r"[a-z]+", lowered)
    head = prefix.group(0) if prefix else ""
    return head + "".join(digits)


def _name_map(candidate_variables, gt_variables):
    """Map candidate variable names onto ground-truth names by normalized form."""
    gt_by_norm = {_normalize(name): name for name in gt_variables}
    mapping = {}
    for name in candidate_variables:
        target = gt_by_norm.get(_normalize(name))
        if target is not None and target != name:
            mapping[name] = target
    return mapping


def _constraint_callable(constraint, name_map):
    coeffs = {}
    for var, coef in constraint.coeffs.items():
        target = name_map.get(var, var)
        coeffs[target] = coeffs.get(target, 0.0) + coef
    constant = constraint.constant

    def func(x, coeffs=coeffs, constant=constant):
        return sum(coef * x.get(var, 0.0) for var, coef in coeffs.items()) + constant

    return func


class RAAdapter:
    name = "ra"

    def __init__(self, tolerance: float = 1e-6, use_cons_precision: bool = False,
                 solver_only: bool = False):
        self.tolerance = tolerance
        self.use_cons_precision = use_cons_precision
        self.solver_only = solver_only

    def score(self, pair: Pair) -> Decision:
        problem = pair.family
        if problem not in PROBLEMS_REGISTRY:
            return Decision(pair.pair_id, self.name, "unsupported",
                            {"reason": f"no GroundTruthProblem for {problem!r}"})

        gt_problem = PROBLEMS_REGISTRY[problem]
        candidate = ir_from_dict(pair.candidate_ir)
        name_map = _name_map(candidate.variables, gt_problem.variables)
        cand_vars = {name_map.get(var, var) for var in candidate.variables}
        cand_cons = [(c.name, _constraint_callable(c, name_map))
                     for c in candidate.constraints]

        started = perf_counter()
        metrics = BaselineEvaluator(problem).evaluate_candidate(
            candidate_variables=cand_vars,
            candidate_constraints=cand_cons,
            candidate_optimal_val=None,
        )
        elapsed = perf_counter() - started

        gap = metrics.optimality_gap
        if self.solver_only:
            faulty = gap is not None and abs(gap) > self.tolerance
        else:
            faulty = (
                metrics.cons_recall < 1.0
                or metrics.cons_rmse > self.tolerance
                or (gap is not None and abs(gap) > self.tolerance)
            )
            if self.use_cons_precision and metrics.cons_precision < 1.0:
                faulty = True

        return Decision(
            pair_id=pair.pair_id,
            method=self.name,
            outcome="faulty" if faulty else "equivalent",
            diagnostics={
                "cons_precision": metrics.cons_precision,
                "cons_recall": metrics.cons_recall,
                "cons_rmse": metrics.cons_rmse,
                "optimality_gap": gap,
                "obj_rmse": metrics.obj_rmse,
                "renamed_variables": name_map,
            },
            cost={"elapsed_seconds": elapsed},
        )
