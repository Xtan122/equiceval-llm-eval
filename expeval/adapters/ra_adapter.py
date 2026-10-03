"""Adapter for the Refai & Ahmed component-metric baseline (vendored ``src.baseline``).

RA has no native verdict, so its metric vector is reduced to a binary decision by
an explicit rule (see ``ra-eval`` notes). The default is generous to RA and uses
its own most predictive signals:

    faulty  <=>  Cons-R < 1  OR  Cons-RMSE > tau  OR  optimality-gap > tau

``Cons-P`` is excluded from the default because RA's own analysis shows harmless
redundant constraints do not change solver quality; it is available as a
sensitivity via ``use_cons_precision=True``.
"""
from __future__ import annotations

from time import perf_counter

from src.baseline.evaluator import BaselineEvaluator
from src.baseline.ground_truth import PROBLEMS_REGISTRY
from src.benchmark.verified_evaluation import ir_from_dict

from expeval.adapters.common import Decision, Pair


def _constraint_callable(constraint):
    coeffs = dict(constraint.coeffs)
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
        candidate = ir_from_dict(pair.candidate_ir)
        cand_cons = [(c.name, _constraint_callable(c)) for c in candidate.constraints]
        started = perf_counter()
        metrics = BaselineEvaluator(problem).evaluate_candidate(
            candidate_variables=set(candidate.variables),
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
            },
            cost={"elapsed_seconds": elapsed},
        )
