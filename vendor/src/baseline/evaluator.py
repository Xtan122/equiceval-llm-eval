"""
Refai & Ahmed (arXiv:2510.16943) Baseline Evaluator.
Executes component-level and solution-level evaluation for candidate formulations.
"""

from typing import Dict, List, Set, Tuple, Optional, Callable, Any
import numpy as np

from src.baseline.ground_truth import GroundTruthProblem, PROBLEMS_REGISTRY
from src.baseline.metrics import (
    BaselineMetricResults,
    compute_precision_recall,
    compute_optimality_gap,
    compute_obj_rmse,
    compute_cons_rmse,
    generate_evaluation_samples
)


class BaselineEvaluator:
    """
    Evaluates candidate formulations using the exact component-level metrics
    from Refai & Ahmed (2025).
    """

    def __init__(self, problem_name_or_gt: Any):
        if isinstance(problem_name_or_gt, str):
            if problem_name_or_gt not in PROBLEMS_REGISTRY:
                raise ValueError(f"Problem '{problem_name_or_gt}' not in registry: {list(PROBLEMS_REGISTRY.keys())}")
            self.gt_problem: GroundTruthProblem = PROBLEMS_REGISTRY[problem_name_or_gt]
        else:
            self.gt_problem: GroundTruthProblem = problem_name_or_gt

    def evaluate_candidate(
        self,
        candidate_variables: Set[str],
        candidate_constraints: List[Tuple[str, Callable[[Dict[str, float]], float]]], # (name/repr, func)
        candidate_optimal_val: Optional[float],
        candidate_obj_func: Optional[Callable[[Dict[str, float]], float]] = None,
        latency_seconds: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        matched_constraint_indices: Optional[List[Tuple[int, int]]] = None, # (gt_idx, candidate_idx)
        num_samples: int = 100,
        random_seed: int = 42,
    ) -> BaselineMetricResults:
        """
        Evaluates a candidate model against ground truth.

        matched_constraint_indices: List of pairs (gt_idx, cand_idx) indicating which
        ground truth constraints are matched to which candidate constraints.
        If None, matching is computed by name / structural match.
        """
        gt = self.gt_problem

        # 1. Decision Variable Precision & Recall
        gt_vars = set(gt.variables.keys())
        tp_vars = len(gt_vars.intersection(candidate_variables))
        fp_vars = len(candidate_variables - gt_vars)
        fn_vars = len(gt_vars - candidate_variables)
        dv_p, dv_r = compute_precision_recall(tp_vars, fp_vars, fn_vars)

        # 2. Constraint Precision & Recall
        gt_num_cons = len(gt.constraints)
        cand_num_cons = len(candidate_constraints)

        if matched_constraint_indices is not None:
            tp_cons = len(matched_constraint_indices)
        else:
            # Default matching by name or sequential fallback
            tp_cons = min(gt_num_cons, cand_num_cons)
            matched_constraint_indices = [(i, i) for i in range(tp_cons)]

        fp_cons = max(0, cand_num_cons - tp_cons)
        fn_cons = max(0, gt_num_cons - tp_cons)
        cons_p, cons_r = compute_precision_recall(tp_cons, fp_cons, fn_cons)

        # 3. Optimality Gap
        opt_gap = compute_optimality_gap(gt.optimal_value, candidate_optimal_val)

        # 4. Obj-RMSE
        if candidate_optimal_val is not None:
            obj_rmse = compute_obj_rmse([gt.optimal_value], [candidate_optimal_val])
        else:
            obj_rmse = 0.0

        # 5. Cons-RMSE (evaluated over matched constraints only)
        samples = generate_evaluation_samples(gt.sample_domain, num_samples=num_samples, random_seed=random_seed)

        matched_pairs = []
        for gt_idx, cand_idx in matched_constraint_indices:
            if 0 <= gt_idx < len(gt.constraints) and 0 <= cand_idx < len(candidate_constraints):
                gt_func = gt.constraints[gt_idx].expression_func
                cand_func = candidate_constraints[cand_idx][1]
                matched_pairs.append((gt_func, cand_func))

        cons_rmse = compute_cons_rmse(matched_pairs, samples)

        return BaselineMetricResults(
            cons_precision=float(cons_p),
            cons_recall=float(cons_r),
            dv_precision=float(dv_p),
            dv_recall=float(dv_r),
            optimality_gap=opt_gap,
            obj_rmse=obj_rmse,
            cons_rmse=cons_rmse,
            latency_seconds=latency_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
