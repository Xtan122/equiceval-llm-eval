"""
Implementation of the exact evaluation metrics proposed by Refai & Ahmed (arXiv:2510.16943):
- Constraint Precision & Recall (Eq. 1 & 2)
- Decision Variable Precision & Recall (Eq. 3 & 4)
- Optimality Gap (Eq. 5)
- Objective RMSE (Obj-RMSE) (Eq. 6)
- Constraint RMSE (Cons-RMSE) (Eq. 7)
- Efficiency Metrics (Latency, Input Tokens, Output Tokens)
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Callable
import numpy as np


@dataclass
class BaselineMetricResults:
    cons_precision: float
    cons_recall: float
    dv_precision: float
    dv_recall: float
    optimality_gap: Optional[float]
    obj_rmse: float
    cons_rmse: float
    latency_seconds: float
    input_tokens: int
    output_tokens: int


def compute_precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    """
    Computes precision and recall:
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall


def compute_optimality_gap(true_optimal: float, llm_optimal: Optional[float], eps: float = 1e-8) -> Optional[float]:
    """
    Computes optimality gap (Eq. 5 in Refai & Ahmed):
    Optimality gap = |Optimal value - LLM objective value| / |Optimal value|

    Edge cases:
    - If llm_optimal is None (infeasible/solver failed), returns None.
    - If true_optimal == 0 (e.g. Aircraft Landing), returns absolute gap |0 - llm_optimal|.
    """
    if llm_optimal is None or np.isnan(llm_optimal):
        return None

    abs_diff = abs(true_optimal - llm_optimal)
    if abs(true_optimal) < eps:
        # Edge case: Aircraft landing has z* = 0
        return abs_diff
    return abs_diff / abs(true_optimal)


def compute_obj_rmse(true_values: List[float], llm_values: List[float]) -> float:
    """
    Computes Objective Function RMSE across instances (Eq. 6 in Refai & Ahmed):
    Obj-RMSE = sqrt( (1/n) * sum_{i=1}^n (True_obj_i - LLM_obj_i)^2 )
    """
    if not true_values or len(true_values) != len(llm_values):
        return 0.0
    arr_true = np.array(true_values, dtype=np.float64)
    arr_llm = np.array(llm_values, dtype=np.float64)
    return float(np.sqrt(np.mean((arr_true - arr_llm) ** 2)))


def generate_evaluation_samples(
    sample_domain: Dict[str, Tuple[float, float]],
    num_samples: int = 100,
    random_seed: int = 42
) -> List[Dict[str, float]]:
    """
    Generates n = 100 random decision variable samples x^{(i)} uniformly
    drawn from the specified valid input domain using a fixed random seed.
    """
    rng = np.random.RandomState(random_seed)
    samples = []
    var_names = list(sample_domain.keys())

    for _ in range(num_samples):
        sample = {}
        for var_name in var_names:
            low, high = sample_domain[var_name]
            if high == float('inf'):
                high = low + 10.0
            sample[var_name] = float(rng.uniform(low, high))
        samples.append(sample)

    return samples


def compute_cons_rmse(
    matched_constraints: List[Tuple[Callable[[Dict[str, float]], float], Callable[[Dict[str, float]], float]]],
    samples: List[Dict[str, float]]
) -> float:
    """
    Computes Constraint Behavior RMSE over matched constraints (Eq. 7 in Refai & Ahmed):
    Cons-RMSE = sqrt( (1 / (|C| * n)) * sum_{k in C} sum_{i=1}^n (f_k^GT(x^{(i)}) - f_k^LLM(x^{(i)}))^2 )

    Note: Computed ONLY over matched constraints |C|. Omitted/hallucinated constraints are excluded.
    """
    if not matched_constraints or not samples:
        return 0.0

    num_matched = len(matched_constraints)
    num_samples = len(samples)
    total_sq_diff = 0.0

    for gt_func, llm_func in matched_constraints:
        for x_sample in samples:
            try:
                gt_val = float(gt_func(x_sample))
                llm_val = float(llm_func(x_sample))
                diff = gt_val - llm_val
                total_sq_diff += diff * diff
            except Exception:
                # If evaluation fails, assign penalty difference
                total_sq_diff += 1e4

    mean_sq_diff = total_sq_diff / (num_matched * num_samples)
    return float(np.sqrt(mean_sq_diff))
