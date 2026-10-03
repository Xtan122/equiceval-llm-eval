import copy

from expeval.adapters import EquiCEvalAdapter, RAAdapter
from expeval.adapters.common import Pair, load_pairs

CONTRACT = "feasible_set_and_objective_affine"


def _pairs():
    return load_pairs("data/sample_pairs.json", CONTRACT)


def test_equiceval_certifies_labelled_equivalent():
    pair = next(p for p in _pairs() if p.label is True)
    decision = EquiCEvalAdapter(seconds=1.0).score(pair)
    assert decision.outcome in {"equivalent", "tolerance"}


def test_ra_accepts_labelled_equivalent():
    pair = next(p for p in _pairs() if p.label is True)
    decision = RAAdapter().score(pair)
    assert decision.outcome == "equivalent"


def test_ra_flags_missing_constraint():
    pair = next(p for p in _pairs() if p.label is True)
    candidate = copy.deepcopy(pair.candidate_ir)
    candidate["constraints"] = []
    mutated = Pair(pair.pair_id, pair.family, pair.reference_ir, candidate, False, pair.contract)
    assert RAAdapter().score(mutated).outcome == "faulty"


def test_ra_normalizes_compact_variable_names():
    """``x1`` (reference_ir) must be scored against ``x_1`` (ground-truth domain)."""
    from src.benchmark.reference_ir import get_reference_ir
    from src.benchmark.verified_evaluation import ir_to_dict

    reference = get_reference_ir("Knapsack")  # compact names x1, x2, x3
    pair = Pair("p", "Knapsack", ir_to_dict(reference), ir_to_dict(reference), True, CONTRACT)
    decision = RAAdapter().score(pair)
    assert decision.diagnostics["renamed_variables"] == {"x1": "x_1", "x2": "x_2", "x3": "x_3"}
    assert decision.outcome == "equivalent"
