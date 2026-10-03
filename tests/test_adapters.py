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
