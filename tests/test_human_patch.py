"""Human-patch export and Cohen's kappa for residual None labels."""
from expeval.human_patch import cohen_kappa, export_sheet


def test_kappa_perfect_agreement():
    result = cohen_kappa([True, False, True, False], [True, False, True, False])
    assert result["n"] == 4
    assert result["kappa"] == 1.0


def test_kappa_matches_standard_example():
    a = [True, True, True, True, False, False, False, False, False, False]
    b = [True, True, True, False, False, False, False, True, True, False]
    result = cohen_kappa(a, b)
    assert abs(result["kappa"] - 0.4) < 1e-9
    assert abs(result["observed_agreement"] - 0.7) < 1e-9


def test_kappa_ignores_none_pairwise():
    result = cohen_kappa([True, None, False], ["true", None, "false"])
    assert result["n"] == 2
    assert result["kappa"] == 1.0


def test_export_sheet_only_includes_none_labels():
    pairs = [
        {"pair_id": "a", "family": "Knapsack", "label": True,
         "reference_ir": {}, "candidate_ir": {}},
        {"pair_id": "b", "family": "AircraftLanding", "label": None,
         "reference_ir": {}, "candidate_ir": {}, "label_evidence": {"method": "unavailable"}},
    ]
    sheet = export_sheet(pairs)
    assert [item["pair_id"] for item in sheet["items"]] == ["b"]
    assert sheet["items"][0]["annotator_1"] is None
    assert sheet["contract"] == "feasible_set_and_objective_affine"
