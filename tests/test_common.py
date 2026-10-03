from expeval.adapters.common import rates


def test_rates_class_conditional():
    labels = [True, True, False, False]
    outcomes = ["faulty", "equivalent", "faulty", "equivalent"]
    metrics = rates(labels, outcomes)
    assert metrics["false_alarm_rate"]["value"] == 0.5
    assert metrics["error_recall"]["value"] == 0.5
    assert metrics["false_acceptance_rate"]["value"] == 0.5
    assert metrics["equivalent_confirmation_rate"]["value"] == 0.5
    assert metrics["coverage"]["value"] == 1.0


def test_rates_unverified_excluded_from_denominators():
    labels = [True, None, False]
    outcomes = ["faulty", "unresolved", "equivalent"]
    metrics = rates(labels, outcomes)
    assert metrics["n_unverified"] == 1
    assert metrics["false_alarm_rate"]["denominator"] == 1
    assert metrics["error_recall"]["denominator"] == 1
    assert metrics["error_recall"]["value"] == 0.0
