from quant_os_core.alerts import AlertRule, Comparison


def test_threshold_and_crossing_rules() -> None:
    assert AlertRule("a", "price", Comparison.ABOVE, 100).triggered(101)
    assert not AlertRule("b", "price", Comparison.BELOW, 100).triggered(101)
    cross = AlertRule("c", "rsi", Comparison.CROSSES_ABOVE, 70)
    assert cross.triggered(71, 69)
    assert not cross.triggered(71, None)
    cross_down = AlertRule("d", "rsi", Comparison.CROSSES_BELOW, 30)
    assert cross_down.triggered(29, 31)


def test_alert_rule_rejects_blank_identity() -> None:
    import pytest

    with pytest.raises(ValueError):
        AlertRule("", "price", Comparison.ABOVE, 1)
