from decimal import Decimal as D

import pytest
from quant_os_core.ops import calculators as c


def test_position_size_floors_shares_and_reports_money_at_risk() -> None:
    # budget 10 000 x 1% = 100; risk/share 50 - 47.5 = 2.5; 100 / 2.5 = 40 shares
    out = c.position_size(D("10000"), D("1"), D("50"), D("47.5"))
    assert out["shares"] == D("40")
    assert out["money_at_risk"] == D("100.00")
    assert out["position_value"] == D("2000.00")
    assert out["position_pct_of_equity"] == D("20.0000")
    # 100 / 3 = 33.33 -> 33 whole shares, 99 at risk
    odd = c.position_size(D("10000"), D("1"), D("50"), D("47"))
    assert odd["shares"] == D("33") and odd["money_at_risk"] == D("99.00")
    with pytest.raises(ValueError, match="below the entry"):
        c.position_size(D("10000"), D("1"), D("50"), D("55"))  # a short: blocked
    with pytest.raises(ValueError):
        c.position_size(D("-1"), D("1"), D("50"), D("45"))


def test_reward_risk_and_break_even_win_rate() -> None:
    # reward 7.5, risk 2.5 -> R = 3; break-even = 1 / (1 + 3) = 25%
    out = c.reward_risk(D("50"), D("47.5"), D("57.5"))
    assert out["reward_risk"] == D("3.0000")
    assert out["breakeven_win_rate_pct"] == D("25.0000")
    with pytest.raises(ValueError):
        c.reward_risk(D("50"), D("47.5"), D("49"))


def test_compound_growth_matches_hand_computation() -> None:
    # 12% a year -> 1% a month; 1 000 start, no deposits: 1000 x 1.01^12 = 1126.825...
    out = c.compound_growth(D("1000"), D("0"), D("12"), 1)
    assert out["final_balance"] == D("1126.83")
    # zero return: 100 a month for 2 years = 2 400 deposited, no growth
    flat = c.compound_growth(D("0"), D("100"), D("0"), 2)
    assert flat["final_balance"] == D("2400.00") and flat["total_growth"] == D("0.00")
    assert [row["year"] for row in flat["table"]] == [1, 2]
    assert flat["table"][0]["balance"] == D("1200.00")
    with pytest.raises(ValueError):
        c.compound_growth(D("0"), D("100"), D("8"), 0)


def test_deposit_goal_inverts_compounding() -> None:
    # 0%: 12 000 over 1 year = 1 000 a month
    assert c.deposit_goal(D("12000"), 1, D("0"))["monthly_deposit"] == D("1000.00")
    # 12%: i = 1%, n = 12, g = 1.01^12; pmt = 12682.50 x 0.01 / (g - 1) = 1000.00
    target = c.compound_growth(D("0"), D("1000"), D("12"), 1)["final_balance"]
    assert c.deposit_goal(target, 1, D("12"))["monthly_deposit"] == D("1000.00")
    # start already enough
    assert c.deposit_goal(D("100"), 1, D("5"), start=D("200"))["monthly_deposit"] == D("0.00")


def test_drawdown_recovery() -> None:
    assert c.drawdown_recovery(D("50"))["required_gain_pct"] == D("100.0000")
    assert c.drawdown_recovery(D("20"))["required_gain_pct"] == D("25.0000")
    with pytest.raises(ValueError):
        c.drawdown_recovery(D("100"))


def test_dca_average_cost_is_amount_weighted_harmonic_mean() -> None:
    # 100 @ 10 = 10 units, 100 @ 20 = 5 units -> 200 / 15 = 13.33
    out = c.dca_average_cost([(D("100"), D("10")), (D("100"), D("20"))])
    assert out["average_cost"] == D("13.33")
    assert out["simple_average_price"] == D("15.00")
    assert out["units"] == D("15.00000000")
    with pytest.raises(ValueError):
        c.dca_average_cost([])


def test_cost_impact() -> None:
    # 200 000 traded at 20 bps = 400; on 100 000 equity that is 0.4%
    out = c.cost_impact(D("200000"), D("20"), D("100000"))
    assert out["annual_cost"] == D("400.00") and out["drag_pct"] == D("0.4000")
    assert c.cost_impact(D("1000"), D("10"))["drag_pct"] is None


def test_zakat_is_on_whole_wealth_once_nisab_is_reached() -> None:
    out = c.zakat(D("1000000"), D("600000"))
    assert out["due"] == D("25000.00") and out["above_nisab"] is True
    below = c.zakat(D("500000"), D("600000"))
    assert below["due"] == D("0.00") and below["above_nisab"] is False
    # nisab from gold: 87.48 g x 10 000 per gram = 874 800
    assert c.nisab_from_metal(D("10000")) == D("874800.00")
    with pytest.raises(ValueError):
        c.zakat(D("100"), D("0"))


def test_purification() -> None:
    out = c.purification(D("12000"), D("3"))
    assert out["purify"] == D("360.00") and out["keep"] == D("11640.00")
    with pytest.raises(ValueError):
        c.purification(D("100"), D("101"))


def test_run_parses_json_and_rejects_nonsense() -> None:
    out = c.run("position_size", {"equity": "10,000", "risk_pct": 1, "entry": "50", "stop": 47.5})
    assert out["shares"] == "40" and out["money_at_risk"] == "100.00"
    dca = c.run("dca", {"buys": "100@10, 100@20"})
    assert dca["average_cost"] == "13.33"
    zakat = c.run("zakat", {"zakatable_wealth": "1000000", "gold_price_per_gram": "10000"})
    assert zakat["nisab"] == "874800.00" and zakat["due"] == "25000.00"
    compound = c.run("compound", {"annual_return_pct": "0", "years": "1", "monthly_deposit": "1"})
    assert compound["table"][0]["balance"] == "12.00"
    with pytest.raises(KeyError):
        c.run("nope", {})
    for bad in (
        {"equity": "abc", "risk_pct": 1, "entry": 50, "stop": 45},
        {"equity": "NaN", "risk_pct": 1, "entry": 50, "stop": 45},
        {"equity": True, "risk_pct": 1, "entry": 50, "stop": 45},
        {"risk_pct": 1, "entry": 50, "stop": 45},
    ):
        with pytest.raises(ValueError):
            c.run("position_size", bad)
    with pytest.raises(ValueError):
        c.run("compound", {"annual_return_pct": "5", "years": "2.5"})
    with pytest.raises(ValueError):
        c.run("drawdown", ["not", "an", "object"])
