from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_os_core.agents.marketplace import (
    Marketplace,
    ProviderKind,
    ProviderStatus,
    SignalOutcome,
    VettingPolicy,
)

START = datetime(2026, 6, 1, tzinfo=UTC)
NOW = START + timedelta(days=120)

WIN = Decimal("0.03")
LOSS = Decimal("-0.01")


def _feed(
    market: Marketplace,
    provider_id: str,
    returns: list[Decimal],
    *,
    start: datetime = START,
    step: timedelta = timedelta(days=3),
    shariah: bool | None = True,
) -> None:
    for index, value in enumerate(returns):
        signal_id = f"{provider_id}-{index}"
        market.record(SignalOutcome(provider_id, signal_id, start + step * index, shariah, True))
        market.grade(provider_id, signal_id, value)


def _good_returns(count: int = 36) -> list[Decimal]:
    pattern = [WIN, WIN, LOSS, WIN, Decimal("0.02"), LOSS]
    return [pattern[i % len(pattern)] for i in range(count)]


def test_provider_listing_and_lookup() -> None:
    market = Marketplace()
    record = market.add_provider("strat.trend", ProviderKind.INTERNAL_STRATEGY, "Trend")
    assert record.status is ProviderStatus.CANDIDATE
    assert market.ensure_provider("strat.trend", ProviderKind.INTERNAL_STRATEGY) is record
    market.ensure_provider("board.claude", ProviderKind.ANALYST_BOARD)
    assert [r.provider_id for r in market.providers()] == ["board.claude", "strat.trend"]
    assert market.provider("board.claude").display_name == "board.claude"
    with pytest.raises(ValueError, match="already listed"):
        market.add_provider("strat.trend", ProviderKind.INTERNAL_STRATEGY)
    with pytest.raises(ValueError, match="blank"):
        market.add_provider(" ", ProviderKind.INTERNAL_STRATEGY)
    with pytest.raises(KeyError, match="unknown provider"):
        market.provider("ghost")
    with pytest.raises(KeyError, match="unknown provider"):
        market.outcomes("ghost")


def test_record_dedupes_and_grade_rules() -> None:
    market = Marketplace()
    market.add_provider("d", ProviderKind.DISCORD_CHANNEL)
    outcome = SignalOutcome("d", "m1", START, True, True)
    assert market.record(outcome)
    assert not market.record(outcome)
    market.record(SignalOutcome("d", "m2", START, False, False))
    graded = market.grade("d", "m1", Decimal("0.05"))
    assert graded.return_pct == Decimal("0.05")
    with pytest.raises(ValueError, match="only approved"):
        market.grade("d", "m2", Decimal("0.01"))
    with pytest.raises(ValueError, match="Decimal"):
        market.grade("d", "m1", 0.01)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="100%"):
        market.grade("d", "m1", Decimal("-1"))
    with pytest.raises(KeyError, match="unknown signal"):
        market.grade("d", "zz", Decimal("0.01"))
    with pytest.raises(KeyError, match="unknown provider"):
        market.record(SignalOutcome("ghost", "x", START, True, True))


def test_outcome_and_policy_validation() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        SignalOutcome("d", "m", datetime(2026, 1, 1), True, True)
    with pytest.raises(ValueError, match="Decimal"):
        SignalOutcome("d", "m", START, True, True, 0.1)  # type: ignore[arg-type]
    for bad in (
        {"min_signals": 2},
        {"min_track_days": -1},
        {"min_deflated_sharpe": 1.5},
        {"min_hit_rate": Decimal("2")},
        {"revet_interval": timedelta(0)},
    ):
        with pytest.raises(ValueError):
            VettingPolicy(**bad)  # type: ignore[arg-type]


def test_good_track_record_gets_vetted_single_provider() -> None:
    market = Marketplace()
    market.add_provider("strat.trend", ProviderKind.INTERNAL_STRATEGY)
    _feed(market, "strat.trend", _good_returns())
    result = market.vet("strat.trend", NOW)
    assert result.passed, result.reasons
    assert result.previous is ProviderStatus.CANDIDATE
    assert result.reasons == ("meets every vetting rule",)
    metrics = result.metrics
    assert metrics.graded == 36 and metrics.track_days == 120
    assert metrics.hit_rate is not None and metrics.hit_rate > Decimal("0.6")
    assert metrics.halal_compliance == Decimal("1")
    assert market.is_vetted("strat.trend")
    assert market.vetted(ProviderKind.INTERNAL_STRATEGY)[0].provider_id == "strat.trend"
    assert market.vetted(ProviderKind.DISCORD_CHANNEL) == ()


def test_deflation_uses_competing_providers() -> None:
    market = Marketplace()
    for name in ("a", "b"):
        market.add_provider(name, ProviderKind.INTERNAL_STRATEGY)
        _feed(market, name, _good_returns())
    metrics = market.metrics("a", NOW)
    assert metrics.deflated_sharpe is not None
    assert metrics.observed_sharpe is not None


def test_short_young_or_haram_records_stay_candidate() -> None:
    market = Marketplace()
    market.add_provider("new", ProviderKind.DISCORD_CHANNEL)
    _feed(market, "new", _good_returns(5), start=NOW - timedelta(days=10), step=timedelta(days=1))
    result = market.vet("new", NOW)
    assert result.status is ProviderStatus.CANDIDATE
    text = " ".join(result.reasons)
    assert "graded signals" in text and "track record" in text

    market.add_provider("haram", ProviderKind.DISCORD_CHANNEL)
    _feed(market, "haram", _good_returns(), shariah=False)
    result = market.vet("haram", NOW)
    assert any("halal compliance" in r for r in result.reasons)

    market.add_provider("empty", ProviderKind.DISCORD_CHANNEL)
    result = market.vet("empty", NOW)
    assert result.metrics.track_days == 0
    assert any("not computable" in r for r in result.reasons)


def test_losing_provider_fails_hit_rate_drawdown_and_sharpe() -> None:
    market = Marketplace()
    market.add_provider("loser", ProviderKind.ANALYST_BOARD)
    pattern = [Decimal("-0.04"), Decimal("-0.03"), Decimal("0.01")]
    _feed(market, "loser", [pattern[i % 3] for i in range(36)])
    reasons = " ".join(market.vet("loser", NOW).reasons)
    assert "hit rate" in reasons
    assert "max drawdown" in reasons
    assert "deflated Sharpe" in reasons


def test_suspension_on_revet_and_manual_hold() -> None:
    market = Marketplace()
    market.add_provider("p", ProviderKind.INTERNAL_STRATEGY)
    _feed(market, "p", _good_returns())
    assert market.vet("p", NOW).passed
    for index in range(8):
        signal_id = f"crash-{index}"
        market.record(SignalOutcome("p", signal_id, NOW + timedelta(hours=index), True, True))
        market.grade("p", signal_id, Decimal("-0.06"))
    result = market.vet("p", NOW + timedelta(days=1))
    assert result.previous is ProviderStatus.VETTED
    assert result.status is ProviderStatus.SUSPENDED
    assert any("drawdown" in r for r in result.reasons)


def test_manual_hold_is_sticky_until_reinstated() -> None:
    market = Marketplace()
    market.add_provider("p", ProviderKind.INTERNAL_STRATEGY)
    _feed(market, "p", _good_returns())
    held = market.suspend("p", "owner review", NOW)
    assert held.status is ProviderStatus.SUSPENDED and held.manual_hold
    result = market.vet("p", NOW)
    assert result.status is ProviderStatus.SUSPENDED
    assert result.reasons[0] == "manual: owner review"
    market.reinstate("p")
    assert market.vet("p", NOW).passed


def test_scheduled_revet() -> None:
    market = Marketplace(VettingPolicy(revet_interval=timedelta(days=7)))
    market.add_provider("a", ProviderKind.INTERNAL_STRATEGY)
    market.add_provider("b", ProviderKind.INTERNAL_STRATEGY)
    assert market.due_for_revet(NOW) == ("a", "b")
    first = market.revet_due(NOW)
    assert [r.provider_id for r in first] == ["a", "b"]
    assert market.due_for_revet(NOW + timedelta(days=3)) == ()
    assert market.due_for_revet(NOW + timedelta(days=7)) == ("a", "b")
    with pytest.raises(ValueError, match="timezone-aware"):
        market.due_for_revet(datetime(2026, 1, 1))
