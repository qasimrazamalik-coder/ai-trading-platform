from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from quant_os_core.agents.marketplace import (
    Marketplace,
    ProviderKind,
    SignalOutcome,
    VettingPolicy,
)
from quant_os_core.agents.strategy_selector import (
    SelectionReport,
    SelectorPolicy,
    StrategyCandidate,
    StrategySelector,
    regime_key,
)
from quant_os_core.regimes.base import RegimeEstimate, RegimeSnapshot

NOW = datetime(2026, 11, 22, tzinfo=UTC)
N = 400
BULL = "bull_trend|low_volatility"
BEAR = "bear_trend|high_volatility"
LABELS = [BULL if (i // 20) % 2 == 0 else BEAR for i in range(N)]


def _market(*vetted: str) -> Marketplace:
    lenient = VettingPolicy(
        min_signals=3,
        min_track_days=0,
        min_deflated_sharpe=0.0,
        min_hit_rate=Decimal("0"),
        max_drawdown=Decimal("1"),
        min_halal_compliance=Decimal("0"),
    )
    market = Marketplace(lenient)
    for name in vetted:
        market.add_provider(name, ProviderKind.INTERNAL_STRATEGY)
        for index, value in enumerate(("0.02", "-0.01", "0.03")):
            market.record(SignalOutcome(name, f"{name}{index}", NOW, True, True))
            market.grade(name, f"{name}{index}", Decimal(value))
        assert market.vet(name, NOW).passed
    return market


def _series(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    baseline = rng.normal(0.0002, 0.01, N)
    good = baseline + rng.normal(0.004, 0.01, N)
    overfit = baseline + np.where(np.arange(N) < 200, 0.02, -0.004) + rng.normal(0, 0.01, N)
    noise = baseline + rng.normal(0.0, 0.01, N)
    return {"baseline": baseline, "good": good, "overfit": overfit, "noise": noise}


def _snapshot() -> RegimeSnapshot:
    return RegimeSnapshot(
        trend=RegimeEstimate("bull_trend", 0.8, 0.02),
        volatility=RegimeEstimate("low_volatility", 0.7, 0.1),
        risk=RegimeEstimate("risk_on", 0.6, 0.3),
    )


def _candidates(data: dict[str, np.ndarray], *names: str) -> list[StrategyCandidate]:
    return [StrategyCandidate(name, data[name]) for name in names]


def test_selects_skilful_strategy_and_rejects_overfit_and_no_skill() -> None:
    data = _series(7)
    selector = StrategySelector(_market("good", "overfit", "noise"), SelectorPolicy(max_pbo=1.0))
    report = selector.select(
        _candidates(data, "good", "overfit", "noise"),
        baseline_returns=data["baseline"],
        regime_labels=LABELS,
        current=_snapshot(),
        now=NOW,
    )
    assert regime_key(_snapshot()) == BULL
    assert report.regime == BULL
    assert report.selected == "good"
    assert report.folds == 7
    by_id = {c.strategy_id: c for c in report.candidates}
    assert by_id["good"].accepted
    assert any("overfit" in r for r in by_id["overfit"].reasons)
    assert any("no skill" in r for r in by_id["noise"].reasons)
    assert report.explanation_source == "rules"
    assert "selected good" in report.explanation


def test_unvetted_candidates_are_never_selected() -> None:
    data = _series(7)
    selector = StrategySelector(_market())
    report = selector.select(
        _candidates(data, "good"),
        baseline_returns=data["baseline"],
        regime_labels=LABELS,
        current=BULL,
        now=NOW,
    )
    assert report.selected is None
    assert report.candidates[0].reasons == ("not a vetted marketplace provider",)
    assert "keeping the baseline" in report.explanation


def test_single_candidate_uses_plain_psr_and_rare_regime_falls_back() -> None:
    data = _series(11)
    selector = StrategySelector(_market("good"))
    report = selector.select(
        _candidates(data, "good"),
        baseline_returns=data["baseline"],
        regime_labels=LABELS,
        current="sideways|normal_volatility",  # never seen: use all fold periods
        now=NOW,
    )
    assert report.selected == "good"
    assert report.pbo is None


def test_high_pbo_rejects_selection() -> None:
    data = _series(2)
    twin = data["baseline"] + np.random.default_rng(102).normal(0.0, 0.01, N)
    selector = StrategySelector(
        _market("noise", "twin"),
        SelectorPolicy(max_pbo=0.5, min_skill_probability=0.0, max_degradation=100.0),
    )
    report = selector.select(
        [StrategyCandidate("noise", data["noise"]), StrategyCandidate("twin", twin)],
        baseline_returns=data["baseline"],
        regime_labels=LABELS,
        current=BULL,
        now=NOW,
    )
    assert report.pbo is not None and report.pbo > 0.5
    assert report.selected is None
    assert all(any("PBO" in r for r in c.reasons) for c in report.candidates)


def test_flat_returns_are_not_computable() -> None:
    data = _series(5)
    flat = StrategyCandidate("flat", np.zeros(N))
    selector = StrategySelector(_market("flat"), SelectorPolicy(pbo_slices=6))
    report = selector.select(
        [flat],
        baseline_returns=np.zeros(N),
        regime_labels=LABELS,
        current=BULL,
        now=NOW,
    )
    reasons = report.candidates[0].reasons
    assert any("not computable" in r for r in reasons)
    assert any("flat or too short" in r for r in reasons)
    assert report.selected is None
    assert data["baseline"].size == N


def test_pbo_errors_are_tolerated() -> None:
    data = _series(9)
    selector = StrategySelector(_market("good", "noise"), SelectorPolicy(pbo_slices=402))
    report = selector.select(
        _candidates(data, "good", "noise"),
        baseline_returns=data["baseline"],
        regime_labels=LABELS,
        current=BULL,
        now=NOW,
    )
    assert report.pbo is None


def test_llm_writes_explanation_only() -> None:
    data = _series(7)
    seen: list[SelectionReport] = []

    def llm(report: SelectionReport) -> str:
        seen.append(report)
        return "  I recommend choosing noise instead.  " + "x" * 5000

    selector = StrategySelector(
        _market("good", "noise"), SelectorPolicy(max_pbo=1.0), explainer=llm
    )
    report = selector.select(
        _candidates(data, "good", "noise"),
        baseline_returns=data["baseline"],
        regime_labels=LABELS,
        current=BULL,
        now=NOW,
    )
    assert report.selected == "good"
    assert report.explanation_source == "llm"
    assert report.explanation.startswith("I recommend")
    assert len(report.explanation) == 2000
    assert seen[0].selected == "good"


@pytest.mark.parametrize("bad", [lambda r: 1 / 0, lambda r: "   ", lambda r: 42])
def test_broken_llm_counts_as_no_opinion(bad: object) -> None:
    data = _series(7)
    selector = StrategySelector(
        _market("good"),
        explainer=bad,  # type: ignore[arg-type]
    )
    report = selector.select(
        _candidates(data, "good"),
        baseline_returns=data["baseline"],
        regime_labels=LABELS,
        current=BULL,
        now=NOW,
    )
    assert report.explanation_source == "rules"
    assert report.selected == "good"


def test_input_validation() -> None:
    data = _series(7)
    selector = StrategySelector(_market("good"))
    kwargs: dict[str, object] = {
        "baseline_returns": data["baseline"],
        "regime_labels": LABELS,
        "current": BULL,
        "now": NOW,
    }
    with pytest.raises(ValueError, match="timezone-aware"):
        selector.select([], **{**kwargs, "now": datetime(2026, 1, 1)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite 1-D"):
        selector.select([], **{**kwargs, "baseline_returns": [np.nan] * N})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="align with baseline_returns"):
        selector.select([], **{**kwargs, "regime_labels": LABELS[:-1]})  # type: ignore[arg-type]
    short = StrategyCandidate("good", data["good"][:-1])
    with pytest.raises(ValueError, match="must align with baseline"):
        selector.select([short], **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite 1-D"):
        StrategyCandidate("x", np.array([np.inf, 1.0]))
    for bad in (
        {"train_size": 2},
        {"min_regime_periods": 2},
        {"max_degradation": -1.0},
        {"max_pbo": 2.0},
    ):
        with pytest.raises(ValueError):
            SelectorPolicy(**bad)  # type: ignore[arg-type]
