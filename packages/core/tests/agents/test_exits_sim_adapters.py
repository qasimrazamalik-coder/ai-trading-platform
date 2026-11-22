import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from quant_os_core.agents import (
    AnalystCouncil,
    EntryPipeline,
    PlanGovernor,
    PlanPolicy,
    RuleAnalyst,
    TradePlan,
)
from quant_os_core.agents.exits import Bar, ExitManager, InvalidationRule, OpenPosition, RuleKind
from quant_os_core.agents.ollama import OllamaAnalyst
from quant_os_core.agents.simulator import ExitPolicy, simulate
from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.market_data.free_sources import AlpacaBars, EdgarFundamentals
from quant_os_core.ml.meta_labeling import (
    BarrierSpec,
    basic_features,
    rolling_volatility,
    triple_barrier_labels,
    walk_forward_meta_label,
)
from quant_os_core.risk_engine import ExitReason, ProtectionManager, RiskEngine, StoplossGuard
from quant_os_core.shariah import CompanyFinancials, ShariahGate, halal_risk_limits

T0 = datetime(2026, 1, 5, 21, tzinfo=UTC)


def _plan(entry: str = "100", stop: str = "95", target: str = "110") -> TradePlan:
    return TradePlan(
        symbol="AAPL",
        entry_price=Decimal(entry),
        profit_target=Decimal(target),
        stop_loss=Decimal(stop),
        invalidation_condition="close below the 20-day average",
        confidence=Decimal("0.7"),
        justification="test",
    )


def _bar(day: int, high: str, low: str, close: str) -> Bar:
    return Bar(T0 + timedelta(days=day), Decimal(high), Decimal(low), Decimal(close))


def test_exit_priority_stop_target_trailing_time_and_invalidation() -> None:
    mgr = ExitManager()
    pos = OpenPosition(_plan(), Decimal("1"), T0)
    event = mgr.on_bar(pos, _bar(1, "111", "94", "100"), ())
    assert event is not None and event.reason is ExitReason.STOP_LOSS  # worst case first
    pos = OpenPosition(_plan(), Decimal("1"), T0)
    event = mgr.on_bar(pos, _bar(1, "111", "99", "109"), ())
    assert (
        event is not None
        and event.reason is ExitReason.TAKE_PROFIT
        and event.price == Decimal("110")
    )
    trail = OpenPosition(_plan(target="200"), Decimal("1"), T0, trail_fraction=Decimal("0.05"))
    assert mgr.on_bar(trail, _bar(1, "120", "101", "119"), ()) is None
    assert trail.stop == Decimal("114.00")
    event = mgr.on_bar(trail, _bar(2, "118", "113", "113"), ())
    assert event is not None and event.reason is ExitReason.TAKE_PROFIT  # locked-in gain
    timed = OpenPosition(_plan(), Decimal("1"), T0, max_holding=timedelta(days=3))
    assert mgr.on_bar(timed, _bar(3, "101", "99", "100"), ()).reason is ExitReason.TIME  # type: ignore[union-attr]
    rule = InvalidationRule(RuleKind.CLOSE_BELOW_SMA, window=3)
    inval = OpenPosition(_plan(), Decimal("1"), T0, rules=(rule,))
    closes = (Decimal("104"), Decimal("103"), Decimal("97"))
    assert mgr.on_bar(inval, _bar(1, "104", "96", "97"), closes).reason is ExitReason.INVALIDATION  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        InvalidationRule(RuleKind.CLOSE_BELOW)
    with pytest.raises(ValueError):
        Bar(T0, Decimal("1"), Decimal("2"), Decimal("1.5"))


def _instrument() -> Instrument:
    return Instrument(
        instrument_id=InstrumentId(venue="alpaca", symbol="AAPL"),
        asset_class=AssetClass.EQUITY,
        instrument_type=InstrumentType.SPOT,
        base_currency="AAPL",
        quote_currency="USD",
        price_precision=2,
        quantity_precision=4,
    )


FIN = CompanyFinancials(
    sector="technology",
    market_cap=Decimal("1000"),
    interest_bearing_debt=Decimal("100"),
    interest_bearing_cash_and_securities=Decimal("50"),
    total_revenue=Decimal("400"),
    noncompliant_revenue=Decimal("4"),
    total_assets=Decimal("900"),
    tangible_assets=Decimal("600"),
    shares_outstanding=Decimal("100"),
)


def _pipeline() -> EntryPipeline:
    return EntryPipeline(
        shariah=ShariahGate(),
        protections=ProtectionManager(
            [
                StoplossGuard(
                    trade_limit=2, lookback=timedelta(days=30), stop_duration=timedelta(days=30)
                )
            ]
        ),
        governor=PlanGovernor(PlanPolicy()),
        council=AnalystCouncil([RuleAnalyst("trend", Decimal("0.6"))]),
        risk=RiskEngine(halal_risk_limits()),
    )


def _falling_bars(n: int = 40) -> list[Bar]:
    bars, price = [], Decimal("100")
    for d in range(n):
        price = price * Decimal("0.97")
        bars.append(
            Bar(T0 + timedelta(days=d), price * Decimal("1.01"), price * Decimal("0.99"), price)
        )
    return bars


def test_simulator_feeds_losses_back_into_protections() -> None:
    def always_buy(history: Sequence[Bar]) -> TradePlan | None:
        c = history[-1].close
        return _plan(
            str(c),
            str((c * Decimal("0.97")).quantize(Decimal("0.01"))),
            str((c * Decimal("1.06")).quantize(Decimal("0.01"))),
        )

    result = simulate(
        _falling_bars(),
        always_buy,
        _pipeline(),
        instrument=_instrument(),
        financials=FIN,
        starting_cash=Decimal("50"),
        exit_policy=ExitPolicy(trail_fraction=None, max_holding=None),
    )
    assert result.trades, "expected some stopped-out trades"
    assert all(t.exit_reason is ExitReason.STOP_LOSS for t in result.trades)
    # StoplossGuard locks the account for 30 days after the second stop, then relocks
    first_lock_end = result.trades[1].exit_time + timedelta(days=30)
    assert all(
        t.exit_time <= result.trades[1].exit_time or t.exit_time > first_lock_end
        for t in result.trades
    )
    assert len(result.trades) == 4
    assert result.blocked_by.get("protections", 0) > 0
    assert result.total_return < 0 and result.win_rate == 0
    assert all(value > 0 for value in result.equity_curve)


def test_alpaca_bars_paginates() -> None:
    pages = [
        {
            "bars": [{"t": "2026-01-05T05:00:00Z", "h": 10, "l": 9, "c": 9.5}],
            "next_page_token": "x",
        },
        {
            "bars": [{"t": "2026-01-06T05:00:00Z", "h": 11, "l": 10, "c": 10.5}],
            "next_page_token": None,
        },
    ]
    seen: list[str] = []

    def fake(url: str, headers: Mapping[str, str]) -> bytes:
        seen.append(url)
        assert headers["APCA-API-KEY-ID"] == "k"
        return json.dumps(pages[len(seen) - 1]).encode()

    bars = AlpacaBars("k", "s", fetch=fake).daily("aapl", "2026-01-01", "2026-01-31")
    assert [b.close for b in bars] == [Decimal("9.5"), Decimal("10.5")]
    assert "page_token=x" in seen[1] and "/AAPL/" in seen[0]


def test_edgar_builds_screenable_financials() -> None:
    def fact(val: int) -> dict[str, object]:
        return {
            "units": {"USD": [{"end": "2025-09-30", "val": val}, {"end": "2024-09-30", "val": 1}]}
        }

    facts = {
        "facts": {
            "us-gaap": {
                "Assets": fact(900),
                "Revenues": fact(400),
                "LongTermDebt": fact(80),
                "DebtCurrent": fact(20),
                "CashAndCashEquivalentsAtCarryingValue": fact(50),
                "Goodwill": fact(200),
                "IntangibleAssetsNetExcludingGoodwill": fact(100),
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [{"end": "2025-10-20", "val": 100}]}
                }
            },
        }
    }
    edgar = EdgarFundamentals(
        "test you@example.com", fetch=lambda url, h: json.dumps(facts).encode()
    )
    fin = edgar.financials(
        320193, market_cap=Decimal("1000"), sector="technology", noncompliant_revenue=Decimal("4")
    )
    assert fin.interest_bearing_debt == Decimal("100")
    assert fin.tangible_assets == Decimal("600")
    assert ShariahGate().screen_company(fin).allowed


def test_ollama_analyst_parses_and_fails_closed() -> None:
    good = {
        "message": {
            "content": json.dumps(
                {
                    "analyst": "news",
                    "stance": "bearish",
                    "conviction": "0.8",
                    "rationale": "guidance cut",
                }
            )
        }
    }
    analyst = OllamaAnalyst("news", post=lambda url, body: json.dumps(good).encode())
    report = analyst.review(_plan())
    assert report.stance.value == "bearish" and report.conviction == Decimal("0.8")
    broken = OllamaAnalyst("news", post=lambda url, body: b"<html>error</html>")
    assert broken.review(_plan()).conviction == Decimal("0")

    def offline(url: str, body: bytes) -> bytes:
        raise OSError("connection refused")

    assert (
        OllamaAnalyst("tech", post=offline).review(_plan()).rationale.startswith("no valid reply")
    )
    sent: list[dict[str, object]] = []

    def capture(url: str, body: bytes) -> bytes:
        sent.append(json.loads(body))
        return json.dumps(good).encode()

    OllamaAnalyst("news", post=capture).review(_plan())
    assert sent[0]["options"] == {"temperature": 0} and "format" in sent[0]


def test_triple_barrier_and_walk_forward_meta_label() -> None:
    sklearn = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(3)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 1500))
    vol = rolling_volatility(close)
    entries = np.arange(220, 1450, 5, dtype=np.int64)
    labels, exits = triple_barrier_labels(close, entries, vol, BarrierSpec(max_holding=10))
    assert set(np.unique(labels)) <= {0, 1}
    assert np.all(exits > entries) and np.all(exits - entries <= 10)
    features = basic_features(close, vol, entries)
    result = walk_forward_meta_label(
        features,
        labels,
        lambda: sklearn.LogisticRegression(max_iter=500),
        train_size=80,
        test_size=20,
        gap=2,
    )
    assert result.tested.any()
    tested = result.probabilities[result.tested]
    assert np.all((tested >= 0) & (tested <= 1))
    assert not result.tested[:80].any()  # nothing predicted inside the first training window
    with pytest.raises(ValueError):
        triple_barrier_labels(close, np.array([0], dtype=np.int64), vol, BarrierSpec())
