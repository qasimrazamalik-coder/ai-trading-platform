from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from quant_os_core.agents.exits import Bar
from quant_os_core.backtesting.models import OrderSide
from quant_os_core.execution.broker import PaperBroker
from quant_os_core.execution.models import BrokerOrderRequest
from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.ops.alerts import StoreSink
from quant_os_core.ops.core_loop import CoreRebalancer
from quant_os_core.ops.paper_loop import UniverseItem
from quant_os_core.ops.signals import CoreTrendSignal
from quant_os_core.ops.state import StateStore
from quant_os_core.risk_engine import ProtectionManager, RiskEngine
from quant_os_core.shariah import (
    CompanyFinancials,
    ShariahGate,
    ShariahProfile,
    halal_risk_limits,
)
from quant_os_core.strategies.diversified_core import CoreMode

T0 = datetime(2026, 1, 2, 21, tzinfo=UTC)
CLEAN = CompanyFinancials(
    sector="technology",
    market_cap=Decimal("1000"),
    interest_bearing_debt=Decimal("10"),
    interest_bearing_cash_and_securities=Decimal("10"),
    total_revenue=Decimal("100"),
    noncompliant_revenue=Decimal("1"),
    total_assets=Decimal("500"),
    tangible_assets=Decimal("400"),
    shares_outstanding=Decimal("10"),
)
SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")


def _etf(symbol: str) -> Instrument:
    return Instrument(
        instrument_id=InstrumentId(venue="PAPER", symbol=symbol),
        asset_class=AssetClass.ETF,
        instrument_type=InstrumentType.SPOT,
        base_currency=symbol,
        quote_currency="USD",
        price_precision=2,
        quantity_precision=4,
    )


def _series(seed: int, n: int = 260, drift: float = 0.001) -> list[Bar]:
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return [
        Bar(
            T0 - timedelta(days=n - i),
            Decimal(f"{c * 1.005:.4f}"),
            Decimal(f"{c * 0.995:.4f}"),
            Decimal(f"{c:.4f}"),
        )
        for i, c in enumerate(closes)
    ]


def _rebalancer(
    tmp_path: Path, broker: PaperBroker, financials: CompanyFinancials | None = CLEAN
) -> CoreRebalancer:
    store = StateStore(tmp_path / "core.db")
    return CoreRebalancer(
        broker=broker,
        store=store,
        universe={s: UniverseItem(_etf(s), financials=financials) for s in SYMBOLS},
        shariah=ShariahGate(ShariahProfile.spot_crypto_and_day_trading()),
        protections=ProtectionManager([]),
        risk=RiskEngine(halal_risk_limits()),
        alerts=StoreSink(store),
        mode=CoreMode.GROWTH,
    )


def _setup(broker: PaperBroker, bars: dict[str, Sequence[Bar]]) -> None:
    for symbol, series in bars.items():
        broker.update_mark(InstrumentId(venue="PAPER", symbol=symbol), series[-1].close)


def _broker(cash: str = "10000") -> PaperBroker:
    return PaperBroker(initial_cash=Decimal(cash), fee_rate=Decimal("0"), slippage_bps=Decimal("0"))


def test_core_buys_every_fund_unlevered_then_waits_for_next_month(tmp_path: Path) -> None:
    broker = _broker()
    bars = {s: _series(i) for i, s in enumerate(SYMBOLS)}
    _setup(broker, bars)
    core = _rebalancer(tmp_path, broker)
    report = core.step(T0, bars)
    assert sorted(report.entries_submitted) == sorted(SYMBOLS)
    account = broker.account()
    assert account.cash >= 0  # never borrowed
    assert all(
        p.quantity * bars[p.instrument_id.symbol][-1].close <= Decimal("2000.01")
        for p in account.positions
    )  # risk engine caps a fund at 20% of equity
    # same month: nothing to do
    again = core.step(T0 + timedelta(days=5), bars)
    assert again.entries_submitted == [] and again.exits == []


def test_core_blocks_unscreened_funds_at_the_shariah_gate(tmp_path: Path) -> None:
    broker = _broker()
    bars = {s: _series(i) for i, s in enumerate(SYMBOLS)}
    _setup(broker, bars)
    core = _rebalancer(tmp_path, broker, financials=None)
    report = core.step(T0, bars)
    assert report.entries_submitted == []
    assert set(report.blocked.values()) == {"shariah R6"}
    assert not broker.account().positions


def test_core_sells_a_fund_whose_target_drops_to_zero(tmp_path: Path) -> None:
    broker = _broker()
    bars = {s: _series(i) for i, s in enumerate(SYMBOLS)}
    _setup(broker, bars)
    core = _rebalancer(tmp_path, broker)
    core.step(T0, bars)
    # next month, in defensive mode, AAA has crashed below its 200-day average
    core.mode = CoreMode.DEFENSIVE
    crash = list(bars["AAA"])
    last = crash[-1].close
    for k in range(1, 31):
        p = last * Decimal(1 - 0.01 * k)
        crash.append(Bar(T0 + timedelta(days=k), p * Decimal("1.005"), p * Decimal("0.995"), p))
    bars2 = {s: (crash if s == "AAA" else bars[s]) for s in SYMBOLS}
    _setup(broker, bars2)
    report = core.step(T0 + timedelta(days=31), bars2)
    assert "AAA" in report.exits
    held = {p.instrument_id.symbol for p in broker.account().positions if p.quantity > 0}
    assert "AAA" not in held


def test_core_pauses_on_foreign_positions_and_kill_switch_flattens(tmp_path: Path) -> None:
    broker = _broker()
    bars = {s: _series(i) for i, s in enumerate(SYMBOLS)}
    _setup(broker, bars)
    core = _rebalancer(tmp_path, broker)
    core.step(T0, bars)
    assert broker.account().positions
    # someone buys a fund outside the core in this account: the core stops trading
    foreign = InstrumentId(venue="PAPER", symbol="ZZZ")
    broker.update_mark(foreign, Decimal("10"))
    broker.submit(
        BrokerOrderRequest(
            client_order_id="manual-1", instrument_id=foreign, side=OrderSide.BUY, quantity=1
        )
    )
    paused = core.step(T0 + timedelta(days=31), bars)
    assert paused.unknown_state and paused.entries_submitted == []
    assert any("not in the core universe" in i for i in paused.issues)
    core.universe = {**core.universe, "ZZZ": UniverseItem(_etf("ZZZ"), financials=CLEAN)}
    core.store.set_kill_switch(True, "test")
    report = core.step(T0 + timedelta(days=1), bars)
    assert report.killed and sorted(report.exits) == sorted([*SYMBOLS, "ZZZ"])
    assert all(p.quantity == 0 for p in broker.account().positions)


def test_core_fails_closed_without_full_data(tmp_path: Path) -> None:
    broker = _broker()
    bars = {s: _series(i) for i, s in enumerate(SYMBOLS)}
    _setup(broker, bars)
    core = _rebalancer(tmp_path, broker)
    partial = {s: b for s, b in bars.items() if s != "FFF"}
    report = core.step(T0, partial)
    assert report.entries_submitted == [] and any("FFF" in i for i in report.issues)


def test_core_trend_signal_plans_pass_plan_validation_bounds() -> None:
    signal = CoreTrendSignal()
    plan = signal("AAA", _series(1, drift=0.002))
    assert plan is not None
    risk = (plan.entry_price - plan.stop_loss) / plan.entry_price
    assert Decimal("0.01") < risk < Decimal("0.15")
    reward = plan.profit_target - plan.entry_price
    assert abs(reward - 2 * (plan.entry_price - plan.stop_loss)) < Decimal("1e-9")
    assert signal("AAA", _series(1, drift=-0.003)) is None  # below its average: no plan


def test_runner_builds_both_core_modes_from_example_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from quant_os_core.ops.paper_loop import PaperTrader
    from quant_os_core.ops.run_paper import build_trader

    monkeypatch.setenv("APCA_API_KEY_ID", "paper-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "paper-secret")
    examples = Path(__file__).resolve().parents[4] / "examples"
    base = {"state": str(tmp_path / "c.db"), "board": str(tmp_path / "b.db")}
    shariah = json.loads((examples / "paper_core.json").read_text()) | base
    core, _ = build_trader(shariah)
    assert isinstance(core, CoreRebalancer) and core.mode is CoreMode.DEFENSIVE
    assert set(core.universe) == {"SPUS", "HLAL", "UMMA", "SPRE", "SPSK", "AAAU"}
    original = json.loads((examples / "paper_core_original.json").read_text()) | base
    plan_mode = original | {"core": {**original["core"], "gate": "trade_plan"}}
    trader, _ = build_trader(plan_mode)
    assert isinstance(trader, PaperTrader) and isinstance(trader.signal, CoreTrendSignal)
    assert len(trader.universe) == 11
    with pytest.raises(ValueError, match=r"core.gate"):
        build_trader(original | {"core": {"gate": "yolo"}})


class _FlakyBroker(PaperBroker):
    """Submit lands at the broker but the response is lost (a timeout after the POST)."""

    def submit(self, request: BrokerOrderRequest):  # type: ignore[no-untyped-def]
        super().submit(request)
        raise TimeoutError("response lost")


def test_core_does_not_trade_again_after_an_ambiguous_submit(tmp_path: Path) -> None:
    broker = _FlakyBroker(
        initial_cash=Decimal("10000"), fee_rate=Decimal("0"), slippage_bps=Decimal("0")
    )
    bars = {s: _series(i) for i, s in enumerate(SYMBOLS)}
    _setup(broker, bars)
    core = _rebalancer(tmp_path, broker)
    core.broker_breaker.failure_threshold = 100  # keep the breaker closed for this test
    first = core.step(T0, bars)
    assert first.entries_submitted == [] and any("order failed" in i for i in first.issues)
    assert core.store.get_value("core:last_rebalance") == ""  # month not marked done
    # the timed-out orders did fill; the next step sees them and does not buy twice
    second = core.step(T0 + timedelta(minutes=5), bars)
    assert second.entries_submitted == [] and len(second.fills) == len(SYMBOLS)
    held = {p.instrument_id.symbol: p.quantity for p in broker.account().positions}
    third = core.step(T0 + timedelta(minutes=10), bars)
    after = {p.instrument_id.symbol: p.quantity for p in broker.account().positions}
    assert after == held and third.entries_submitted == []  # inside the band: no duplicates
