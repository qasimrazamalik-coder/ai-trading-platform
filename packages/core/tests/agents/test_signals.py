import inspect
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from quant_os_core.agents import signals as signals_module
from quant_os_core.agents.board import AgentBoard
from quant_os_core.agents.exits import Bar
from quant_os_core.agents.signals import (
    BOARD_SYMBOL,
    SIGNAL_TITLE,
    HoldingsSnapshot,
    SignalKind,
    compute_signals,
    holdings_from_account,
    holdings_from_store,
    load_screening,
    publish_signals,
    summarize,
)
from quant_os_core.execution.models import AccountSnapshot, BrokerPosition
from quant_os_core.instruments import InstrumentId
from quant_os_core.ops.state import StateStore
from quant_os_core.strategies.diversified_core import CoreMode, target_weights

TODAY = date(2026, 11, 22)
START = datetime(2026, 1, 1, tzinfo=UTC)


def _bars(closes: list[float]) -> list[Bar]:
    return [
        Bar(START + timedelta(days=i), Decimal(str(c)) * 2, Decimal(str(c)) / 2, Decimal(str(c)))
        for i, c in enumerate(closes)
    ]


def _rising(n: int = 260, start: float = 100.0) -> list[float]:
    return [round(start + 0.1 * i + (0.3 if i % 2 else 0.0), 4) for i in range(n)]


class Sink:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, level: str, title: str, message: str) -> None:
        self.sent.append((level, title, message))


def test_rebalance_due_once_per_month() -> None:
    bars = {"SPUS": _bars(_rising()), "HLAL": _bars(_rising(start=50))}
    never = compute_signals(bars, HoldingsSnapshot(Decimal(100), {}), TODAY)
    assert SignalKind.REBALANCE_DUE in {s.kind for s in never}
    done = HoldingsSnapshot(Decimal(100), {}, last_rebalance=date(2026, 11, 2))
    assert SignalKind.REBALANCE_DUE not in {s.kind for s in compute_signals(bars, done, TODAY)}
    old = HoldingsSnapshot(Decimal(100), {}, last_rebalance=date(2026, 10, 30))
    due = [s for s in compute_signals(bars, old, TODAY) if s.kind is SignalKind.REBALANCE_DUE]
    assert due and due[0].symbol == BOARD_SYMBOL and "2026-10-30" in due[0].detail


def test_drift_alert_only_outside_the_band() -> None:
    bars = {"SPUS": _bars(_rising()), "HLAL": _bars(_rising(start=50))}
    cash_only = HoldingsSnapshot(Decimal(100), {}, last_rebalance=date(2026, 11, 1))
    drift = [s for s in compute_signals(bars, cash_only, TODAY) if s.kind is SignalKind.DRIFT_ALERT]
    assert {s.symbol for s in drift} == {"SPUS", "HLAL"}
    assert all(s.level == "warning" and "target" in s.detail for s in drift)

    weights = target_weights({s: [float(b.close) for b in v] for s, v in bars.items()})
    values = {s: w * Decimal(1000) for s, w in weights.items()}
    cash = Decimal(1000) - sum(values.values(), Decimal(0))
    on_target = HoldingsSnapshot(cash, values, last_rebalance=date(2026, 11, 1))
    assert not [
        s for s in compute_signals(bars, on_target, TODAY) if s.kind is SignalKind.DRIFT_ALERT
    ]


def test_too_little_history_is_reported_not_guessed() -> None:
    short = {"SPUS": _bars(_rising(50))}
    out = compute_signals(short, HoldingsSnapshot(Decimal(1), {}, date(2026, 11, 1)), TODAY)
    assert [s.kind for s in out] == [SignalKind.DRIFT_ALERT]
    assert "weights unavailable" in out[0].detail


def test_trend_entry_and_exit_on_200_day_crossings_in_defensive_mode() -> None:
    flat = [100.0 + (0.5 if i % 2 else -0.5) for i in range(220)]
    entry = [*flat[:-2], 99.0, 110.0]
    exit_ = [*flat[:-2], 101.0, 90.0]
    bars = {"SPUS": _bars(entry), "HLAL": _bars(exit_)}
    holdings = HoldingsSnapshot(Decimal(100), {}, date(2026, 11, 1))
    out = {s.symbol: s for s in compute_signals(bars, holdings, TODAY) if "TREND" in s.kind}
    assert out["SPUS"].kind is SignalKind.TREND_ENTRY and "above" in out["SPUS"].detail
    assert out["HLAL"].kind is SignalKind.TREND_EXIT and out["HLAL"].level == "warning"
    growth = compute_signals(bars, holdings, TODAY, mode=CoreMode.GROWTH)
    assert not [s for s in growth if "TREND" in s.kind]  # growth mode has no trend filter


def _config(tmp_path: Path) -> Path:
    fin = {
        "sector": "diversified_fund",
        "market_cap": "100",
        "interest_bearing_debt": "1",
        "interest_bearing_cash_and_securities": "1",
        "total_revenue": "10",
        "noncompliant_revenue": "0.1",
        "total_assets": "50",
        "tangible_assets": "40",
        "shares_outstanding": "1",
    }
    config = {
        "universe": [
            {"symbol": "SPUS", "financials": fin, "financials_source": "SEC N-PORT 2026-08-31"},
            {"symbol": "HLAL", "financials": fin, "financials_source": "SEC N-PORT 2026-01-31"},
            {"symbol": "SPRE", "financials": fin, "financials_date": "2026-11-01"},
            {"symbol": "UMMA"},
        ]
    }
    path = tmp_path / "core.json"
    path.write_text(json.dumps(config), "utf-8")
    return path


def test_screen_stale_after_120_days_or_missing(tmp_path: Path) -> None:
    screening = load_screening(_config(tmp_path))
    assert screening["SPUS"].as_of == date(2026, 8, 31)
    assert screening["SPRE"].as_of == date(2026, 11, 1)
    assert screening["UMMA"].financials is None
    bars = {"SPUS": _bars(_rising())}
    holdings = HoldingsSnapshot(Decimal(0), {}, date(2026, 11, 1))
    stale = {
        s.symbol: s
        for s in compute_signals(bars, holdings, TODAY, screening=screening)
        if s.kind is SignalKind.SCREEN_STALE
    }
    assert set(stale) == {"HLAL", "UMMA"}  # SPUS is 83 days old, SPRE 21: both fresh
    assert "295 days old" in stale["HLAL"].detail
    assert "blocks it" in stale["UMMA"].detail


def test_summaries_in_three_languages() -> None:
    bars = {"SPUS": _bars(_rising()), "HLAL": _bars(_rising(start=50))}
    found = compute_signals(bars, HoldingsSnapshot(Decimal(100), {}), TODAY)
    english = summarize(found, "en")
    urdu = summarize(found, "ur")
    arabic = summarize(found, "ar")
    assert english.startswith("Core signals (information only")
    assert urdu.startswith(signals_module._HEADER["ur"]) and "SPUS" in urdu
    assert arabic.startswith(signals_module._HEADER["ar"]) and "HLAL" in arabic
    assert summarize([], "ur") == signals_module._NONE["ur"]
    with pytest.raises(ValueError):
        summarize(found, "fr")


def test_publish_posts_to_board_and_alerts_and_touches_no_trading_state(tmp_path: Path) -> None:
    board = AgentBoard(tmp_path / "board.db")
    store = StateStore(tmp_path / "core.db")
    store.set_value("core:last_rebalance", "2026-10-01")
    sink = Sink()
    bars = {"SPUS": _bars(_rising()), "HLAL": _bars(_rising(start=50))}
    holdings = holdings_from_store(store, {}, Decimal(100))
    assert holdings.last_rebalance == date(2026, 10, 1) and holdings.values == {}
    found = compute_signals(bars, holdings, TODAY)
    before = [store.recent(t) for t in ("decisions", "closed_trades")]

    result = publish_signals(found, board=board, alerts=sink, lang="en")

    tasks = board.open_tasks()
    assert result.task_id == tasks[0]["id"] and tasks[0]["symbol"] == BOARD_SYMBOL
    assert "nothing executes" in tasks[0]["question"]
    assert sink.sent == [("warning", SIGNAL_TITLE, result.summary)]
    assert [store.recent(t) for t in ("decisions", "closed_trades")] == before
    assert store.open_positions() == [] and store.pending() == []
    assert not board.proposals()  # a signal never becomes a proposal or an order
    empty = publish_signals([], board=board, alerts=sink)
    assert empty.task_id is None and not empty.alerted and len(sink.sent) == 1


def test_holdings_from_broker_snapshot() -> None:
    account = AccountSnapshot(
        cash=Decimal("10"),
        equity=Decimal("40"),
        buying_power=Decimal("10"),
        positions=(
            BrokerPosition(
                InstrumentId(venue="ALPACA", symbol="SPUS"),
                Decimal("2"),
                Decimal("14"),
                Decimal("15"),
            ),
        ),
        timestamp=START,
    )
    holdings = holdings_from_account(account)
    assert holdings.values == {"SPUS": Decimal("30")} and holdings.equity == Decimal("40")
    assert holdings.last_rebalance is None


def test_module_has_no_order_path() -> None:
    source = inspect.getsource(signals_module)
    for word in ("submit(", "ExecutionBroker", "record_proposal", "EntryPipeline"):
        assert word not in source
