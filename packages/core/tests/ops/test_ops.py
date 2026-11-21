import json
import threading
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from quant_os_core.agents.board import AgentBoard, AnalystRole, BoardAnalyst
from quant_os_core.agents.council import AnalystCouncil, RuleAnalyst, Stance
from quant_os_core.agents.day_trading import Market, flatten_time, session_close, settlement_time
from quant_os_core.agents.exits import Bar, InvalidationRule, OpenPosition, RuleKind
from quant_os_core.agents.pipeline import EntryPipeline
from quant_os_core.agents.profit_engine import CostModel, RiskDial, dial_policy
from quant_os_core.agents.simulator import ExitPolicy
from quant_os_core.agents.trade_plan import PlanGovernor, TradePlan
from quant_os_core.backtesting.models import OrderSide, OrderStatus
from quant_os_core.execution.broker import PaperBroker
from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.ml.champion import compare, run_gate
from quant_os_core.ops.alerts import MultiSink, NtfySink, StoreSink, WebhookSink
from quant_os_core.ops.alpaca_paper import AlpacaPaperBroker
from quant_os_core.ops.dashboard import make_handler, snapshot
from quant_os_core.ops.orders import client_order_id, entry_order
from quant_os_core.ops.paper_loop import PaperTrader, UniverseItem
from quant_os_core.ops.resilience import (
    BreakerOpenError,
    BreakerState,
    CircuitBreaker,
    TransientError,
    retry,
)
from quant_os_core.ops.signals import DonchianBreakout
from quant_os_core.ops.state import StateStore
from quant_os_core.risk_engine import ClosedTrade, ExitReason, ProtectionManager, RiskEngine
from quant_os_core.shariah import CryptoAsset, ShariahGate, ShariahProfile, halal_risk_limits

T0 = datetime(2026, 11, 23, 15, 0, tzinfo=UTC)


def _plan(symbol: str = "BTC/USD", entry: str = "100", stop: str = "96", target: str = "110"):  # type: ignore[no-untyped-def]
    return TradePlan(
        symbol=symbol,
        entry_price=Decimal(entry),
        profit_target=Decimal(target),
        stop_loss=Decimal(stop),
        invalidation_condition="daily close below the 50-day average",
        confidence=Decimal("0.6"),
        justification="test",
    )


# ---- state ----
def test_state_round_trip_restores_positions_exactly(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s" / "trading.db")
    pos = OpenPosition(
        _plan(),
        Decimal("0.25"),
        T0,
        rules=(InvalidationRule(RuleKind.CLOSE_BELOW, level=Decimal("95")),),
        trail_fraction=Decimal("0.05"),
        max_holding=timedelta(days=3),
        flatten_at=T0 + timedelta(hours=6),
    )
    pos.stop, pos.highest = Decimal("101.5"), Decimal("107")
    store.save_position(pos, "crypto")
    restored = StateStore(tmp_path / "s" / "trading.db").open_positions()[0]  # "after a crash"
    p = restored.position
    assert (p.stop, p.highest, p.quantity, p.max_holding) == (
        Decimal("101.5"),
        Decimal("107"),
        Decimal("0.25"),
        timedelta(days=3),
    )
    assert p.rules[0].level == Decimal("95") and p.flatten_at == T0 + timedelta(hours=6)
    store.record_close(
        ClosedTrade("BTC/USD", T0, Decimal("0.02"), ExitReason.TAKE_PROFIT),
        Decimal("0.25"),
        Decimal("100"),
        Decimal("102"),
    )
    assert store.closed_trades()[0].profit_ratio == Decimal("0.02")
    store.set_kill_switch(True, "test")
    assert store.kill_switch_on() and store.recent("events")[0]["kind"] == "kill_switch"
    with pytest.raises(ValueError):
        store.recent("kv")


# ---- resilience ----
def test_retry_backs_off_then_raises() -> None:
    calls, sleeps = [], []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("429")
        return "ok"

    assert retry(flaky, sleep=sleeps.append, jitter=lambda: 1.0) == "ok"
    assert sleeps == [0.5, 1.0]
    with pytest.raises(TransientError):
        retry(lambda: (_ for _ in ()).throw(TransientError("x")), attempts=2, sleep=lambda _s: None)
    with pytest.raises(ValueError):  # non-transient errors are not retried
        retry(lambda: (_ for _ in ()).throw(ValueError("bad")), sleep=lambda _s: None)


def test_circuit_breaker_opens_half_opens_and_closes() -> None:
    now = [0.0]
    breaker = CircuitBreaker("broker", failure_threshold=2, reset_after=60, clock=lambda: now[0])

    def boom() -> None:
        raise ConnectionError("down")

    for _ in range(2):
        with pytest.raises(ConnectionError):
            breaker.call(boom)
    assert breaker.state is BreakerState.OPEN
    with pytest.raises(BreakerOpenError):
        breaker.call(lambda: 1)
    now[0] = 61
    assert breaker.state is BreakerState.HALF_OPEN
    with pytest.raises(ConnectionError):
        breaker.call(boom)  # failed trial re-opens immediately
    assert breaker.state is BreakerState.OPEN
    now[0] = 130
    assert breaker.call(lambda: 5) == 5 and breaker.healthy


# ---- alpaca paper adapter ----
class FakeAlpaca:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, object]] = {}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None):  # type: ignore[no-untyped-def]
        path = url.removeprefix("https://paper-api.alpaca.markets")
        self.calls.append((method, path))
        if method == "POST" and path == "/v2/orders":
            data = json.loads(body or b"{}")
            order = {
                **data,
                "id": f"id-{len(self.orders)}",
                "status": "filled",
                "filled_qty": data["qty"],
                "filled_avg_price": "100.5",
            }
            self.orders[str(data["client_order_id"])] = order
            return 200, json.dumps(order).encode()
        if path.startswith("/v2/orders:by_client_order_id"):
            cid = path.split("client_order_id=")[1]
            order = self.orders.get(cid)
            return (200, json.dumps(order).encode()) if order else (404, b"{}")
        if path == "/v2/account":
            return 200, json.dumps({"cash": "40.00", "equity": "50.05"}).encode()
        if path == "/v2/positions":
            return 200, json.dumps(
                [
                    {
                        "symbol": "BTCUSD",
                        "asset_class": "crypto",
                        "qty": "0.1",
                        "avg_entry_price": "100.5",
                        "current_price": "100.5",
                    }
                ]
            ).encode()
        return 500, b"oops"


def test_alpaca_adapter_maps_orders_idempotently_and_refuses_live() -> None:
    fake = FakeAlpaca()
    broker = AlpacaPaperBroker("k", "s", http=fake)
    plan = _plan()
    req = entry_order(
        plan, InstrumentId(venue="ALPACA", symbol="BTC/USD"), Decimal("0.1"), Decimal("100"), T0
    )
    assert req is not None
    state = broker.submit(req)
    assert state.status is OrderStatus.FILLED and state.average_fill_price == Decimal("100.5")
    sent = fake.orders[req.client_order_id]
    assert sent["time_in_force"] == "gtc" and sent["type"] == "limit"
    broker.submit(req)  # second submit finds the existing order; no new POST
    assert sum(1 for m, p in fake.calls if m == "POST") == 1
    account = broker.account()
    assert account.buying_power == Decimal("40.00")  # settled cash, never margin
    assert account.positions[0].instrument_id.symbol == "BTC/USD"
    with pytest.raises(ValueError):
        AlpacaPaperBroker("k", "s", base_url="https://api.alpaca.markets")


# ---- orders ----
def test_entry_orders_never_chase() -> None:
    iid = InstrumentId(venue="X", symbol="AAPL")
    assert entry_order(_plan("AAPL"), iid, Decimal("1"), Decimal("100.50"), T0) is None
    order = entry_order(_plan("AAPL"), iid, Decimal("1"), Decimal("99.876"), T0)
    assert order is not None and order.limit_price == Decimal("99.87")
    assert client_order_id("BTC/USD", OrderSide.SELL, T0) == "qos-BTCUSD-sell-202611231500"


# ---- alerts ----
def test_alert_sinks_and_isolation(tmp_path: Path) -> None:
    sent: list[tuple[str, bytes, Mapping[str, str]]] = []
    ntfy = NtfySink("quant-os-private-topic-123", post=lambda u, b, h: sent.append((u, b, h)))
    hook = WebhookSink("https://example.invalid/hook", post=lambda u, b, h: sent.append((u, b, h)))

    def broken(_u: str, _b: bytes, _h: Mapping[str, str]) -> None:
        raise ConnectionError("no network")

    store = StateStore(tmp_path / "a.db")
    sink = MultiSink([ntfy, WebhookSink("https://x.invalid", post=broken), hook, StoreSink(store)])
    sink.send("critical", "kill switch", "flattened all positions")
    assert sent[0][0].endswith("/quant-os-private-topic-123") and sent[0][2]["Priority"] == "5"
    assert json.loads(sent[1][1])["content"].startswith("[CRITICAL]")
    assert store.recent("events")[0]["kind"] == "alert"
    with pytest.raises(ValueError):
        NtfySink("short")
    quiet = MultiSink([ntfy], min_level="warning")
    quiet.send("info", "x", "y")
    assert len(sent) == 2


# ---- paper loop end to end ----
def _bars(prices: Sequence[str], start: datetime = T0) -> list[Bar]:
    return [
        Bar(start + timedelta(hours=i), Decimal(p) + 1, Decimal(p) - 1, Decimal(p))
        for i, p in enumerate(prices)
    ]


def _trader(tmp_path: Path, broker: PaperBroker, board: AgentBoard | None = None) -> PaperTrader:
    store = StateStore(tmp_path / "trading.db")
    instrument = Instrument(
        instrument_id=InstrumentId(venue="PAPER", symbol="BTC/USD"),
        asset_class=AssetClass.CRYPTO,
        instrument_type=InstrumentType.SPOT,
        base_currency="BTC",
        quote_currency="USD",
        price_precision=2,
        quantity_precision=6,
    )
    pipeline = EntryPipeline(
        shariah=ShariahGate(ShariahProfile.spot_crypto_and_day_trading()),
        protections=ProtectionManager([]),
        governor=PlanGovernor(dial_policy(RiskDial.BALANCED)),
        council=AnalystCouncil([RuleAnalyst("trend", Decimal("0.6"))]),
        risk=RiskEngine(halal_risk_limits()),
        cost_model=CostModel(),
    )
    fired = {"done": False}

    def signal(symbol: str, bars: Sequence[Bar]) -> TradePlan | None:
        if fired["done"]:
            return None
        fired["done"] = True
        return _plan(symbol)

    return PaperTrader(
        broker=broker,
        store=store,
        pipeline=pipeline,
        universe={"BTC/USD": UniverseItem(instrument, crypto_asset=CryptoAsset("BTC", "layer1"))},
        signal=signal,
        alerts=StoreSink(store),
        exit_policy=ExitPolicy(trail_fraction=None, max_holding=None),
        board=board,
    )


def test_paper_loop_enters_exits_and_survives_restart(tmp_path: Path) -> None:
    broker = PaperBroker(
        initial_cash=Decimal("1000"), fee_rate=Decimal("0"), slippage_bps=Decimal("0")
    )
    board = AgentBoard(tmp_path / "board.db")
    trader = _trader(tmp_path, broker, board)
    iid = trader.universe["BTC/USD"].instrument.instrument_id
    # An analyst called it before the entry (reports after entry are not graded).
    board.submit_report(
        symbol="BTC/USD",
        role=AnalystRole.NEWS_SENTIMENT,
        source="claude",
        stance=Stance.BULLISH,
        conviction=Decimal("0.7"),
        rationale="Flows supportive and no negative headlines this week.",
    )
    start = datetime.now(UTC) + timedelta(seconds=1)
    broker.update_mark(iid, Decimal("100"))
    first = trader.step(start, {"BTC/USD": _bars(["100"], start)})
    assert first.entries_submitted == ["BTC/USD"] and first.fills == ["BTC/USD"]
    assert len(board.open_tasks()) == 3  # bot asked the analysts to review the trade
    # "Crash": rebuild the trader from disk; the open position must still be there.
    trader2 = _trader(tmp_path, broker, board)
    assert trader2.store.open_positions()[0].position.plan.symbol == "BTC/USD"
    broker.update_mark(iid, Decimal("111"))
    second = trader2.step(start + timedelta(hours=1), {"BTC/USD": _bars(["100", "111"], start)})
    assert second.exits == ["BTC/USD"] and not trader2.store.open_positions()
    trade = trader2.store.closed_trades()[-1]
    assert trade.exit_reason is ExitReason.TAKE_PROFIT and trade.profit_ratio > 0
    assert board.leaderboard()[0]["source"] == "claude"  # the call was graded


def test_paper_loop_pauses_on_mismatch_and_kill_switch_flattens(tmp_path: Path) -> None:
    broker = PaperBroker(
        initial_cash=Decimal("1000"), fee_rate=Decimal("0"), slippage_bps=Decimal("0")
    )
    trader = _trader(tmp_path, broker)
    iid = trader.universe["BTC/USD"].instrument.instrument_id
    broker.update_mark(iid, Decimal("100"))
    trader.step(T0, {"BTC/USD": _bars(["100"])})
    stored = trader.store.open_positions()[0].position
    stored.quantity = stored.quantity * 2  # bot and broker now disagree
    trader.store.save_position(stored, "crypto")
    report = trader.step(T0 + timedelta(hours=1), {"BTC/USD": _bars(["100", "100"])})
    assert report.unknown_state and any("mismatch" in i for i in report.issues)
    stored.quantity = stored.quantity / 2
    trader.store.save_position(stored, "crypto")
    trader.store.set_kill_switch(True, "test")
    killed = trader.step(T0 + timedelta(hours=2), {"BTC/USD": _bars(["100", "100", "99"])})
    assert killed.killed and killed.exits == ["BTC/USD"] and not trader.store.open_positions()


def test_paper_loop_fails_closed_when_broker_is_down(tmp_path: Path) -> None:
    class DownBroker(PaperBroker):
        def account(self):  # type: ignore[no-untyped-def]
            raise ConnectionError("broker down")

    trader = _trader(tmp_path, DownBroker(initial_cash=Decimal("1000")))
    report = trader.step(T0, {"BTC/USD": _bars(["100"])})
    assert report.unknown_state and not report.entries_submitted


# ---- dashboard ----
def test_dashboard_state_and_kill_switch(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "d.db")
    store.record_equity(T0, Decimal("50"), Decimal("50"))
    store.record_equity(T0 + timedelta(days=1), Decimal("45"), Decimal("45"))
    store.record_decision(T0, "AAPL", False, "edge", Decimal("0"), [])
    snap = snapshot(store)
    assert snap["max_drawdown"] == pytest.approx(-0.1) and snap["blocked_by"] == {"edge": 1}
    token = "d" * 20
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, token))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "Kill switch" in page and "<svg" in page
        req = urllib.request.Request(
            base + "/api/kill",
            data=b'{"on": true}',
            method="POST",
            headers={"Authorization": "Bearer wrong"},
        )
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(req, timeout=5)
        req = urllib.request.Request(
            base + "/api/kill",
            data=b'{"on": true}',
            method="POST",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert json.loads(urllib.request.urlopen(req, timeout=5).read())["kill_switch"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---- champion / challenger, grading ----
def test_champion_challenger_gate(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "m.db")
    first = run_gate(store, "meta", "v1", None, [0.1] * 5, deflated_sharpe=0.97, pbo=0.3)
    assert first.promote and store.champion("meta")["version"] == "v1"  # type: ignore[index]
    worse = run_gate(store, "meta", "v2", [0.1] * 5, [0.09] * 5, deflated_sharpe=0.97, pbo=0.3)
    assert not worse.promote and store.champion("meta")["version"] == "v1"  # type: ignore[index]
    better = run_gate(
        store,
        "meta",
        "v3",
        [0.1] * 5,
        [0.12, 0.11, 0.13, 0.09, 0.12],
        deflated_sharpe=0.96,
        pbo=0.2,
    )
    assert better.promote and store.champion("meta")["version"] == "v3"  # type: ignore[index]
    overfit = compare([0.1] * 5, [0.2] * 5, deflated_sharpe=0.5, pbo=0.2)
    assert not overfit.promote and "deflated" in overfit.reason
    assert not compare([0.1] * 3, [0.2] * 3, deflated_sharpe=1, pbo=0).promote  # too few folds


def test_analyst_grading_weights_council_seats(tmp_path: Path) -> None:
    board = AgentBoard(tmp_path / "g.db")
    entry = datetime.now(UTC)
    for i in range(10):
        board.submit_report(
            symbol="AAPL",
            role=AnalystRole.TECHNICAL,
            source="gpt",
            stance=Stance.BULLISH,
            conviction=Decimal("0.8"),
            rationale=f"breakout number {i} with rising volume",
        )
    assert board.grade_trade("AAPL", entry + timedelta(seconds=1), Decimal("-0.03")) == 10
    assert board.grade_trade("AAPL", entry + timedelta(seconds=1), Decimal("-0.03")) == 0
    assert board.reliability("gpt", "technical") == 0.25  # always wrong -> cut to the floor
    seat = BoardAnalyst(board, AnalystRole.TECHNICAL)
    assert seat.review(_plan("AAPL")).conviction == Decimal("0.2000")
    assert board.reliability("nobody", "technical") == 0.75


# ---- PSX and the default signal ----
def test_psx_sessions_and_settlement() -> None:
    thursday = datetime(2026, 11, 26, 6, 0, tzinfo=UTC)  # 11:00 Karachi
    close = session_close(AssetClass.EQUITY, thursday, Market.PSX)
    assert close is not None and close.astimezone(UTC) == datetime(2026, 11, 26, 10, 30, tzinfo=UTC)
    friday = thursday + timedelta(days=1)
    fri_close = session_close(AssetClass.EQUITY, friday, Market.PSX)
    assert fri_close is not None and fri_close.hour == 16 and fri_close.minute == 30
    flat = flatten_time(AssetClass.EQUITY, thursday, market=Market.PSX)
    assert flat == close - timedelta(minutes=15)
    settle = settlement_time(AssetClass.EQUITY, friday, market=Market.PSX)
    assert settle.date().isoformat() == "2026-11-30"  # T+1 skips the weekend


def test_donchian_signal_builds_valid_plans() -> None:
    prices = [str(100 + i * 0.2) for i in range(60)] + ["115"]
    plan = DonchianBreakout()("AAPL", _bars(prices))
    assert plan is not None and plan.stop_loss < plan.entry_price < plan.profit_target
    assert DonchianBreakout()("AAPL", _bars([*prices[:-1], "100"])) is None


def test_runner_builds_from_example_config_and_reads_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_os_core.ops.run_paper import build_trader, load_csv_bars

    monkeypatch.setenv("APCA_API_KEY_ID", "paper-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "paper-secret")
    example = Path(__file__).resolve().parents[4] / "examples" / "paper.json"
    config = json.loads(example.read_text()) | {
        "state": str(tmp_path / "t.db"),
        "board": str(tmp_path / "b.db"),
        "day_trading": True,
    }
    trader, _ = build_trader(config)
    assert set(trader.universe) == {"AAPL", "BTC/USD"}
    assert trader.pipeline.governor.policy.risk_per_trade == Decimal("0.005")  # half of 1%
    assert trader.exit_policy.flatten_before_close
    csv_file = tmp_path / "MEBL.csv"
    csv_file.write_text("time,high,low,close\n2026-11-20T10:30:00+00:00,301,295,299.5\n")
    assert load_csv_bars(csv_file)[0].close == Decimal("299.5")
