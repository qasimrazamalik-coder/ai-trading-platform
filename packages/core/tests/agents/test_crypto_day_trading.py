import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from quant_os_core.agents import (
    AnalystCouncil,
    EntryPipeline,
    PlanGovernor,
    RuleAnalyst,
    TradePlan,
    day_trading_policy,
    flatten_time,
    session_close,
    settlement_time,
)
from quant_os_core.agents.exits import Bar, ExitManager, OpenPosition
from quant_os_core.agents.simulator import ExitPolicy, simulate
from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.market_data.free_sources import AlpacaBars, AlpacaCryptoBars
from quant_os_core.risk_engine import ExitReason, ProtectionManager, RiskEngine
from quant_os_core.shariah import (
    CryptoAsset,
    Madhhab,
    ShariahGate,
    ShariahOrderContext,
    ShariahProfile,
    halal_risk_limits,
)

OWNER = ShariahProfile.spot_crypto_and_day_trading()
STRICT = ShariahProfile(allow_pre_settlement_resale=False)
BTC = CryptoAsset("BTC", "layer1")


def _instrument(asset: AssetClass, kind: InstrumentType = InstrumentType.SPOT) -> Instrument:
    symbol = "BTC/USD" if asset is AssetClass.CRYPTO else "AAPL"
    return Instrument(
        instrument_id=InstrumentId(venue="ALPACA", symbol=symbol),
        asset_class=asset,
        instrument_type=kind,
        base_currency="BTC" if asset is AssetClass.CRYPTO else "AAPL",
        quote_currency="USD",
        price_precision=2,
        quantity_precision=6,
    )


def _ctx(**kw: Decimal | None) -> ShariahOrderContext:
    base: dict[str, Decimal | None] = {"owned_quantity": Decimal("0"), "free_cash": Decimal("1000")}
    base.update(kw)
    return ShariahOrderContext(**base)  # type: ignore[arg-type]


def _buy(gate: ShariahGate, asset: CryptoAsset | None) -> str:
    return gate.check_order(
        _instrument(AssetClass.CRYPTO),
        is_sell=False,
        quantity=Decimal("0.01"),
        notional=Decimal("500"),
        context=_ctx(),
        crypto_asset=asset,
    ).rule_id


def test_owner_profile_enables_spot_crypto_and_same_day_resale() -> None:
    assert OWNER.madhhab is Madhhab.PERMISSIVE_SPOT_CRYPTO
    assert OWNER.crypto_enabled and OWNER.allow_pre_settlement_resale
    gate = ShariahGate(OWNER)
    assert _buy(gate, BTC) == "OK"
    assert _buy(gate, None) == "R13"  # no token profile, fail closed
    assert _buy(ShariahGate(), BTC) == "R13"  # PK Hanafi default still blocks crypto


@pytest.mark.parametrize(
    ("asset", "rule"),
    [
        (CryptoAsset("AAVE", "lending"), "R13"),
        (CryptoAsset("DOGE", "meme"), "R13"),
        (CryptoAsset("XYZ", "Gambling"), "R13"),
        (CryptoAsset("USDY", "stablecoin", pays_interest=True), "R13"),
        (CryptoAsset("BTC3L", "layer1", is_leveraged_token=True), "R2"),
        (CryptoAsset("HYPE1", "layer1", has_utility=False), "R13"),
    ],
)
def test_token_screen_blocks_interest_gambling_derivatives(asset: CryptoAsset, rule: str) -> None:
    assert _buy(ShariahGate(OWNER), asset) == rule


def test_crypto_derivatives_stay_blocked_even_with_crypto_on() -> None:
    gate = ShariahGate(OWNER)
    for kind in (InstrumentType.PERPETUAL, InstrumentType.FUTURE, InstrumentType.OPTION):
        decision = gate.check_instrument(_instrument(AssetClass.CRYPTO, kind))
        assert decision.rule_id == "R2"


def test_strict_profile_blocks_equity_sale_before_settlement_not_crypto() -> None:
    gate = ShariahGate(STRICT)
    sell = {"is_sell": True, "quantity": Decimal("5"), "notional": Decimal("500")}
    unsettled = _ctx(owned_quantity=Decimal("5"), settled_quantity=Decimal("0"))
    settled = _ctx(owned_quantity=Decimal("5"), settled_quantity=Decimal("5"))
    equity = _instrument(AssetClass.EQUITY)
    assert gate.check_order(equity, context=unsettled, **sell).rule_id == "R9"
    assert gate.check_order(_instrument(AssetClass.EQUITY), context=settled, **sell).allowed
    unknown = _ctx(owned_quantity=Decimal("5"))
    assert gate.check_order(_instrument(AssetClass.EQUITY), context=unknown, **sell).rule_id == "R9"
    assert gate.check_order(_instrument(AssetClass.CRYPTO), context=unsettled, **sell).allowed
    assert (
        ShariahGate(OWNER)
        .check_order(_instrument(AssetClass.EQUITY), context=unsettled, **sell)
        .allowed
    )  # owner profile allows day trading


def test_session_flatten_and_settlement_times() -> None:
    friday = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)  # 10:00 New York
    close = session_close(AssetClass.EQUITY, friday)
    assert close is not None and close.astimezone(UTC) == datetime(2026, 11, 20, 21, tzinfo=UTC)
    assert flatten_time(AssetClass.EQUITY, friday) == close - timedelta(minutes=15)
    assert session_close(AssetClass.CRYPTO, friday) is None
    monday = settlement_time(AssetClass.EQUITY, friday)
    assert monday.astimezone(UTC).date() == date(2026, 11, 23)  # T+1 skips the weekend
    holiday = settlement_time(AssetClass.EQUITY, friday, holidays={date(2026, 11, 23)})
    assert holiday.astimezone(UTC).date() == date(2026, 11, 24)
    assert settlement_time(AssetClass.CRYPTO, friday) == friday
    with pytest.raises(ValueError):
        session_close(AssetClass.EQUITY, datetime(2026, 11, 20, 15))


def _plan(symbol: str, entry: str, stop: str, target: str) -> TradePlan:
    return TradePlan(
        symbol=symbol,
        entry_price=Decimal(entry),
        profit_target=Decimal(target),
        stop_loss=Decimal(stop),
        invalidation_condition="breaks below the opening range low",
        confidence=Decimal("0.8"),
        justification="test",
    )


def test_exit_manager_flattens_and_respects_settlement() -> None:
    mgr = ExitManager()
    t0 = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
    flat = flatten_time(AssetClass.EQUITY, t0)
    pos = OpenPosition(_plan("AAPL", "100", "99", "102"), Decimal("1"), t0, flatten_at=flat)
    late = t0 + timedelta(hours=5, minutes=50)
    bar = Bar(late, Decimal("100.5"), Decimal("99.5"), Decimal("100.2"))
    event = mgr.on_bar(pos, bar, ())
    assert event is not None and event.reason is ExitReason.TIME and event.price == Decimal("100.2")
    held = OpenPosition(
        _plan("AAPL", "100", "99", "102"),
        Decimal("1"),
        t0,
        earliest_exit=settlement_time(AssetClass.EQUITY, t0),
    )
    crash = Bar(t0 + timedelta(hours=1), Decimal("100"), Decimal("90"), Decimal("91"))
    assert mgr.on_bar(held, crash, ()) is None  # strict view: cannot sell yet


def _pipeline(profile: ShariahProfile) -> EntryPipeline:
    return EntryPipeline(
        shariah=ShariahGate(profile),
        protections=ProtectionManager(()),
        governor=PlanGovernor(day_trading_policy()),
        council=AnalystCouncil([RuleAnalyst("trend", Decimal("0.6"))]),
        risk=RiskEngine(halal_risk_limits()),
    )


def test_simulated_crypto_day_trade_round_trip() -> None:
    t0 = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)
    prices = ["100", "100.5", "101", "103", "102"]
    bars = [
        Bar(t0 + timedelta(hours=i), Decimal(p) + 1, Decimal(p) - Decimal("0.5"), Decimal(p))
        for i, p in enumerate(prices)
    ]

    def signal(history: Sequence[Bar]) -> TradePlan | None:
        if len(history) == 1:
            return _plan("BTC/USD", "100", "99", "102")
        return None

    result = simulate(
        bars,
        signal,
        _pipeline(OWNER),
        instrument=_instrument(AssetClass.CRYPTO),
        financials=None,
        crypto_asset=BTC,
        starting_cash=Decimal("50"),
        exit_policy=ExitPolicy(trail_fraction=None, max_holding=None, flatten_before_close=True),
    )
    assert [t.exit_reason for t in result.trades] == [ExitReason.TAKE_PROFIT]
    assert result.trades[0].exit_time - t0 < timedelta(days=1)
    blocked = simulate(
        bars,
        signal,
        _pipeline(OWNER),
        instrument=_instrument(AssetClass.CRYPTO),
        financials=None,
        crypto_asset=CryptoAsset("DOGE", "meme"),
        starting_cash=Decimal("50"),
    )
    assert blocked.trades == [] and blocked.blocked_by == {"shariah": 1}


def test_strict_profile_refuses_equity_day_trading_simulation() -> None:
    bar = Bar(datetime(2026, 11, 20, 15, tzinfo=UTC), Decimal("101"), Decimal("99"), Decimal("100"))
    with pytest.raises(ValueError):
        simulate(
            [bar],
            lambda _h: None,
            _pipeline(STRICT),
            instrument=_instrument(AssetClass.EQUITY),
            financials=None,
            starting_cash=Decimal("50"),
            exit_policy=ExitPolicy(flatten_before_close=True),
        )


class _Recorder:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages
        self.urls: list[str] = []
        self.headers: list[Mapping[str, str]] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> bytes:
        self.urls.append(url)
        self.headers.append(headers)
        return json.dumps(self.pages[len(self.urls) - 1]).encode()


def test_crypto_bars_adapter_pages_and_keys_by_symbol() -> None:
    raw = {"t": "2026-11-20T00:00:00Z", "o": 1, "h": 101, "l": 99, "c": 100, "v": 3}
    fetch = _Recorder(
        [
            {"bars": {"BTC/USD": [raw]}, "next_page_token": "abc"},
            {"bars": {"BTC/USD": [raw | {"t": "2026-11-20T01:00:00Z"}]}, "next_page_token": None},
        ]
    )
    bars = AlpacaCryptoBars(fetch=fetch).bars("btc/usd", "2026-11-20", "2026-11-21", "1Hour")
    assert len(bars) == 2 and bars[1].time == datetime(2026, 11, 20, 1, tzinfo=UTC)
    assert "symbols=BTC%2FUSD" in fetch.urls[0] and "timeframe=1Hour" in fetch.urls[0]
    assert "page_token=abc" in fetch.urls[1] and fetch.headers[0] == {}
    with pytest.raises(ValueError):
        AlpacaCryptoBars(fetch=fetch).bars("BTCUSD", "a", "b")
    with pytest.raises(ValueError):
        AlpacaCryptoBars(fetch=fetch).bars("BTC/USD", "a", "b", "2Min")


def test_stock_bars_adapter_accepts_intraday_timeframe() -> None:
    raw = {"t": "2026-11-20T14:35:00Z", "h": 101, "l": 99, "c": 100}
    fetch = _Recorder([{"bars": [raw], "next_page_token": None}])
    bars = AlpacaBars("k", "s", fetch=fetch).bars("aapl", "2026-11-20", "2026-11-20", "5Min")
    assert len(bars) == 1 and "AAPL/bars?timeframe=5Min" in fetch.urls[0]
