import sys
import types
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_os_core.market_data import exchanges
from quant_os_core.market_data.exchanges import (
    SUPPORTED_EXCHANGES,
    SpotExchange,
    ccxt_client,
    is_spot_market,
    spread_bps,
)

SPOT = {
    "type": "spot",
    "spot": True,
    "base": "BTC",
    "quote": "USDT",
    "active": True,
    "linear": None,
    "inverse": None,
}
MARKETS = {
    "BTC/USDT": SPOT,
    "ETH/USDT": {**SPOT, "base": "ETH"},
    "BTC/USDT:USDT": {
        **SPOT,
        "type": "swap",
        "spot": False,
        "swap": True,
        "contract": True,
        "linear": True,
    },
    "BTC/USD:BTC-261225": {
        **SPOT,
        "type": "future",
        "spot": False,
        "future": True,
        "contract": True,
        "inverse": True,
    },
    "BTC/USD:BTC-261225-90000-C": {**SPOT, "type": "option", "spot": False, "option": True},
    "BTCUP/USDT": {**SPOT, "base": "BTCUP"},
    "ETHBULL/USDT": {**SPOT, "base": "ETHBULL"},
    "XRP3L/USDT": {**SPOT, "base": "XRP3L"},
    "OLD/USDT": {**SPOT, "base": "OLD", "active": False},
}
DAY_MS = 86_400_000


class FakeClient:
    def __init__(self, ohlcv=None, book=None):
        self.calls = []
        self._ohlcv = ohlcv or []
        self._book = book or {}

    def load_markets(self):
        self.calls.append("load_markets")
        return MARKETS

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append(("ohlcv", symbol, timeframe, since, limit))
        return self._ohlcv

    def fetch_order_book(self, symbol, limit):
        self.calls.append(("book", symbol, limit))
        return self._book


def test_spot_filter():
    assert is_spot_market(SPOT)
    for key in (
        "BTC/USDT:USDT",
        "BTC/USD:BTC-261225",
        "BTC/USD:BTC-261225-90000-C",
        "BTCUP/USDT",
        "ETHBULL/USDT",
        "XRP3L/USDT",
        "OLD/USDT",
    ):
        assert not is_spot_market(MARKETS[key]), key
    assert not is_spot_market({**SPOT, "linear": True})
    assert not is_spot_market({**SPOT, "contract": True})
    assert not is_spot_market({**SPOT, "base": ""})


def test_only_supported_exchanges():
    assert set(SUPPORTED_EXCHANGES) == {"binance", "kraken", "coinbase", "bybit", "okx"}
    assert SpotExchange(" OKX ", client=FakeClient()).exchange_id == "okx"
    with pytest.raises(ValueError):
        SpotExchange("deribit", client=FakeClient())


def test_markets_are_spot_only_and_cached():
    client = FakeClient()
    ex = SpotExchange("binance", client=client)
    assert set(ex.spot_markets()) == {"BTC/USDT", "ETH/USDT"}
    ex.spot_markets()
    assert client.calls.count("load_markets") == 1


def test_derivative_symbols_are_refused():
    ex = SpotExchange("bybit", client=FakeClient())
    for symbol in ("BTC/USDT:USDT", "BTCUP/USDT", "DOGE/USDT"):
        with pytest.raises(ValueError, match="not a spot market"):
            ex.ohlcv(symbol)
        with pytest.raises(ValueError, match="not a spot market"):
            ex.order_book(symbol)


def test_ohlcv_converts_to_decimal_candles_and_drops_forming_bar():
    now = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    start = int((now - timedelta(days=2)).timestamp() * 1000)
    rows = [
        [start + DAY_MS, 101.0, 103.0, 100.0, 102.5, 7.0],
        [start, 100.0, 102.0, 99.0, 101.0, None],
        [start + 2 * DAY_MS, 102.5, 104.0, 102.0, 103.0, 3.0],  # today's bar, still forming
    ]
    client = FakeClient(ohlcv=rows)
    ex = SpotExchange("kraken", client=client)
    since = datetime(2026, 1, 1, tzinfo=UTC)
    candles = ex.ohlcv("btc/usdt", "1d", since=since, limit=3)
    assert [c.close for c in candles] == [Decimal("101.0"), Decimal("102.5")]
    assert candles[0].volume == Decimal("0") and candles[1].volume == Decimal("7.0")
    assert candles[0].instrument_id.venue == "KRAKEN"
    assert candles[0].source == "ccxt:kraken"
    assert client.calls[-1] == ("ohlcv", "BTC/USDT", "1d", int(since.timestamp() * 1000), 3)


def test_ohlcv_guards():
    ex = SpotExchange("coinbase", client=FakeClient())
    with pytest.raises(ValueError):
        ex.ohlcv("BTC/USDT", "1M!")
    with pytest.raises(ValueError):
        ex.ohlcv("BTC/USDT", limit=0)
    with pytest.raises(ValueError):
        ex.ohlcv("BTC/USDT", since=datetime(2026, 1, 1))


def test_order_book_snapshot_and_spread():
    book = {
        "bids": [[100.0, 1.0], [99.5, 0.0], [99.0, 2.0]],
        "asks": [[100.2, 0.5], [100.5, 1.5]],
        "timestamp": 1_767_225_600_000,
        "nonce": 42,
    }
    client = FakeClient(book=book)
    snap = SpotExchange("okx", client=client).order_book("BTC/USDT", depth=5)
    assert [lvl.price for lvl in snap.bids] == [Decimal("100.0"), Decimal("99.0")]
    assert snap.sequence == 42 and snap.timestamp.tzinfo is not None
    assert spread_bps(snap) == pytest.approx(Decimal("19.98"), rel=Decimal("0.001"))
    assert client.calls[-1] == ("book", "BTC/USDT", 5)


def test_order_book_without_timestamp_or_nonce():
    book = {"bids": [[10.0, 1.0]], "asks": [[10.1, 1.0]], "timestamp": None, "nonce": None}
    snap = SpotExchange("binance", client=FakeClient(book=book)).order_book("ETH/USDT")
    assert snap.sequence is None
    with pytest.raises(ValueError):
        SpotExchange("binance", client=FakeClient(book=book)).order_book("ETH/USDT", depth=0)


def test_factory_used_lazily():
    made = []

    def factory(exchange_id):
        made.append(exchange_id)
        return FakeClient()

    ex = SpotExchange("binance", factory=factory)
    assert made == []
    ex.spot_markets()
    assert made == ["binance"]


def test_ccxt_missing_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "ccxt", None)
    with pytest.raises(RuntimeError, match="pip install ccxt"):
        ccxt_client("binance")


def test_ccxt_client_is_keyless_and_spot(monkeypatch):
    seen = {}

    class FakeExchange:
        def __init__(self, config):
            seen.update(config)

    fake = types.ModuleType("ccxt")
    fake.kraken = FakeExchange
    monkeypatch.setitem(sys.modules, "ccxt", fake)
    client = ccxt_client("kraken")
    assert isinstance(client, FakeExchange)
    assert seen == {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    assert "apiKey" not in seen and "secret" not in seen


def test_module_has_no_order_methods():
    names = {n.lower() for n in dir(SpotExchange)} | {n.lower() for n in dir(exchanges)}
    assert not any(
        word in n
        for n in names
        for word in ("order_create", "create_order", "cancel", "withdraw", "balance")
    )
