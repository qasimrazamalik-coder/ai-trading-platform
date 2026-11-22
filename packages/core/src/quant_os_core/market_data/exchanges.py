"""Read-only crypto SPOT market data from several exchanges through ccxt.

Supported: Binance, Kraken, Coinbase, Bybit (spot) and OKX (spot). Only two calls are made:
OHLCV candles and order books. No API keys are accepted, so this module cannot place, change
or cancel an order or read an account.

Spot only (GUARDRAILS R1/R2): every market whose ccxt record is a swap, future, option or
any other contract is dropped, and leveraged tokens (``BTCUP``, ``ETHBULL``, ``XRP3L`` ...)
are dropped too. Asking for one raises ``ValueError``. Passing this filter is not a Shariah
approval: tokens still go through the token screen (R13) before any paper order.

ccxt is optional and imported lazily; install it with ``pip install ccxt``. Tests inject a
client, so they never touch the network.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import BookLevel, Candle, OrderBookSnapshot
from quant_os_core.market_data.timeframes import Timeframe

SUPPORTED_EXCHANGES: tuple[str, ...] = ("binance", "kraken", "coinbase", "bybit", "okx")
_DERIVATIVE_FLAGS = ("swap", "future", "option", "contract")
_LEVERAGED_TOKEN = re.compile(r"(UP|DOWN|BULL|BEAR|\d+[LS])$")


class ExchangeClient(Protocol):
    """The read-only subset of a ccxt exchange used here."""

    def load_markets(self) -> Mapping[str, Mapping[str, Any]]: ...

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int | None, limit: int | None
    ) -> Sequence[Sequence[Any]]: ...

    def fetch_order_book(self, symbol: str, limit: int | None) -> Mapping[str, Any]: ...


ClientFactory = Callable[[str], ExchangeClient]


def ccxt_client(exchange_id: str) -> ExchangeClient:
    """Public (keyless) ccxt client defaulting to spot markets. Needs ``pip install ccxt``."""

    try:
        ccxt = importlib.import_module("ccxt")
    except ImportError as exc:
        raise RuntimeError(
            "ccxt is not installed; run `pip install ccxt` to read exchange market data"
        ) from exc
    cls = getattr(ccxt, exchange_id)
    client: ExchangeClient = cls({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    return client


def is_spot_market(market: Mapping[str, Any]) -> bool:
    """True for a plain spot pair that is not a derivative or a leveraged token."""

    if market.get("type", "spot") != "spot" or market.get("spot") is False:
        return False
    if any(market.get(flag) for flag in _DERIVATIVE_FLAGS):
        return False
    if market.get("linear") is not None or market.get("inverse") is not None:
        return False  # ccxt sets these only on contracts
    if market.get("active") is False:
        return False
    base = str(market.get("base", "")).upper()
    return bool(base) and not _LEVERAGED_TOKEN.search(base)


def _dec(value: Any) -> Decimal:
    return Decimal(str(value))


@dataclass(slots=True)
class SpotExchange:
    """Read-only spot data from one exchange. ``client`` defaults to a keyless ccxt client."""

    exchange_id: str
    client: ExchangeClient | None = None
    factory: ClientFactory = ccxt_client
    _markets: dict[str, Mapping[str, Any]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.exchange_id = self.exchange_id.strip().lower()
        if self.exchange_id not in SUPPORTED_EXCHANGES:
            raise ValueError(f"exchange must be one of {SUPPORTED_EXCHANGES}")

    def _client(self) -> ExchangeClient:
        if self.client is None:
            self.client = self.factory(self.exchange_id)
        return self.client

    def spot_markets(self) -> dict[str, Mapping[str, Any]]:
        """Spot markets only, keyed by ccxt symbol (``BTC/USDT``). Cached after first load."""

        if self._markets is None:
            raw = self._client().load_markets()
            self._markets = {s: m for s, m in raw.items() if is_spot_market(m)}
        return self._markets

    def _require_spot(self, symbol: str) -> str:
        pair = symbol.strip().upper()
        if pair not in self.spot_markets():
            raise ValueError(
                f"{pair} is not a spot market on {self.exchange_id}; derivatives and "
                "leveraged tokens are blocked (GUARDRAILS R1/R2)"
            )
        return pair

    def _instrument(self, pair: str) -> InstrumentId:
        return InstrumentId(venue=self.exchange_id, symbol=pair)

    def ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """Closed candles, oldest first. ``since`` must be timezone-aware."""

        pair = self._require_spot(symbol)
        frame = Timeframe.parse(timeframe)
        if not 1 <= limit <= 5000:
            raise ValueError("limit must be between 1 and 5000")
        since_ms = None
        if since is not None:
            if since.tzinfo is None:
                raise ValueError("since must be timezone-aware")
            since_ms = int(since.timestamp() * 1000)
        rows = self._client().fetch_ohlcv(pair, timeframe, since_ms, limit)
        now = datetime.now(UTC)
        candles = []
        for ts, o, h, low, c, v in (tuple(row[:6]) for row in rows):
            start = datetime.fromtimestamp(int(ts) / 1000, tz=UTC)
            end = start + frame.delta
            if end > now:
                continue  # still forming
            candles.append(
                Candle(
                    instrument_id=self._instrument(pair),
                    timestamp=start,
                    close_time=end,
                    source=f"ccxt:{self.exchange_id}",
                    timeframe=timeframe,
                    open=_dec(o),
                    high=_dec(h),
                    low=_dec(low),
                    close=_dec(c),
                    volume=_dec(v or 0),
                )
            )
        return sorted(candles, key=lambda candle: candle.timestamp)

    def order_book(self, symbol: str, depth: int = 20) -> OrderBookSnapshot:
        pair = self._require_spot(symbol)
        if not 1 <= depth <= 500:
            raise ValueError("depth must be between 1 and 500")
        raw = self._client().fetch_order_book(pair, depth)

        def levels(side: str) -> tuple[BookLevel, ...]:
            return tuple(
                BookLevel(price=_dec(level[0]), quantity=_dec(level[1]))
                for level in (raw.get(side) or [])[:depth]
                if _dec(level[1]) > 0
            )

        stamp = raw.get("timestamp")
        when = (
            datetime.fromtimestamp(int(stamp) / 1000, tz=UTC)
            if stamp is not None
            else datetime.now(UTC)
        )
        nonce = raw.get("nonce")
        return OrderBookSnapshot(
            instrument_id=self._instrument(pair),
            timestamp=when,
            source=f"ccxt:{self.exchange_id}",
            sequence=int(nonce) if isinstance(nonce, int) and nonce >= 0 else None,
            bids=levels("bids"),
            asks=levels("asks"),
        )


def spread_bps(book: OrderBookSnapshot) -> Decimal:
    """Best ask minus best bid, in basis points of the midpoint."""

    bid, ask = book.best_bid.price, book.best_ask.price
    return (ask - bid) / ((ask + bid) / 2) * Decimal(10_000)
