"""Alpaca **paper** trading adapter implementing :class:`ExecutionBroker`.

Endpoints (Trading API v2, https://docs.alpaca.markets/reference/postorder):
``POST /v2/orders``, ``GET /v2/orders:by_client_order_id``, ``DELETE /v2/orders/{id}``,
``GET /v2/orders?status=open``, ``GET /v2/account``, ``GET /v2/positions``.

Only the paper host is accepted; live trading stays behind the platform's two-key interlock.
Stocks use ``day`` orders because Alpaca allows fractional quantities only for market and day
orders; crypto uses ``gtc``. A market order whose request carries ``notional`` is sent as a
dollar amount (Alpaca fractional "notional" orders) instead of a share quantity.

Buying power is the most conservative cash figure Alpaca reports: the smaller of ``cash`` and
``non_marginable_buying_power``. Alpaca's ``buying_power`` includes margin and is never used.
Paper accounts do not model T+1 settlement exactly, so the core loop also waits one step after
a sell before it buys with the proceeds.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import quote

from quant_os_core.backtesting.models import OrderStatus, OrderType
from quant_os_core.execution.models import (
    AccountSnapshot,
    BrokerOrderRequest,
    BrokerOrderState,
    BrokerPosition,
)
from quant_os_core.instruments import InstrumentId
from quant_os_core.ops.resilience import TransientError, retry

PAPER_URL = "https://paper-api.alpaca.markets"
Http = Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, bytes]]

_STATUS = {
    "new": OrderStatus.PENDING,
    "accepted": OrderStatus.PENDING,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.PENDING,
    "held": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.CANCELLED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "replaced": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.PENDING,
    "pending_replace": OrderStatus.PENDING,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "stopped": OrderStatus.PENDING,
    "calculated": OrderStatus.PENDING,
}


def urllib_http(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as err:
        return int(err.code), err.read()


def _dec(value: Any) -> Decimal:
    return Decimal(str(value)) if value not in (None, "") else Decimal("0")


def settled_cash(raw: Mapping[str, Any]) -> Decimal:
    """Cash that can fund a buy without margin: min of the cash figures Alpaca reports."""

    figures = [_dec(raw.get("cash"))]
    if raw.get("non_marginable_buying_power") not in (None, ""):
        figures.append(_dec(raw.get("non_marginable_buying_power")))
    return min(figures)


class AlpacaPaperBroker:
    venue = "ALPACA"
    supports_notional = True  # fractional market orders by dollar amount

    def __init__(
        self, key_id: str, secret_key: str, *, http: Http = urllib_http, base_url: str = PAPER_URL
    ) -> None:
        if base_url.rstrip("/") != PAPER_URL:
            raise ValueError("only the Alpaca paper host is allowed here")
        if not key_id or not secret_key:
            raise ValueError("Alpaca paper keys are required (APCA_API_KEY_ID/SECRET_KEY)")
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type": "application/json",
        }
        self._http = http
        self._base = PAPER_URL

    def _call(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = json.dumps(payload).encode() if payload is not None else None

        def once() -> Any:
            try:
                status, raw = self._http(method, self._base + path, self._headers, body)
            except (TimeoutError, OSError) as exc:
                raise TransientError(str(exc)) from exc
            if status == 429 or status >= 500:
                raise TransientError(f"alpaca {status}")
            if status == 404:
                return None
            if status >= 400:
                message = raw.decode(errors="replace")[:300]
                raise ValueError(f"alpaca rejected {method} {path}: {status} {message}")
            return json.loads(raw) if raw else {}

        # Order placement is not retried blindly: a timeout may have placed it. The loop
        # looks the order up by client_order_id before ever resubmitting.
        if method == "POST":
            return once()
        return retry(once)

    def _state(self, raw: Mapping[str, Any]) -> BrokerOrderState:
        avg = raw.get("filled_avg_price")
        return BrokerOrderState(
            client_order_id=str(raw["client_order_id"]),
            venue_order_id=str(raw["id"]),
            instrument_id=InstrumentId(venue=self.venue, symbol=str(raw["symbol"])),
            status=_STATUS.get(str(raw.get("status")), OrderStatus.PENDING),
            submitted_quantity=_dec(raw.get("qty")),
            filled_quantity=_dec(raw.get("filled_qty")),
            average_fill_price=_dec(avg) if avg else None,
            updated_at=datetime.now(UTC),
        )

    def submit(self, request: BrokerOrderRequest) -> BrokerOrderState:
        existing = self.order(request.client_order_id)
        if existing is not None:  # idempotent: never place the same order twice
            return existing
        symbol = request.instrument_id.symbol
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": request.side.value,
            "type": request.order_type.value,
            "time_in_force": "gtc" if "/" in symbol else "day",
            "client_order_id": request.client_order_id,
        }
        if request.notional is not None and request.order_type is OrderType.MARKET:
            payload["notional"] = str(request.notional)
        else:
            payload["qty"] = str(request.quantity)
        if request.order_type is OrderType.LIMIT:
            payload["limit_price"] = str(request.limit_price)
        if request.order_type is OrderType.STOP:
            payload["stop_price"] = str(request.stop_price)
        return self._state(self._call("POST", "/v2/orders", payload))

    def order(self, client_order_id: str) -> BrokerOrderState | None:
        raw = self._call(
            "GET", f"/v2/orders:by_client_order_id?client_order_id={quote(client_order_id)}"
        )
        return self._state(raw) if raw else None

    def cancel(self, client_order_id: str) -> BrokerOrderState:
        state = self.order(client_order_id)
        if state is None:
            raise KeyError(client_order_id)
        if state.status in {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}:
            self._call("DELETE", f"/v2/orders/{state.venue_order_id}")
            refreshed = self.order(client_order_id)
            return refreshed or state
        return state

    def open_orders(self) -> tuple[BrokerOrderState, ...]:
        rows = self._call("GET", "/v2/orders?status=open&limit=500") or []
        return tuple(self._state(r) for r in rows)

    def account(self) -> AccountSnapshot:
        raw = self._call("GET", "/v2/account") or {}
        positions = self._call("GET", "/v2/positions") or []
        return AccountSnapshot(
            cash=_dec(raw.get("cash")),
            equity=_dec(raw.get("equity")),
            # Halal mode never uses margin: buying power is settled cash, not Alpaca's figure.
            buying_power=settled_cash(raw),
            positions=tuple(
                BrokerPosition(
                    InstrumentId(venue=self.venue, symbol=_symbol(p)),
                    _dec(p.get("qty")),
                    _dec(p.get("avg_entry_price")),
                    _dec(p.get("current_price")),
                )
                for p in positions
            ),
            timestamp=datetime.now(UTC),
        )


def _symbol(position: Mapping[str, Any]) -> str:
    """Alpaca reports crypto positions as BTCUSD while orders use BTC/USD."""

    raw = str(position["symbol"])
    if position.get("asset_class") != "crypto" or "/" in raw:
        return raw
    for quote_ccy in ("USDT", "USDC", "USD", "BTC"):
        if raw.endswith(quote_ccy) and len(raw) > len(quote_ccy):
            return f"{raw[: -len(quote_ccy)]}/{quote_ccy}"
    return raw
