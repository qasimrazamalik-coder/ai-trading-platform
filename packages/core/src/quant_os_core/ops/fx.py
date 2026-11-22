"""Currency rates for the dashboard: USD, PKR and other currencies the owner may think in.

Source: the free, keyless ExchangeRate-API endpoint ``https://open.er-api.com/v6/latest/USD``
(updated about once a day). The fetch is an injected callable so tests never touch the network.

Order of preference when rates are asked for:

1. the in-memory copy while it is younger than ``ttl``;
2. a fresh fetch, which is also written to ``state/fx_rates.json`` as the last good copy;
3. if the fetch fails: the older in-memory copy, then the last good file (marked ``stale``);
4. nothing at all: ``FxUnavailable`` with a message the page can show.

Display only. Orders are priced and settled in the broker's currency; nothing here sizes trades.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

DEFAULT_URL = "https://open.er-api.com/v6/latest/USD"
SOURCE = "open.er-api.com (ExchangeRate-API)"
CURRENCIES = (
    "USD",
    "PKR",
    "EUR",
    "GBP",
    "AED",
    "SAR",
    "MYR",
    "TRY",
    "INR",
    "CNY",
    "JPY",
    "AUD",
    "CAD",
    "CHF",
    "BTC",  # optional: kept only when the source returns it
)

Fetch = Callable[[str], Any]
Clock = Callable[[], datetime]


class FxUnavailable(RuntimeError):
    """No rates from the source, memory or the last-good file."""


def http_fetch(url: str) -> Any:
    """GET ``url`` and parse JSON with exact decimals (10 second timeout)."""

    request = urllib.request.Request(url, headers={"User-Agent": "quant-os-dashboard"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read(), parse_float=Decimal)


@dataclass(frozen=True, slots=True)
class FxRates:
    """Units of each currency per 1 USD."""

    rates: dict[str, Decimal]
    as_of: datetime  # when the source last updated the rates
    fetched_at: datetime  # when this process (or an earlier one) downloaded them
    stale: bool = False

    def rate(self, base: str, quote: str) -> Decimal:
        """Units of ``quote`` per 1 ``base``: usd[quote] / usd[base]."""

        base, quote = base.upper(), quote.upper()
        for code in (base, quote):
            if code not in self.rates:
                raise ValueError(f"unsupported currency: {code}")
        return self.rates[quote] / self.rates[base]

    def convert(self, amount: Decimal, base: str, quote: str) -> Decimal:
        """amount x usd[quote] / usd[base], multiplying first to keep exact results exact."""

        self.rate(base, quote)  # validates both codes
        return amount * self.rates[quote.upper()] / self.rates[base.upper()]


def _to_decimal(raw: Any) -> Decimal | None:
    if isinstance(raw, bool):
        return None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() and value > 0 else None


def parse_payload(payload: Any, fetched_at: datetime) -> FxRates:
    """Validate an open.er-api.com response and keep the supported currencies."""

    if not isinstance(payload, dict) or payload.get("result") != "success":
        raise ValueError("rate source did not return success")
    if payload.get("base_code") != "USD":
        raise ValueError("rate source must be USD-based")
    raw = payload.get("rates")
    if not isinstance(raw, dict):
        raise ValueError("rate source returned no rates")
    rates = {}
    for code in CURRENCIES:
        value = _to_decimal(raw.get(code))
        if value is not None:
            rates[code] = value
    rates["USD"] = Decimal("1")
    if "PKR" not in rates:
        raise ValueError("rate source has no PKR rate")
    updated = payload.get("time_last_update_unix")
    try:
        as_of = datetime.fromtimestamp(int(str(updated)), tz=UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        as_of = fetched_at
    return FxRates(rates, as_of, fetched_at)


class FxService:
    """Cached, fail-soft USD-based rate table."""

    def __init__(
        self,
        fetch: Fetch = http_fetch,
        cache_path: Path = Path("state/fx_rates.json"),
        *,
        ttl: timedelta = timedelta(hours=6),
        retry_after: timedelta = timedelta(minutes=5),
        clock: Clock = lambda: datetime.now(UTC),
        url: str = DEFAULT_URL,
    ) -> None:
        self.fetch, self.cache_path, self.ttl, self.clock, self.url = (
            fetch,
            cache_path,
            ttl,
            clock,
            url,
        )
        self.retry_after = retry_after
        self._rates: FxRates | None = None
        self._last_failure: datetime | None = None
        self._lock = threading.Lock()

    def rates(self) -> FxRates:
        with self._lock:
            now = self.clock()
            if self._rates and not self._rates.stale and now - self._rates.fetched_at < self.ttl:
                return self._rates
            recently_failed = (
                self._last_failure is not None and now - self._last_failure < self.retry_after
            )
            if not recently_failed:
                try:
                    fresh = parse_payload(self.fetch(self.url), now)
                except Exception:  # offline, timeout, bad JSON: fall back below
                    self._last_failure = now
                else:
                    self._rates, self._last_failure = fresh, None
                    self._save(fresh)
                    return fresh
            if self._rates is not None:
                self._rates = FxRates(
                    self._rates.rates, self._rates.as_of, self._rates.fetched_at, stale=True
                )
                return self._rates
            saved = self._load()
            if saved is not None:
                self._rates = saved
                return saved
            raise FxUnavailable(
                "No currency rates: the rate source is unreachable and no saved rates exist "
                f"at {self.cache_path}. Connect to the internet once to download them."
            )

    def convert(self, amount: Decimal, from_code: str, to_code: str) -> Decimal:
        """``amount`` in ``from_code`` expressed in ``to_code`` (not rounded)."""

        if not amount.is_finite():
            raise ValueError("amount must be a number")
        return self.rates().convert(amount, from_code, to_code)

    def table(self, base: str = "USD") -> dict[str, Any]:
        """JSON-ready rates relative to ``base`` for the dashboard."""

        fx = self.rates()
        base = base.upper()
        return {
            "base": base,
            "rates": {code: str(fx.rate(base, code)) for code in fx.rates},
            "as_of": fx.as_of.isoformat(),
            "fetched_at": fx.fetched_at.isoformat(),
            "stale": fx.stale,
            "source": SOURCE,
        }

    def _save(self, fx: FxRates) -> None:
        body = {
            "rates": {k: str(v) for k, v in fx.rates.items()},
            "as_of": fx.as_of.isoformat(),
            "fetched_at": fx.fetched_at.isoformat(),
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(body, indent=1), "utf-8")
            os.replace(tmp, self.cache_path)
        except OSError:  # a read-only disk must not break the dashboard
            pass

    def _load(self) -> FxRates | None:
        try:
            body = json.loads(self.cache_path.read_text("utf-8"))
            rates = {k: _to_decimal(v) for k, v in body["rates"].items() if k in CURRENCIES}
            clean = {k: v for k, v in rates.items() if v is not None}
            as_of = datetime.fromisoformat(body["as_of"])
            fetched = datetime.fromisoformat(body["fetched_at"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
        if "USD" not in clean or "PKR" not in clean or as_of.tzinfo is None:
            return None
        if fetched.tzinfo is None:
            return None
        return FxRates(clean, as_of, fetched, stale=True)
