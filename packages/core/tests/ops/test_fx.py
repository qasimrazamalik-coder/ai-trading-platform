import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from quant_os_core.ops.fx import FxService, FxUnavailable, parse_payload

T0 = datetime(2026, 11, 22, 12, tzinfo=UTC)


def _payload(pkr: str = "280.5") -> dict[str, Any]:
    return {
        "result": "success",
        "base_code": "USD",
        "time_last_update_unix": int(T0.timestamp()),
        "rates": {"USD": 1, "PKR": Decimal(pkr), "EUR": Decimal("0.9"), "XYZ": 5},
    }


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


def test_converts_both_ways_and_between_currencies(tmp_path: Path) -> None:
    fx = FxService(lambda _url: _payload(), tmp_path / "fx.json", clock=Clock())
    assert fx.convert(Decimal("2"), "USD", "PKR") == Decimal("561.0")
    assert fx.convert(Decimal("561"), "pkr", "usd") == Decimal("2")
    assert fx.convert(Decimal("9"), "EUR", "USD") == Decimal("10")
    table = fx.table("PKR")
    assert table["rates"]["PKR"] == "1" and set(table["rates"]) == {"USD", "PKR", "EUR"}
    assert table["as_of"] == T0.isoformat() and table["stale"] is False
    with pytest.raises(ValueError):
        fx.convert(Decimal("1"), "USD", "XYZ")  # not a supported currency


def test_cache_ttl_then_refetch(tmp_path: Path) -> None:
    calls: list[str] = []
    clock = Clock()

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return _payload("280" if len(calls) == 1 else "290")

    fx = FxService(fetch, tmp_path / "fx.json", ttl=timedelta(hours=1), clock=clock)
    assert fx.rates().rates["PKR"] == Decimal("280")
    clock.now += timedelta(minutes=30)
    assert fx.rates().rates["PKR"] == Decimal("280") and len(calls) == 1
    clock.now += timedelta(hours=1)
    assert fx.rates().rates["PKR"] == Decimal("290") and len(calls) == 2


def test_falls_back_to_last_good_file_then_raises(tmp_path: Path) -> None:
    path = tmp_path / "state" / "fx_rates.json"
    FxService(lambda _url: _payload(), path, clock=Clock()).rates()
    assert json.loads(path.read_text("utf-8"))["rates"]["PKR"] == "280.5"

    def offline(_url: str) -> Any:
        raise OSError("no internet")

    fx = FxService(offline, path, clock=Clock())
    saved = fx.rates()
    assert saved.stale and saved.rates["PKR"] == Decimal("280.5")
    with pytest.raises(FxUnavailable, match="No currency rates"):
        FxService(offline, tmp_path / "missing.json", clock=Clock()).rates()


def test_bad_payloads_are_rejected(tmp_path: Path) -> None:
    for bad in (
        {"result": "error"},
        {**_payload(), "base_code": "EUR"},
        {**_payload(), "rates": {"USD": 1, "PKR": -3}},
        [],
    ):
        with pytest.raises(ValueError):
            parse_payload(bad, T0)
    fx = FxService(lambda _url: {"result": "error"}, tmp_path / "none.json", clock=Clock())
    with pytest.raises(FxUnavailable):
        fx.rates()
