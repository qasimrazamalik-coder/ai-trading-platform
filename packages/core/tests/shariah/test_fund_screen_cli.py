"""The fund look-through command line, run offline with SEC and Yahoo access faked."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from quant_os_core.shariah import fund_screen, yahoo_fundamentals
from quant_os_core.shariah.fund_screen import ScreenResult
from quant_os_core.shariah.rules import CompanyFinancials


def _financials(debt: str) -> CompanyFinancials:
    return CompanyFinancials(
        sector="technology",
        market_cap=Decimal("1000"),
        interest_bearing_debt=Decimal(debt),
        interest_bearing_cash_and_securities=Decimal("50"),
        total_revenue=Decimal("500"),
        noncompliant_revenue=Decimal("5"),
        total_assets=Decimal("800"),
        tangible_assets=Decimal("600"),
        shares_outstanding=Decimal("10"),
    )


RESULTS = {
    "HLAL": ScreenResult(
        "HLAL", "2026-09-30", _financials("100"), 0.97, 120, 0.0, ["SAP SE"]
    ),
    "DEBT": ScreenResult("DEBT", "2026-09-30", _financials("600"), 0.95, 80, 0.02),
    "BOND": ScreenResult("BOND", reason="not an equity fund"),
}


@pytest.fixture
def faked(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {"clients": []}

    class _Client:
        def __init__(self, user_agent: str, cache_dir: Path) -> None:
            self.fallback: Any = None
            seen["clients"].append((user_agent, cache_dir, self))

    class _Yahoo:
        def __init__(self) -> None:
            self.used = ["SAP SE"]

        def resolve(self, holding: object) -> None:
            return None

    def fake_screen(client: Any, ticker: str, *, min_coverage: float) -> ScreenResult:
        seen.setdefault("coverage", set()).add(min_coverage)
        return RESULTS[ticker]

    monkeypatch.setattr(fund_screen, "SecClient", _Client)
    monkeypatch.setattr(fund_screen, "screen_fund", fake_screen)
    monkeypatch.setattr(yahoo_fundamentals, "YahooFundamentals", _Yahoo)
    return seen


def test_cli_prints_verdicts_and_yahoo_usage(
    faked: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["hlal", "debt", "bond", "--user-agent", "Tester t@example.invalid"]
    assert fund_screen.main([*args, "--cache", str(tmp_path), "--min-coverage", "0.8"]) == 0
    out = capsys.readouterr().out
    assert "HLAL (2026-09-30, 120 holdings, 97% resolved)" in out and "-> PASS" in out
    assert "unresolved (1): SAP SE" in out
    assert "DEBT (" in out and "-> FAIL" in out  # 60% debt/market cap breaches the screen
    assert "BOND: NOT SCREENED - not an equity fund" in out
    assert "Yahoo data used for 1 holding(s)" in out
    ((agent, cache, client),) = faked["clients"]
    assert agent == "Tester t@example.invalid" and cache == tmp_path
    assert client.fallback is not None  # Yahoo resolver wired in unless --no-yahoo
    assert faked["coverage"] == {0.8}


def test_cli_write_updates_config_and_needs_config_path(
    faked: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "paper_core.json"
    config.write_text(json.dumps({"universe": [{"symbol": "HLAL"}, {"symbol": "SPUS"}]}))
    base = ["hlal", "--user-agent", "T t@example.invalid", "--cache", str(tmp_path), "--no-yahoo"]
    assert fund_screen.main([*base, "--write", "--config", str(config)]) == 0
    assert "wrote financials for 1 fund(s)" in capsys.readouterr().out
    universe = json.loads(config.read_text())["universe"]
    assert universe[0]["financials"]["sector"] == "technology"
    assert "financials" not in universe[1]
    assert faked["clients"][0][2].fallback is None  # --no-yahoo keeps SEC data only

    with pytest.raises(SystemExit) as exc:
        fund_screen.main([*base, "--write"])
    assert exc.value.code == 2
    assert "--write needs --config" in capsys.readouterr().err
