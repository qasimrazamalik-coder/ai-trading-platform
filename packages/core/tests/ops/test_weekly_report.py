import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from quant_os_core.ops.core_loop import TARGETS_KEY, WEIGHTS_KEY
from quant_os_core.ops.report import build_report, main, screening_warnings, send_report
from quant_os_core.ops.state import StateStore

NOW = datetime(2026, 11, 22, 12, tzinfo=UTC)


class FakeSink:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, level: str, title: str, message: str) -> None:
        self.sent.append((level, title, message))


def _config(source: str = "SEC N-PORT 2026-08-31, look-through") -> dict[str, object]:
    return {
        "strategy": "core",
        "universe": [
            {
                "symbol": "SPUS",
                "asset_class": "etf",
                "financials": {"sector": "x"},
                "financials_source": source,
            },
            {"symbol": "UMMA", "asset_class": "etf"},
            {"symbol": "BTC/USD", "asset_class": "crypto", "token_category": "layer1"},
        ],
    }


def _store(tmp_path: Path) -> StateStore:
    store = StateStore(tmp_path / "s.db")
    # 40 days: 10000 -> 10200 (day 10) -> peak 11000 (day 20) -> 9900 (day 25) -> 10450 now
    path = {0: "10000", 10: "10200", 20: "11000", 25: "9900", 33: "10000", 40: "10450"}
    level = Decimal("10000")
    for day in range(41):
        level = Decimal(path.get(day, str(level)))
        store.record_equity(NOW - timedelta(days=40 - day), level, Decimal("100"))
    stages = [{"stage": "cost_guard", "passed": True, "detail": ""}]
    store.record_decision(NOW - timedelta(days=2), "SPUS", True, None, Decimal("1"), stages)
    store.record_decision(NOW - timedelta(days=2), "UMMA", False, "shariah", Decimal(0), stages)
    store.record_decision(NOW - timedelta(days=3), "HLAL", False, "shariah", Decimal(0), stages)
    store.record_decision(NOW - timedelta(days=4), "SPRE", False, "risk", Decimal(0), stages)
    store.record_decision(NOW - timedelta(days=20), "SPUS", False, "risk", Decimal(0), stages)
    store.set_value(TARGETS_KEY, json.dumps({"SPUS": "0.5", "HLAL": "0.3"}))
    store.set_value(WEIGHTS_KEY, json.dumps({"SPUS": "0.45"}))
    return store


def test_weekly_report_summarises_equity_blocks_drift_and_stale_screens(tmp_path: Path) -> None:
    report = build_report(_store(tmp_path), _config(), now=NOW)
    assert report.equity == Decimal("10450")
    assert report.change_week == Decimal("10450") / Decimal("10000") - 1
    assert report.change_month == Decimal("10450") / Decimal("10200") - 1  # day 10
    assert report.worst_drawdown_all == Decimal("0.1")  # 11000 -> 9900
    assert report.approved == 1 and dict(report.blocked) == {"shariah": 2, "risk": 1}
    assert report.drift == {"SPUS": Decimal("-0.05"), "HLAL": Decimal("-0.3")}
    # SPUS screened 2026-08-31: 83 days old, fine; UMMA has no financials; crypto is skipped
    assert report.warnings == ["UMMA has no financials: the Shariah gate blocks it (R6)"]
    sink = FakeSink()
    send_report(report, sink)
    level, title, text = sink.sent[0]
    assert level == "warning" and title == "weekly paper report"
    assert "+4.50%" in text and "shariah 2" in text and "SPUS -5.0%" in text


def test_screening_data_older_than_120_days_is_flagged() -> None:
    warnings = screening_warnings(_config("SEC N-PORT 2026-05-31"), NOW.date())
    assert any("SPUS screening data is 175 days old" in w for w in warnings)
    undated = screening_warnings(_config("hand-entered"), NOW.date())
    assert any("no dated source" in w for w in undated)


def test_report_without_history_warns_and_needs_aware_time(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "empty.db")
    report = build_report(store, {"universe": []}, now=NOW)
    assert report.equity is None and "no equity snapshots" in report.warnings[0]
    assert "n/a" in report.text()
    with pytest.raises(ValueError):
        build_report(store, {}, now=datetime(2026, 1, 1))  # naive


def test_report_cli_dry_run_prints_without_sending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _store(tmp_path)
    config = tmp_path / "paper.json"
    config.write_text(json.dumps(_config() | {"state": str(tmp_path / "s.db")}), "utf-8")
    assert main(["--config", str(config), "--dry-run"]) == 0
    assert "Week to" in capsys.readouterr().out
