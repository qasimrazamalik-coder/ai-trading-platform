import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from quant_os_core.ops.readiness import check_readiness, main, mark_resolved
from quant_os_core.ops.state import StateStore

GUARD = [{"stage": "cost_guard", "passed": True, "detail": ""}]


def _healthy(tmp_path: Path, days: int = 35, dip: str = "9600") -> tuple[StateStore, datetime]:
    """Paper record that passes: 35 days, 4% dip, two monthly rebalances, quiet log."""

    store = StateStore(tmp_path / "paper.db")
    now = datetime.now(UTC)
    for day in range(days + 1):
        equity = Decimal(dip) if day == days // 2 else Decimal("10000") + day
        store.record_equity(now - timedelta(days=days - day, minutes=1), equity, Decimal("50"))
    # 32 days apart, so always in two different calendar months
    store.record_decision(now - timedelta(days=33), "SPUS", True, None, Decimal("1"), GUARD)
    store.record_decision(now - timedelta(days=1), "SPUS", True, None, Decimal("1"), GUARD)
    store.record_decision(now - timedelta(days=1), "UMMA", False, "shariah", Decimal(0), GUARD)
    return store, now


def _kv_rows(store: StateStore) -> list[tuple[str, str]]:
    with sqlite3.connect(store.path) as db:
        return db.execute("SELECT key, value FROM kv ORDER BY key").fetchall()


def test_clean_paper_record_passes_and_changes_nothing(tmp_path: Path) -> None:
    store, now = _healthy(tmp_path)
    before = _kv_rows(store)
    result = check_readiness(store, now=now)
    assert result.passed, result.reasons
    assert _kv_rows(store) == before  # advisory: no switch, flag or key is written
    text = result.text()
    assert text.startswith("Live readiness: PASS") and "unlocks nothing" in text


def test_short_history_drawdown_and_too_few_rebalances_fail(tmp_path: Path) -> None:
    store, now = _healthy(tmp_path, days=10, dip="8000")
    result = check_readiness(store, now=now, min_rebalances=3)
    failed = {name for name, ok, _ in result.checks if not ok}
    assert {"history", "drawdown", "rebalances"} <= failed and not result.passed
    assert any("10 days of paper equity" in r for r in result.reasons)
    # the 20% dip breaks the balanced limit (15%) but fits the growth dial (25%)
    assert check_readiness(store, now=now, dial="growth", min_days=5, min_rebalances=1).passed
    with pytest.raises(ValueError):
        check_readiness(store, dial="yolo")


def test_mismatch_stays_open_until_resolved_and_critical_events_block_a_week(
    tmp_path: Path,
) -> None:
    store, _ = _healthy(tmp_path)
    store.log("critical", "loop", "position mismatch: AAPL bot=1 broker=2")
    now = datetime.now(UTC)
    failed = {n for n, ok, _ in check_readiness(store, now=now).checks if not ok}
    assert {"mismatches", "critical events"} <= failed
    mark_resolved(store, "closed the stray AAPL share by hand")
    later = check_readiness(store, now=datetime.now(UTC))
    failed = {n for n, ok, _ in later.checks if not ok}
    assert "mismatches" not in failed and "critical events" in failed  # still inside 7 days
    # eight days on, the same critical event no longer blocks
    week_later = check_readiness(store, now=now + timedelta(days=8), min_days=30)
    assert "critical events" not in {n for n, ok, _ in week_later.checks if not ok}
    with pytest.raises(ValueError):
        mark_resolved(store, "  ")


def test_reconciliation_needs_kill_switch_off_and_no_stuck_core_order(tmp_path: Path) -> None:
    store, now = _healthy(tmp_path)
    store.set_value("core:pending", "qos-SPUS-buy-202611220300")
    result = check_readiness(store, now=now)
    assert any("core order still pending" in r for r in result.reasons)
    store.set_value("core:pending", "")
    store.set_kill_switch(True, "test")
    result = check_readiness(store, now=datetime.now(UTC))
    assert any("kill switch on" in r for r in result.reasons) and not result.passed


def test_stale_record_fails_fresh_check(tmp_path: Path) -> None:
    store, now = _healthy(tmp_path)
    result = check_readiness(store, now=now + timedelta(days=6))
    assert "fresh" in {n for n, ok, _ in result.checks if not ok}


def test_cli_prints_verdict_and_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store, _ = _healthy(tmp_path)
    config = tmp_path / "paper.json"
    config.write_text(
        json.dumps({"strategy": "core", "dial": "balanced", "state": str(store.path)}), "utf-8"
    )
    assert main(["--config", str(config)]) == 0
    assert "PASS" in capsys.readouterr().out
    store.log("critical", "core", "unexpected account state: short: ['SPUS'] core paused")
    assert main(["--config", str(config)]) == 1
    assert main(["--config", str(config), "--mark-resolved", "fixed"]) == 1  # critical < 7 days
    out = capsys.readouterr().out
    assert "FAIL" in out and "[ok] mismatches" in out
