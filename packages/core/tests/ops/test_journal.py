from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from quant_os_core.ops.journal import Journal
from quant_os_core.ops.ledger import Ledger

T0 = datetime(2026, 4, 1, 14, tzinfo=UTC)
D = Decimal


def test_entries_link_fills_and_score_setups(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    ledger = Ledger(db)
    journal = Journal(db, ledger)
    b1 = ledger.record_fill(T0, "SPUS", "buy", D("10"), D("40"), D("1"))
    s1 = ledger.record_fill(T0 + timedelta(days=3), "SPUS", "sell", D("10"), D("44"), D("1"))
    b2 = ledger.record_fill(T0 + timedelta(days=4), "SPUS", "buy", D("10"), D("44"))
    s2 = ledger.record_fill(T0 + timedelta(days=5), "SPUS", "sell", D("10"), D("43"))
    b3 = ledger.record_fill(T0 + timedelta(days=6), "SPUS", "buy", D("1"), D("43"))

    win = journal.add(
        T0, "spus", "Breakout", "closed above range", "calm", True,
        tags=["Trend", "trend", " "], fill_ids=[b1, s1], screenshot="shots\\a.png",
    )  # fmt: skip
    journal.add(T0 + timedelta(days=4), "SPUS", "breakout", "chased", "greedy", False,
                fill_ids=[b2, s2], lesson="wait for the close")  # fmt: skip
    journal.add(T0 + timedelta(days=6), "SPUS", "pullback", "dip", "unsure", True, fill_ids=[b3])
    journal.add(T0 + timedelta(days=7), "SPUS", "pullback", "no fills", "calm", True)

    entries = journal.entries()
    assert [e.setup for e in entries] == ["pullback", "pullback", "breakout", "breakout"]
    first = next(e for e in entries if e.id == win)
    assert first.symbol == "SPUS" and first.tags == ("trend",)
    assert first.screenshot == "shots/a.png" and first.as_dict()["fill_ids"] == [b1, s1]
    assert journal.outcome(first) == D("38")  # 440 - 400 - 2 fees
    stats = journal.stats()
    assert stats["entries"] == 4 and stats["rule_adherence"] == 0.75
    assert stats["by_setup"]["breakout"] == {"entries": 2, "closed": 2, "wins": 1, "win_rate": 0.5}
    assert stats["by_setup"]["pullback"]["closed"] == 0
    assert stats["by_setup"]["pullback"]["win_rate"] is None
    assert stats["emotions"] == {"calm": 2, "greedy": 1, "unsure": 1}


def test_empty_stats_and_validation(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    assert journal.stats()["rule_adherence"] is None
    fill = journal.ledger.record_fill(T0, "HLAL", "buy", D("1"), D("50"))
    base = {"symbol": "SPUS", "setup": "x", "reason": "", "emotion": "calm", "rule_followed": True}
    bad_cases: list[tuple[dict[str, object], str]] = [
        ({"emotion": "angry"}, "emotion"),
        ({"setup": ""}, "setup"),
        ({"symbol": "x" * 61}, "symbol"),
        ({"rule_followed": "yes"}, "true or false"),
        ({"reason": "x" * 2001}, "reason"),
        ({"fill_ids": [999]}, "unknown"),
        ({"fill_ids": [fill]}, "same symbol"),
        ({"fill_ids": list(range(51))}, "at most"),
        ({"tags": [f"t{i}" for i in range(11)]}, "tags"),
        ({"screenshot": "../secret.png"}, "screenshot"),
        ({"screenshot": "http://x/y.png"}, "screenshot"),
        ({"screenshot": "a" * 261}, "screenshot"),
    ]
    for change, message in bad_cases:
        args = {**base, **change}
        with pytest.raises(ValueError, match=message):
            journal.add(
                T0,
                args.pop("symbol"),  # type: ignore[arg-type]
                args.pop("setup"),  # type: ignore[arg-type]
                args.pop("reason"),  # type: ignore[arg-type]
                args.pop("emotion"),  # type: ignore[arg-type]
                args.pop("rule_followed"),  # type: ignore[arg-type]
                **args,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="timezone"):
        journal.add(datetime(2026, 1, 1), "SPUS", "x", "", "calm", True)
