import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant_os_core.ops import manual_backtest as mb
from quant_os_core.ops.manual_backtest import ManualBacktester, load_bars


def _csv(folder: Path, symbol: str, closes: list[float], with_open: bool = False) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["time,open,high,low,close" if with_open else "time,high,low,close"]
    for i, c in enumerate(closes):
        day = (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)).isoformat()
        lines.append(
            f"{day},{c},{c + 1},{c - 1},{c}" if with_open else f"{day},{c + 1},{c - 1},{c}"
        )
    (folder / f"{symbol}.csv").write_text("\n".join(lines))


def test_load_bars_fills_open_and_rejects_bad_input(tmp_path: Path) -> None:
    _csv(tmp_path, "AAA", [100.0] * 40)
    _csv(tmp_path, "OPN", [50.0] * 40, with_open=True)
    _csv(tmp_path, "TINY", [1.0] * 5)
    bars = load_bars(tmp_path, "AAA")
    assert len(bars) == 40 and bars[0].open == bars[0].close
    assert load_bars(tmp_path, "OPN")[3].open == 50
    for bad in ("../x", "NOPE", "TINY"):
        with pytest.raises(ValueError):
            load_bars(tmp_path, bad)


def test_step_buy_stop_target_and_stats(tmp_path: Path) -> None:
    closes = [100.0] * 31 + [100, 104, 110, 100, 100, 95, 90, 100, 100]
    _csv(tmp_path, "AAA", closes)
    bt = ManualBacktester(tmp_path)
    view = bt.start({"symbol": "AAA", "start": 31, "cash": "10000", "cost_bps": "10"})
    sid = view["session"]
    assert view["revealed"] == 32 and view["price"] == "100" and view["position"] is None
    assert all(b["time"] <= "2024-02-01" for b in view["bars"])  # no future bars leak

    with pytest.raises(ValueError, match="long only"):
        bt.act(sid, "buy", {"stop": "101", "target": "110"})
    view = bt.act(sid, "buy", {"stop": "95", "target": "108", "fraction": "0.5"})
    pos = view["position"]
    assert pos["stop"] == "95" and pos["target"] == "108"
    assert float(view["cash"]) < 5001  # half the cash plus the 10 bps cost
    with pytest.raises(ValueError, match="pyramiding"):
        bt.act(sid, "buy", {"stop": "95", "target": "108"})

    view = bt.act(sid, "next", {})  # 104: high 105, no exit
    assert view["position"] is not None
    view = bt.act(sid, "next", {})  # 110: high 111 >= 108, target
    assert view["position"] is None
    trade = view["trades"][-1]
    assert trade["reason"] == "target" and trade["exit"] == "109"  # gapped over the target
    assert float(trade["r"]) > 1

    bt.act(sid, "next", {"steps": 1})  # 100
    bt.act(sid, "buy", {"stop": "96", "target": "120"})
    view = bt.act(sid, "next", {})  # 100: low 99, nothing
    view = bt.act(sid, "next", {})  # 95: low 94 <= 96, stop
    assert view["trades"][-1]["reason"] == "stop" and view["trades"][-1]["exit"] == "96"
    stats = view["stats"]
    assert stats["trades"] == 2 and stats["wins"] == 1 and stats["win_rate"] == "0.5000"
    assert float(stats["costs"]) > 0 and float(stats["max_drawdown_pct"]) <= 0

    bt.act(sid, "buy", {"stop": "80", "target": "130"})
    view = bt.act(sid, "sell", {})
    assert view["trades"][-1]["reason"] == "manual"
    with pytest.raises(ValueError, match="nothing to sell"):
        bt.act(sid, "sell", {})
    view = bt.act(sid, "next", {"steps": "50"})
    assert view["finished"] and view["remaining"] == 0
    with pytest.raises(ValueError, match="no more bars"):
        bt.act(sid, "buy", {"stop": "1", "target": "1000"})
    assert bt.act(sid, "view", {})["finished"]


def test_gap_below_stop_fills_at_the_open(tmp_path: Path) -> None:
    _csv(tmp_path, "GAP", [100.0] * 31 + [100, 80, 80])
    bt = ManualBacktester(tmp_path)
    sid = bt.start({"symbol": "GAP", "start": 31, "cost_bps": "0"})["session"]
    bt.act(sid, "buy", {"stop": "95", "target": "120"})
    view = bt.act(sid, "next", {})
    assert view["trades"][-1]["exit"] == "81"  # gapped below the stop: filled at the open


def test_session_errors_limits_and_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _csv(tmp_path, "AAA", [100.0] * 60)
    bt = ManualBacktester(tmp_path)
    random_start = bt.start({"symbol": "AAA"})
    assert mb.MIN_HISTORY + 1 <= random_start["revealed"] <= 59
    for bad in ({"cash": "5"}, {"cost_bps": "900"}, {"start": "3"}):
        with pytest.raises(ValueError):
            bt.start({"symbol": "AAA", **bad})
    sid = random_start["session"]
    with pytest.raises(KeyError):
        bt.act("missing", "view", {})
    with pytest.raises(ValueError, match="action"):
        bt.act(sid, "short", {})
    with pytest.raises(ValueError, match="steps"):
        bt.act(sid, "next", {"steps": "99"})
    with pytest.raises(ValueError, match="fraction"):
        bt.act(sid, "buy", {"stop": "90", "target": "110", "fraction": "2"})

    monkeypatch.setattr(mb, "MAX_SESSIONS", 2)
    a = bt.start({"symbol": "AAA"})["session"]
    b = bt.start({"symbol": "AAA"})["session"]
    assert sid not in bt._sessions and {a, b} <= set(bt._sessions)
    bt._sessions[a].touched -= mb.SESSION_TTL * 2
    bt.start({"symbol": "AAA"})
    assert a not in bt._sessions


def test_tiny_cash_cannot_buy_a_unit(tmp_path: Path) -> None:
    _csv(tmp_path, "BIG", [10_000_000.0] * 40)
    bt = ManualBacktester(tmp_path)
    sid = bt.start({"symbol": "BIG", "start": 31, "cash": "100"})["session"]
    with pytest.raises(ValueError, match="not enough"):
        bt.act(sid, "buy", {"stop": "9000000", "target": "11000000", "fraction": "0.0001"})


def test_never_imports_the_order_path() -> None:
    tree = ast.parse(Path(mb.__file__).read_text("utf-8"))
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    banned = ("alpaca", "orders", "paper_loop", "core_loop", "execution", "state", "pipeline")
    assert not [m for m in modules if any(b in m for b in banned)], modules
