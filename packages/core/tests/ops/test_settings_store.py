import json
from decimal import Decimal
from pathlib import Path

import pytest

from quant_os_core.agents.trade_plan import PlanPolicy
from quant_os_core.ops import settings_store as ss
from quant_os_core.ops.settings_store import SettingsError, SettingsStore, settings_path_for
from quant_os_core.shariah import ShariahProfile, halal_risk_limits


def test_defaults_come_from_the_enforcing_code_and_hard_limits_are_locked() -> None:
    by = ss.BY_KEY
    assert by["risk_per_trade_pct"].default == PlanPolicy().risk_per_trade * 100
    assert by["daily_loss_limit_pct"].maximum == halal_risk_limits().max_daily_loss_fraction * 100
    assert by["max_noncompliant_revenue_pct"].default == (
        ShariahProfile().max_noncompliant_revenue_ratio * 100
    )
    locked = {s.key for s in ss.SETTINGS if s.locked}
    assert {
        "max_leverage",
        "allow_short",
        "allow_margin",
        "blocked_instruments",
        "paper_only",
        "live_trading",
        "max_interest_debt_pct",
        "max_interest_cash_pct",
        "max_noncompliant_revenue_pct",
        "min_tangible_assets_pct",
    } == locked
    assert by["max_leverage"].default == Decimal("1.0") and by["allow_short"].default is False
    assert "future" in by["blocked_instruments"].default
    assert all(s.lock_reason for s in ss.SETTINGS if s.locked)
    assert {s.group for s in ss.SETTINGS} <= set(ss.GROUPS)
    assert all(s.kind in ss.KINDS and s.description for s in ss.SETTINGS)
    # ranges never allow looser than the bot's own limits
    assert by["risk_per_trade_pct"].maximum == Decimal("2")
    assert by["max_position_pct"].maximum == Decimal("30")
    assert by["core_band_pct"].minimum == Decimal("2")
    assert by["min_edge_to_cost"].minimum == Decimal("1.5")


def test_paths() -> None:
    assert settings_path_for(Path("cfg/paper_core.json"), Path("state/a.db")) == Path(
        "cfg/paper_core.settings.json"
    )
    assert settings_path_for(None, Path("state/a.db")) == Path("state/settings.json")


def test_save_validated_values_and_describe(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "s.json")
    out = store.save(
        {
            "risk_per_trade_pct": "0.75",
            "max_trades_per_day": 2,
            "day_trading": True,
            "alert_topics": ["warnings", "fills", "fills"],
            "ntfy_topic": " my-private_topic ",
            "language": "ur",
            "core_band_pct": 5,
        }
    )
    assert out["risk_per_trade_pct"] == Decimal("0.75") and out["max_trades_per_day"] == 2
    assert out["alert_topics"] == ("fills", "warnings") and out["ntfy_topic"] == "my-private_topic"
    saved = json.loads((tmp_path / "s.json").read_text("utf-8"))
    assert "max_leverage" not in saved and saved["risk_per_trade_pct"] == "0.75"
    assert "dial" not in saved  # unchanged defaults are not written
    described = store.describe()
    row = next(s for s in described["settings"] if s["key"] == "max_position_pct")
    assert row["value"] == "20" and row["max"] == "30" and row["locked"] is False
    lock = next(s for s in described["settings"] if s["key"] == "allow_short")
    assert lock["locked"] and lock["value"] is False
    assert described["rejected"] == {} and described["path"] == "s.json"


@pytest.mark.parametrize(
    ("update", "key", "message"),
    [
        ({"max_leverage": 2}, "max_leverage", "locked"),
        ({"allow_short": True}, "allow_short", "locked"),
        ({"max_noncompliant_revenue_pct": 10}, "max_noncompliant_revenue_pct", "locked"),
        ({"live_trading": True}, "live_trading", "locked"),
        ({"risk_per_trade_pct": "5"}, "risk_per_trade_pct", "at most"),
        ({"daily_loss_limit_pct": "4"}, "daily_loss_limit_pct", "at most"),
        ({"core_band_pct": "1"}, "core_band_pct", "at least"),
        ({"min_edge_to_cost": "1"}, "min_edge_to_cost", "at least"),
        ({"max_trades_per_day": 9}, "max_trades_per_day", "at most"),
        ({"max_trades_per_day": "1.5"}, "max_trades_per_day", "whole"),
        ({"min_confidence": "abc"}, "min_confidence", "number"),
        ({"min_confidence": "NaN"}, "min_confidence", "number"),
        ({"min_confidence": True}, "min_confidence", "number"),
        ({"dial": "yolo"}, "dial", "one of"),
        ({"day_trading": "yes"}, "day_trading", "true or false"),
        ({"alert_topics": ["critical"]}, "alert_topics", "list"),
        ({"ntfy_topic": "bad topic!"}, "ntfy_topic", "not allowed"),
        ({"ntfy_topic": 5}, "ntfy_topic", "text"),
        ({"webhook_url": "http://plain"}, "webhook_url", "not allowed"),
        ({"nope": 1}, "nope", "unknown"),
        ({"risk_per_trade_pct": "1.5"}, "risk_per_trade_pct", "balanced dial"),
        ({"dial": "conservative"}, "risk_per_trade_pct", "conservative dial"),
    ],
)
def test_invalid_or_loosening_writes_are_refused(
    tmp_path: Path, update: dict[str, object], key: str, message: str
) -> None:
    store = SettingsStore(tmp_path / "s.json")
    with pytest.raises(SettingsError) as info:
        store.save(update)
    assert message in info.value.errors[key]
    assert not (tmp_path / "s.json").exists()


def test_growth_dial_allows_its_own_caps(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "s.json")
    out = store.save({"dial": "growth", "risk_per_trade_pct": "2", "max_position_pct": "30"})
    assert out["risk_per_trade_pct"] == Decimal("2") and out["dial"] == "growth"
    out = store.save({"dial": "conservative", "risk_per_trade_pct": "0.5", "max_position_pct": 15})
    assert out["dial"] == "conservative"
    with pytest.raises(SettingsError):
        store.save({})
    with pytest.raises(SettingsError):
        store.save(["not", "a", "dict"])


def test_hand_edited_file_cannot_loosen_limits(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "max_leverage": "3",
                "allow_short": True,
                "risk_per_trade_pct": "9",
                "min_confidence": "0.7",
                "unknown": 1,
            }
        ),
        "utf-8",
    )
    store = SettingsStore(path)
    values = store.values()
    assert values["max_leverage"] == Decimal("1.0") and values["allow_short"] is False
    assert values["risk_per_trade_pct"] == Decimal("1") and values["min_confidence"] == Decimal(
        "0.7"
    )
    assert set(store.rejected) == {"max_leverage", "allow_short", "risk_per_trade_pct"}

    path.write_text(json.dumps({"dial": "conservative", "max_position_pct": "18"}), "utf-8")
    values = store.values()  # over the conservative dial: back to the safe defaults
    assert values["max_position_pct"] == Decimal("20") and values["dial"] == "balanced"
    assert "max_position_pct" in store.rejected

    path.write_text("{broken", "utf-8")
    assert store.values()["dial"] == "balanced" and "_file" in store.rejected
    path.write_text("[1, 2]", "utf-8")
    assert store.values()["dial"] == "balanced"


def test_failed_write_leaves_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SettingsStore(tmp_path / "s.json")

    def boom(*_args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ss.os, "replace", boom)
    with pytest.raises(OSError):
        store.save({"language": "ar"})
    assert list(tmp_path.iterdir()) == []
