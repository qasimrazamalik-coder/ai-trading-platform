"""Settings panel model: every tunable setting with its type, range, default and description,
plus the hard limits shown as locked.

Values are written, after validation, to a JSON file next to the paper config
(``paper_core.json`` -> ``paper_core.settings.json``). The ranges only ever allow a setting to be
as strict as the code's default or stricter where a hard limit applies (GUARDRAILS 8-12 and the
Shariah thresholds), so a write here cannot loosen a hard limit. Locked settings are displayed
but every attempt to change them is refused, and a hand-edited file that tries is ignored.

Defaults are read from the code that enforces them (``PlanPolicy``, ``CostModel``,
``halal_risk_limits``, ``ShariahProfile``...) so the panel cannot drift from the bot.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from quant_os_core.agents.profit_engine import CostModel, RiskDial, dial_policy
from quant_os_core.agents.trade_plan import PlanPolicy
from quant_os_core.ops.core_loop import CoreRebalancer
from quant_os_core.ops.signals import DonchianBreakout
from quant_os_core.shariah import ShariahProfile, halal_risk_limits
from quant_os_core.shariah.rules import BLOCKED_ASSET_CLASSES, BLOCKED_INSTRUMENT_TYPES
from quant_os_core.strategies.diversified_core import CoreMode

HUNDRED = Decimal("100")
KINDS = ("number", "int", "choice", "multi", "text", "bool", "list")
GROUPS = ("risk", "core", "signals", "alerts", "display", "screening", "hard_limits")
MAX_TEXT = 300


@dataclass(frozen=True, slots=True)
class Setting:
    key: str
    group: str
    kind: str
    default: Any
    description: str
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    choices: tuple[str, ...] = ()
    unit: str = ""
    pattern: str | None = None
    locked: bool = False
    lock_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "group": self.group,
            "kind": self.kind,
            "default": _plain(self.default),
            "description": self.description,
            "min": None if self.minimum is None else str(self.minimum),
            "max": None if self.maximum is None else str(self.maximum),
            "choices": list(self.choices),
            "unit": self.unit,
            "locked": self.locked,
            "lock_reason": self.lock_reason,
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple | list):
        return list(value)
    return value


def _clean(value: Decimal) -> Decimal:
    """``Decimal("20.00")`` -> ``20``, never ``2E+1``."""

    value = value.normalize()
    return value.quantize(Decimal(1)) if value == value.to_integral_value() else value


def _pct(fraction: Decimal) -> Decimal:
    return _clean(fraction * HUNDRED)


_POLICY = PlanPolicy()
_LIMITS = halal_risk_limits()
_PROFILE = ShariahProfile()
_COST = CostModel()
_CORE_DEFAULTS = {f.name: f.default for f in fields(CoreRebalancer)}
_CORE_BAND = Decimal(str(_CORE_DEFAULTS["band"]))
_CORE_MIN = Decimal(str(_CORE_DEFAULTS["min_notional"]))
_SIGNAL = DonchianBreakout()
_GROWTH = dial_policy(RiskDial.GROWTH)
_BLOCKED = tuple(
    sorted(
        {str(t.value) for t in BLOCKED_INSTRUMENT_TYPES}
        | {str(a.value) for a in BLOCKED_ASSET_CLASSES}
    )
)
_LOCK_SHARIAH = (
    "Shariah threshold (AAOIFI Standard 21); changing it needs a code change and a scholar's "
    "sign-off"
)


def _locked(
    key: str, kind: str, default: Any, description: str, reason: str, unit: str = ""
) -> Setting:
    return Setting(
        key, "hard_limits", kind, default, description, unit=unit, locked=True, lock_reason=reason
    )


SETTINGS: tuple[Setting, ...] = (
    # ---- risk ------------------------------------------------------------------------------
    Setting(
        "dial",
        "risk",
        "choice",
        RiskDial.BALANCED.value,
        "Risk dial: conservative (0.5% per trade, 15% per position), balanced (1%, 20%) or "
        "the opt-in growth dial (2%, 30%). Caps the two settings below.",
        choices=tuple(d.value for d in RiskDial),
    ),
    Setting(
        "risk_per_trade_pct",
        "risk",
        "number",
        _pct(_POLICY.risk_per_trade),
        "Equity lost if a trade's stop is hit. Never above the dial's limit (2% at most).",
        Decimal("0.1"),
        _pct(_GROWTH.risk_per_trade),
        unit="%",
    ),
    Setting(
        "max_position_pct",
        "risk",
        "number",
        _pct(_POLICY.max_position_fraction),
        "Largest single position as a share of equity. Never above the dial's limit; the "
        "risk engine also caps positions at 20%.",
        Decimal("1"),
        _pct(_GROWTH.max_position_fraction),
        unit="%",
    ),
    Setting(
        "daily_loss_limit_pct",
        "risk",
        "number",
        _pct(_LIMITS.max_daily_loss_fraction),
        "A loss this large in one day halts new entries. Can only be tightened.",
        Decimal("0.5"),
        _pct(_LIMITS.max_daily_loss_fraction),
        unit="%",
    ),
    Setting(
        "max_drawdown_pct",
        "risk",
        "number",
        _pct(_LIMITS.max_drawdown_fraction),
        "A fall this far from the equity high halts new entries. Can only be tightened.",
        Decimal("2"),
        _pct(_LIMITS.max_drawdown_fraction),
        unit="%",
    ),
    Setting(
        "max_trades_per_day",
        "risk",
        "int",
        _POLICY.max_trades_per_day,
        "New trades allowed per day. Can only be lowered.",
        Decimal("1"),
        Decimal(_POLICY.max_trades_per_day),
    ),
    Setting(
        "day_trading",
        "risk",
        "bool",
        False,
        "Intraday mode: trades are closed before the market closes and risk half the dial's "
        "per-trade budget.",
    ),
    # ---- diversified core ------------------------------------------------------------------
    Setting(
        "core_mode",
        "core",
        "choice",
        CoreMode.DEFENSIVE.value,
        "Target mix of the diversified core: defensive or growth.",
        choices=tuple(m.value for m in CoreMode),
    ),
    Setting(
        "core_gate",
        "core",
        "choice",
        "cost_guard",
        "cost_guard rebalances at most monthly; trade_plan sends each fund through the full "
        "trade-plan pipeline with a stop and target.",
        choices=("cost_guard", "trade_plan"),
    ),
    Setting(
        "core_band_pct",
        "core",
        "number",
        _pct(_CORE_BAND),
        "Rebalance only when a fund is more than this far off its target weight. Cannot go "
        "below the owner-approved 2 points.",
        _pct(_CORE_BAND),
        Decimal("20"),
        unit="%",
    ),
    Setting(
        "core_min_notional",
        "core",
        "number",
        _CORE_MIN,
        "Smallest order the core will send, in USD. Cannot go below $1.",
        _CORE_MIN,
        Decimal("100000"),
        unit="USD",
    ),
    Setting(
        "invest_new_cash_immediately",
        "core",
        "bool",
        False,
        "Invest a deposit at once (buys only) instead of waiting for the monthly rebalance.",
    ),
    # ---- signals ---------------------------------------------------------------------------
    Setting(
        "signal",
        "signals",
        "choice",
        "donchian",
        "Entry signal for the trade-plan loop: Donchian breakout or the volatility-forecast "
        "breakout.",
        choices=("donchian", "forecast"),
    ),
    Setting(
        "min_confidence",
        "signals",
        "number",
        _POLICY.min_confidence,
        "Lowest signal confidence accepted. Can only be raised.",
        _POLICY.min_confidence,
        Decimal("0.95"),
    ),
    Setting(
        "min_reward_to_risk",
        "signals",
        "number",
        _POLICY.min_reward_to_risk,
        "Target distance divided by stop distance must be at least this. Can only be raised.",
        _POLICY.min_reward_to_risk,
        Decimal("10"),
    ),
    Setting(
        "min_edge_to_cost",
        "signals",
        "number",
        _COST.min_edge_to_cost,
        "Expected edge must beat round-trip cost by this multiple. Can only be raised.",
        _COST.min_edge_to_cost,
        Decimal("10"),
    ),
    Setting(
        "max_stop_distance_pct",
        "signals",
        "number",
        _pct(_POLICY.max_stop_distance),
        "Widest stop accepted, as a share of the entry price. Can only be tightened.",
        Decimal("2"),
        _pct(_POLICY.max_stop_distance),
        unit="%",
    ),
    Setting(
        "breakout_lookback",
        "signals",
        "int",
        _SIGNAL.lookback,
        "Bars in the breakout channel.",
        Decimal("10"),
        Decimal("120"),
        unit="bars",
    ),
    Setting(
        "trend_window",
        "signals",
        "int",
        _SIGNAL.trend_window,
        "Bars in the trend filter's moving average.",
        Decimal("20"),
        Decimal("250"),
        unit="bars",
    ),
    # ---- alerts ----------------------------------------------------------------------------
    Setting(
        "ntfy_topic",
        "alerts",
        "text",
        "",
        "Private ntfy topic for phone alerts (letters, digits, - and _; long and hard to guess).",
        pattern=r"^[A-Za-z0-9_-]{0,64}$",
    ),
    Setting(
        "webhook_url",
        "alerts",
        "text",
        "",
        "Optional https webhook for alerts.",
        pattern=r"^(https://[^\s]{4,290})?$",
    ),
    Setting(
        "alert_topics",
        "alerts",
        "multi",
        ("fills", "exits", "daily_summary", "warnings"),
        "Which alerts to send. Critical alerts (kill switch, crashes, unknown state) are "
        "always sent.",
        choices=("fills", "exits", "daily_summary", "warnings", "deposits"),
    ),
    # ---- display ---------------------------------------------------------------------------
    Setting(
        "language", "display", "choice", "en", "Dashboard language.", choices=("en", "ur", "ar")
    ),
    Setting(
        "currency",
        "display",
        "choice",
        "PKR",
        "Currency the dashboard shows money in.",
        choices=("USD", "PKR", "SAR", "AED", "GBP", "EUR"),
    ),
    # ---- screening -------------------------------------------------------------------------
    Setting(
        "screening_max_age_days",
        "screening",
        "int",
        180,
        "Warn when a fund's financials are older than this (fund_screen --max-age-days).",
        Decimal("30"),
        Decimal("365"),
        unit="days",
    ),
    # ---- hard limits (locked) --------------------------------------------------------------
    _locked(
        "max_leverage",
        "number",
        _LIMITS.max_gross_leverage,
        "Gross and net leverage.",
        "GUARDRAILS rule 8: leverage is fixed at 1.0",
        unit="x",
    ),
    _locked(
        "allow_short",
        "bool",
        _LIMITS.allow_short,
        "Short selling.",
        "GUARDRAILS rule 3: short selling is not permissible",
    ),
    _locked(
        "allow_margin",
        "bool",
        False,
        "Margin accounts and borrowing.",
        "GUARDRAILS rule 2: margin involves interest (riba)",
    ),
    _locked(
        "blocked_instruments",
        "list",
        _BLOCKED,
        "Instrument types and asset classes the Shariah gate always blocks.",
        "GUARDRAILS rule 1: spot ownership only",
    ),
    _locked(
        "paper_only",
        "bool",
        True,
        "Orders go to the Alpaca paper account only.",
        "GUARDRAILS rule 17: paper trading only",
    ),
    _locked(
        "live_trading",
        "bool",
        False,
        "Real-money trading.",
        "GUARDRAILS rule 17: needs the two-key production interlock",
    ),
    _locked(
        "max_interest_debt_pct",
        "number",
        _pct(_PROFILE.max_interest_debt_to_market_cap),
        "Interest-bearing debt / market value must stay below this.",
        _LOCK_SHARIAH,
        unit="%",
    ),
    _locked(
        "max_interest_cash_pct",
        "number",
        _pct(_PROFILE.max_interest_cash_to_market_cap),
        "Interest-bearing cash and securities / market value must stay below this.",
        _LOCK_SHARIAH,
        unit="%",
    ),
    _locked(
        "max_noncompliant_revenue_pct",
        "number",
        _pct(_PROFILE.max_noncompliant_revenue_ratio),
        "Non-compliant revenue / total revenue must stay below this.",
        _LOCK_SHARIAH,
        unit="%",
    ),
    _locked(
        "min_tangible_assets_pct",
        "number",
        _pct(_PROFILE.min_tangible_asset_ratio),
        "Tangible assets / total assets must be at least this.",
        _LOCK_SHARIAH,
        unit="%",
    ),
)

BY_KEY: dict[str, Setting] = {s.key: s for s in SETTINGS}


class SettingsError(ValueError):
    def __init__(self, errors: Mapping[str, str]) -> None:
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = dict(errors)


def settings_path_for(config: Path | None, state: Path) -> Path:
    """``paper_core.json`` -> ``paper_core.settings.json``; without a config, next to the state."""

    if config is not None:
        return config.with_name(config.stem + ".settings.json")
    return state.parent / "settings.json"


def _coerce(spec: Setting, raw: Any) -> Any:
    """Validated value for ``spec`` or ``ValueError`` with a message for the owner."""

    if spec.kind == "bool":
        if not isinstance(raw, bool):
            raise ValueError("must be true or false")
        return raw
    if spec.kind in ("number", "int"):
        if isinstance(raw, bool) or not isinstance(raw, int | float | str) or len(str(raw)) > 30:
            raise ValueError("must be a number")
        try:
            value = Decimal(str(raw).strip())
        except InvalidOperation:
            raise ValueError("must be a number") from None
        if not value.is_finite():
            raise ValueError("must be a number")
        if spec.kind == "int" and value != value.to_integral_value():
            raise ValueError("must be a whole number")
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"must be at least {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"must be at most {spec.maximum}")
        return int(value) if spec.kind == "int" else _clean(value)
    if spec.kind == "choice":
        if raw not in spec.choices:
            raise ValueError(f"must be one of {', '.join(spec.choices)}")
        return raw
    if spec.kind == "multi":
        if not isinstance(raw, list | tuple) or any(v not in spec.choices for v in raw):
            raise ValueError(f"must be a list drawn from {', '.join(spec.choices)}")
        return tuple(sorted(set(raw), key=spec.choices.index))
    if spec.kind == "text":
        if not isinstance(raw, str) or len(raw) > MAX_TEXT:
            raise ValueError(f"must be text up to {MAX_TEXT} characters")
        text = raw.strip()
        if spec.pattern and not re.fullmatch(spec.pattern, text):
            raise ValueError("has characters or a form that is not allowed")
        return text
    raise ValueError("cannot be set")  # "list" settings are locked


def _cross_check(values: Mapping[str, Any]) -> dict[str, str]:
    """Rules that tie settings together: the dial caps risk per trade and position size."""

    errors: dict[str, str] = {}
    policy = dial_policy(RiskDial(values["dial"]))
    risk_cap, position_cap = _pct(policy.risk_per_trade), _pct(policy.max_position_fraction)
    if Decimal(values["risk_per_trade_pct"]) > risk_cap:
        errors["risk_per_trade_pct"] = f"the {values['dial']} dial allows at most {risk_cap}%"
    if Decimal(values["max_position_pct"]) > position_cap:
        errors["max_position_pct"] = f"the {values['dial']} dial allows at most {position_cap}%"
    return errors


@dataclass
class SettingsStore:
    path: Path
    rejected: dict[str, str] = field(default_factory=dict)

    def defaults(self) -> dict[str, Any]:
        return {s.key: s.default for s in SETTINGS}

    def _saved(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            self.rejected = {"_file": "settings file is unreadable; using defaults"}
            return {}
        return raw if isinstance(raw, dict) else {}

    def values(self) -> dict[str, Any]:
        """Defaults overlaid with valid saved values. Locked or invalid saved values are
        ignored and listed in ``rejected``; so is a saved set that breaks a cross-check."""

        self.rejected = {}
        merged = self.defaults()
        for key, raw in self._saved().items():
            spec = BY_KEY.get(key)
            if spec is None:
                continue
            if spec.locked:
                self.rejected[key] = "locked"
                continue
            try:
                merged[key] = _coerce(spec, raw)
            except ValueError as exc:
                self.rejected[key] = str(exc)
        for key in _cross_check(merged):
            self.rejected[key] = "over the dial's limit"
            merged[key] = BY_KEY[key].default
        if _cross_check(merged):  # the default itself is over a tighter dial: fall back fully
            for key in ("dial", "risk_per_trade_pct", "max_position_pct"):
                merged[key] = BY_KEY[key].default
        return merged

    def validate(self, updates: Any) -> dict[str, Any]:
        """Clean values for ``updates`` merged over the current ones, or ``SettingsError``.

        Locked and unknown keys are refused; every value must be in range; the dial must
        allow the risk settings.
        """

        if not isinstance(updates, Mapping) or not updates:
            raise SettingsError({"_": "send an object of settings to change"})
        errors: dict[str, str] = {}
        merged = self.values()
        for key, raw in updates.items():
            spec = BY_KEY.get(str(key))
            if spec is None:
                errors[str(key)] = "unknown setting"
            elif spec.locked:
                errors[spec.key] = f"locked: {spec.lock_reason}"
            else:
                try:
                    merged[spec.key] = _coerce(spec, raw)
                except ValueError as exc:
                    errors[spec.key] = str(exc)
        if not errors:
            errors.update(_cross_check(merged))
        if errors:
            raise SettingsError(errors)
        return merged

    def save(self, updates: Any) -> dict[str, Any]:
        merged = self.validate(updates)
        unlocked = {
            k: _plain(v)
            for k, v in merged.items()
            if not BY_KEY[k].locked and v != BY_KEY[k].default
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".settings-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(unlocked, handle, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return self.values()

    def describe(self) -> dict[str, Any]:
        """What ``GET /api/settings`` returns."""

        values = self.values()
        return {
            "groups": list(GROUPS),
            "settings": [{**s.as_dict(), "value": _plain(values[s.key])} for s in SETTINGS],
            "rejected": dict(self.rejected),
            "path": self.path.name,
        }
