"""Money calculators for the dashboard: pure ``Decimal`` functions, no I/O.

Conventions used by every function here:

* Percentages are passed as percent numbers: ``1`` means 1%, ``2.5`` means 2.5%.
* Money results are rounded half-up to 2 decimal places, ratios to 4, and only at the end.
* Nonsense input (negative money, a stop above the entry, a 100% drop...) raises ``ValueError``.
* Long only: the position sizer refuses a stop at or above the entry, because short selling is
  blocked (GUARDRAILS rule 3).

``run(name, payload)`` is the JSON entry point the dashboard's ``POST /api/calc/<name>`` uses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

CENT = Decimal("0.01")
RATIO = Decimal("0.0001")
HUNDRED = Decimal("100")
ZAKAT_RATE = Decimal("0.025")
NISAB_GOLD_GRAMS = Decimal("87.48")  # 7.5 tola of gold, the common hanafi/majority figure
NISAB_SILVER_GRAMS = Decimal("612.36")  # 52.5 tola of silver
MAX_YEARS = 100


def _money(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def _ratio(x: Decimal) -> Decimal:
    return x.quantize(RATIO, rounding=ROUND_HALF_UP)


def _positive(name: str, x: Decimal) -> None:
    if not x.is_finite() or x <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _non_negative(name: str, x: Decimal) -> None:
    if not x.is_finite() or x < 0:
        raise ValueError(f"{name} cannot be negative")


def _percent(name: str, x: Decimal, low: Decimal, high: Decimal) -> None:
    if not x.is_finite() or not low <= x <= high:
        raise ValueError(f"{name} must be between {low} and {high} percent")


def _years(years: int) -> None:
    if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= MAX_YEARS:
        raise ValueError(f"years must be a whole number from 1 to {MAX_YEARS}")


def position_size(
    equity: Decimal, risk_pct: Decimal, entry: Decimal, stop: Decimal, *, whole_shares: bool = True
) -> dict[str, Decimal]:
    """Shares to buy so that hitting the stop loses ``risk_pct`` of equity.

    risk per share = entry - stop
    budget         = equity x risk_pct / 100
    shares         = floor(budget / risk per share)   (whole shares unless ``whole_shares=False``)
    money at risk  = shares x risk per share
    position value = shares x entry

    The sizer does not cap the position; the risk engine still applies its own limits
    (1% per trade and 20% per position by default).
    """

    _positive("equity", equity)
    _percent("risk_pct", risk_pct, Decimal("0.01"), HUNDRED)
    _positive("entry", entry)
    _positive("stop", stop)
    if stop >= entry:
        raise ValueError("stop must be below the entry (long only; short selling is blocked)")
    per_share = entry - stop
    budget = equity * risk_pct / HUNDRED
    raw = budget / per_share
    shares = raw.quantize(Decimal("1") if whole_shares else RATIO, rounding=ROUND_DOWN)
    return {
        "shares": shares,
        "risk_per_share": _money(per_share),
        "risk_budget": _money(budget),
        "money_at_risk": _money(shares * per_share),
        "position_value": _money(shares * entry),
        "position_pct_of_equity": _ratio(shares * entry / equity * HUNDRED),
    }


def reward_risk(entry: Decimal, stop: Decimal, target: Decimal) -> dict[str, Decimal]:
    """Reward-to-risk ratio of a long plan and the win rate needed to break even.

    R                   = (target - entry) / (entry - stop)
    break-even win rate = 1 / (1 + R)        (as a percent; ignores costs)
    """

    _positive("entry", entry)
    _positive("stop", stop)
    _positive("target", target)
    if stop >= entry:
        raise ValueError("stop must be below the entry")
    if target <= entry:
        raise ValueError("target must be above the entry")
    r = (target - entry) / (entry - stop)
    return {
        "reward_per_share": _money(target - entry),
        "risk_per_share": _money(entry - stop),
        "reward_risk": _ratio(r),
        "breakeven_win_rate_pct": _ratio(HUNDRED / (1 + r)),
    }


def compound_growth(
    start: Decimal, monthly_deposit: Decimal, annual_return_pct: Decimal, years: int
) -> dict[str, Any]:
    """Balance after ``years`` of monthly deposits growing at a fixed yearly rate.

    monthly rate i = annual_return_pct / 100 / 12        (nominal rate, compounded monthly)
    each month:    balance = balance x (1 + i) + monthly_deposit   (deposit at month end)

    Returns the final balance, total deposited, total growth and a year-by-year table.
    An illustration, not a forecast: real returns are uneven and can be negative.
    """

    _non_negative("start", start)
    _non_negative("monthly_deposit", monthly_deposit)
    _percent("annual_return_pct", annual_return_pct, Decimal("-99"), HUNDRED)
    _years(years)
    i = annual_return_pct / HUNDRED / 12
    balance, deposited = start, start
    table = []
    for year in range(1, years + 1):
        for _ in range(12):
            balance = balance * (1 + i) + monthly_deposit
            deposited += monthly_deposit
        table.append(
            {
                "year": year,
                "deposited": _money(deposited),
                "balance": _money(balance),
                "growth": _money(balance - deposited),
            }
        )
    return {
        "final_balance": _money(balance),
        "total_deposited": _money(deposited),
        "total_growth": _money(balance - deposited),
        "table": table,
    }


def deposit_goal(
    target: Decimal, years: int, annual_return_pct: Decimal, start: Decimal = Decimal("0")
) -> dict[str, Decimal]:
    """Monthly deposit needed to reach ``target`` in ``years`` (inverse of compound_growth).

    i = annual_return_pct / 100 / 12, n = 12 x years, g = (1 + i)^n
    monthly deposit = (target - start x g) x i / (g - 1)       when i != 0
    monthly deposit = (target - start) / n                     when i == 0
    Zero when the starting amount alone already grows past the target.
    """

    _positive("target", target)
    _non_negative("start", start)
    _percent("annual_return_pct", annual_return_pct, Decimal("-99"), HUNDRED)
    _years(years)
    i = annual_return_pct / HUNDRED / 12
    n = 12 * years
    g = (1 + i) ** n
    gap = target - start * g
    if gap <= 0:
        payment = Decimal("0")
    elif i == 0:
        payment = gap / n
    else:
        payment = gap * i / (g - 1)
    return {
        "monthly_deposit": _money(payment),
        "total_deposited": _money(start + payment * n),
        "start_grows_to": _money(start * g),
    }


def drawdown_recovery(drop_pct: Decimal) -> dict[str, Decimal]:
    """Gain needed to win back a loss: a drop of d needs a rise of d / (1 - d).

    required gain % = 100 x (1 / (1 - drop_pct / 100) - 1)
    Example: -50% needs +100%.
    """

    _percent("drop_pct", drop_pct, Decimal("0"), Decimal("99.99"))
    d = drop_pct / HUNDRED
    return {"drop_pct": _ratio(drop_pct), "required_gain_pct": _ratio(HUNDRED * (1 / (1 - d) - 1))}


def dca_average_cost(buys: Sequence[tuple[Decimal, Decimal]]) -> dict[str, Decimal]:
    """Average cost of buying a fixed or varying amount of money at different prices.

    Each buy is ``(amount of money, price)``.
    units        = sum(amount / price)
    average cost = sum(amount) / units      (a harmonic mean of the prices, weighted by amount)
    """

    if not buys:
        raise ValueError("add at least one buy")
    if len(buys) > 1000:
        raise ValueError("at most 1000 buys")
    spent = units = Decimal("0")
    for amount, price in buys:
        _positive("amount", amount)
        _positive("price", price)
        spent += amount
        units += amount / price
    return {
        "total_spent": _money(spent),
        "units": units.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP),
        "average_cost": _money(spent / units),
        "simple_average_price": _money(sum((p for _, p in buys), Decimal("0")) / len(buys)),
    }


def cost_impact(
    turnover: Decimal, round_trip_bps: Decimal, equity: Decimal | None = None
) -> dict[str, Decimal | None]:
    """Yearly cost of spreads and fees on the money traded.

    cost        = turnover x round_trip_bps / 10 000       (1 basis point = 0.01%)
    drag %      = 100 x cost / equity                       (only when equity is given)
    ``turnover`` is the money bought and sold again in a year; one round trip = buy + sell.
    """

    _non_negative("turnover", turnover)
    _percent("round_trip_bps", round_trip_bps, Decimal("0"), Decimal("10000"))
    cost = turnover * round_trip_bps / Decimal("10000")
    drag: Decimal | None = None
    if equity is not None:
        _positive("equity", equity)
        drag = _ratio(HUNDRED * cost / equity)
    return {"annual_cost": _money(cost), "drag_pct": drag}


def nisab_from_metal(price_per_gram: Decimal, grams: Decimal = NISAB_GOLD_GRAMS) -> Decimal:
    """Nisab value in the caller's currency: grams x price per gram (87.48 g gold by default)."""

    _positive("price_per_gram", price_per_gram)
    _positive("grams", grams)
    return _money(price_per_gram * grams)


def zakat(zakatable_wealth: Decimal, nisab: Decimal) -> dict[str, Any]:
    """Zakat due on wealth held for a lunar year.

    due = 2.5% x zakatable wealth      when zakatable wealth >= nisab
    due = 0                            otherwise
    Zakat is on the whole amount once it reaches the nisab, not only on the part above it.
    ``nisab`` is supplied by the caller in the same currency (see ``nisab_from_metal``).
    Zakatable wealth = cash + trade goods + zakatable share of investments - debts due now;
    ask a scholar for your own case.
    """

    _non_negative("zakatable_wealth", zakatable_wealth)
    _positive("nisab", nisab)
    above = zakatable_wealth >= nisab
    return {
        "due": _money(zakatable_wealth * ZAKAT_RATE) if above else Decimal("0.00"),
        "above_nisab": above,
        "nisab": _money(nisab),
    }


def purification(dividends: Decimal, non_compliant_pct: Decimal) -> dict[str, Decimal]:
    """Part of dividends to give away because it came from non-compliant revenue.

    amount to purify = dividends x non_compliant_pct / 100
    The percent is the company's non-permissible income share (AAOIFI Standard 21 allows
    holding only when it is under 5%).
    """

    _non_negative("dividends", dividends)
    _percent("non_compliant_pct", non_compliant_pct, Decimal("0"), HUNDRED)
    amount = dividends * non_compliant_pct / HUNDRED
    return {"purify": _money(amount), "keep": _money(dividends - amount)}


# ---- JSON entry point -------------------------------------------------------------------


def parse_decimal(payload: Mapping[str, Any], key: str, default: str | None = None) -> Decimal:
    """Read ``payload[key]`` as a finite Decimal (commas allowed); ValueError otherwise."""

    raw = payload.get(key, default)
    if raw is None or isinstance(raw, bool) or str(raw).strip() == "":
        raise ValueError(f"{key} is required")
    if not isinstance(raw, int | float | str | Decimal) or len(str(raw)) > 40:
        raise ValueError(f"{key} must be a number")
    try:
        value = Decimal(str(raw).strip().replace(",", ""))
    except InvalidOperation:
        raise ValueError(f"{key} must be a number") from None
    if not value.is_finite():
        raise ValueError(f"{key} must be a number")
    return value


def _int(payload: Mapping[str, Any], key: str) -> int:
    value = parse_decimal(payload, key)
    if value != value.to_integral_value():
        raise ValueError(f"{key} must be a whole number")
    return int(value)


def _optional(payload: Mapping[str, Any], key: str) -> Decimal | None:
    raw = payload.get(key)
    return None if raw is None or str(raw).strip() == "" else parse_decimal(payload, key)


def _buys(payload: Mapping[str, Any]) -> list[tuple[Decimal, Decimal]]:
    rows = payload.get("buys")
    if isinstance(rows, str):  # "amount@price, amount@price" from a text box
        rows = [dict(zip(("amount", "price"), r.split("@"), strict=False)) for r in rows.split(",")]
    if not isinstance(rows, list):
        raise ValueError("buys must be a list of {amount, price}")
    out = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each buy needs an amount and a price")
        out.append((parse_decimal(row, "amount"), parse_decimal(row, "price")))
    return out


def _zakat(p: Mapping[str, Any]) -> dict[str, Any]:
    nisab = _optional(p, "nisab")
    if nisab is None:
        grams = _optional(p, "nisab_grams") or NISAB_GOLD_GRAMS
        nisab = nisab_from_metal(parse_decimal(p, "gold_price_per_gram"), grams)
    return zakat(parse_decimal(p, "zakatable_wealth"), nisab)


CALCULATORS: dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {
    "position_size": lambda p: position_size(
        parse_decimal(p, "equity"),
        parse_decimal(p, "risk_pct"),
        parse_decimal(p, "entry"),
        parse_decimal(p, "stop"),
    ),
    "reward_risk": lambda p: reward_risk(
        parse_decimal(p, "entry"), parse_decimal(p, "stop"), parse_decimal(p, "target")
    ),
    "compound": lambda p: compound_growth(
        parse_decimal(p, "start", "0"),
        parse_decimal(p, "monthly_deposit", "0"),
        parse_decimal(p, "annual_return_pct"),
        _int(p, "years"),
    ),
    "goal": lambda p: deposit_goal(
        parse_decimal(p, "target"),
        _int(p, "years"),
        parse_decimal(p, "annual_return_pct"),
        parse_decimal(p, "start", "0"),
    ),
    "drawdown": lambda p: drawdown_recovery(parse_decimal(p, "drop_pct")),
    "dca": lambda p: dca_average_cost(_buys(p)),
    "cost": lambda p: cost_impact(
        parse_decimal(p, "turnover"), parse_decimal(p, "round_trip_bps"), _optional(p, "equity")
    ),
    "zakat": _zakat,
    "purification": lambda p: purification(
        parse_decimal(p, "dividends"), parse_decimal(p, "non_compliant_pct")
    ),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def run(name: str, payload: Any) -> dict[str, Any]:
    """Run calculator ``name`` on a JSON payload; Decimals come back as strings.

    Raises ``KeyError`` for an unknown calculator and ``ValueError`` for bad input.
    """

    if name not in CALCULATORS:
        raise KeyError(name)
    if not isinstance(payload, Mapping):
        raise ValueError("send a JSON object")
    return dict(_jsonable(CALCULATORS[name](payload)))
