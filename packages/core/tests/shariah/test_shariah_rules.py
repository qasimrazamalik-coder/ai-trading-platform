from decimal import Decimal

import pytest

from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.shariah import (
    CompanyFinancials,
    Madhhab,
    ShariahGate,
    ShariahOrderContext,
    ShariahProfile,
    halal_risk_limits,
    purification_amount,
)


def _instrument(
    asset_class: AssetClass = AssetClass.EQUITY,
    instrument_type: InstrumentType = InstrumentType.SPOT,
) -> Instrument:
    return Instrument(
        instrument_id=InstrumentId(venue="alpaca", symbol="AAPL"),
        asset_class=asset_class,
        instrument_type=instrument_type,
        base_currency="AAPL",
        quote_currency="USD",
        price_precision=2,
        quantity_precision=4,
    )


def _financials(**kwargs: object) -> CompanyFinancials:
    params: dict[str, object] = {
        "sector": "technology",
        "market_cap": Decimal("1000"),
        "interest_bearing_debt": Decimal("100"),
        "interest_bearing_cash_and_securities": Decimal("50"),
        "total_revenue": Decimal("400"),
        "noncompliant_revenue": Decimal("4"),
        "total_assets": Decimal("900"),
        "tangible_assets": Decimal("600"),
        "shares_outstanding": Decimal("100"),
    }
    params.update(kwargs)
    return CompanyFinancials(**params)  # type: ignore[arg-type]


def _ctx(**kwargs: object) -> ShariahOrderContext:
    params: dict[str, object] = {"owned_quantity": Decimal("0"), "free_cash": Decimal("50")}
    params.update(kwargs)
    return ShariahOrderContext(**params)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "instrument_type",
    [
        InstrumentType.PERPETUAL,
        InstrumentType.FUTURE,
        InstrumentType.OPTION,
        InstrumentType.CFD,
        InstrumentType.BINARY,
    ],
)
def test_derivatives_are_always_blocked(instrument_type: InstrumentType) -> None:
    decision = ShariahGate().check_instrument(_instrument(instrument_type=instrument_type))
    assert not decision.allowed
    assert decision.rule_id == "R2"


def test_crypto_follows_profile() -> None:
    crypto = _instrument(AssetClass.CRYPTO)
    assert ShariahGate().check_instrument(crypto).rule_id == "R13"
    permissive = ShariahGate(ShariahProfile(madhhab=Madhhab.PERMISSIVE_SPOT_CRYPTO))
    assert permissive.check_instrument(crypto).allowed
    perp = _instrument(AssetClass.CRYPTO, InstrumentType.PERPETUAL)
    assert not permissive.check_instrument(perp).allowed


def test_forex_speculation_blocked() -> None:
    assert ShariahGate().check_instrument(_instrument(AssetClass.FOREX)).rule_id == "R10"


@pytest.mark.parametrize(
    ("override", "rule"),
    [
        ({"sector": "Conventional_Finance"}, "R7"),
        ({"interest_bearing_debt": Decimal("300")}, "R6"),
        ({"interest_bearing_cash_and_securities": Decimal("301")}, "R6"),
        ({"noncompliant_revenue": Decimal("20")}, "R6"),
        ({"tangible_assets": Decimal("200")}, "R6"),
    ],
)
def test_aaoifi_screen_rejections(override: dict[str, object], rule: str) -> None:
    decision = ShariahGate().screen_company(_financials(**override))
    assert not decision.allowed
    assert decision.rule_id == rule


def test_compliant_company_passes() -> None:
    assert ShariahGate().screen_company(_financials()).allowed


def test_order_rules_margin_leverage_short_and_swap() -> None:
    gate = ShariahGate()
    eq = _instrument()
    fin = _financials()
    buy = {"is_sell": False, "quantity": Decimal("1"), "financials": fin}
    assert gate.check_order(eq, notional=Decimal("10"), context=_ctx(), **buy).allowed
    assert gate.check_order(eq, notional=Decimal("60"), context=_ctx(), **buy).rule_id == "R3"
    margin = _ctx(margin_enabled=True)
    assert gate.check_order(eq, notional=Decimal("10"), context=margin, **buy).rule_id == "R3"
    swap = _ctx(financing_charge_rate=Decimal("0.0001"))
    assert gate.check_order(eq, notional=Decimal("10"), context=swap, **buy).rule_id == "R5"
    short = gate.check_order(
        eq, is_sell=True, quantity=Decimal("2"), notional=Decimal("20"), context=_ctx()
    )
    assert short.rule_id == "R4"
    no_data = gate.check_order(
        eq, is_sell=False, quantity=Decimal("1"), notional=Decimal("10"), context=_ctx()
    )
    assert no_data.rule_id == "R6"
    unlicensed = _ctx(venue_licensed=False)
    assert gate.check_order(eq, notional=Decimal("10"), context=unlicensed, **buy).rule_id == "R14"


def test_exiting_a_now_noncompliant_holding_is_allowed() -> None:
    gate = ShariahGate()
    crypto = _instrument(AssetClass.CRYPTO)
    exit_ok = gate.check_order(
        crypto,
        is_sell=True,
        quantity=Decimal("1"),
        notional=Decimal("10"),
        context=_ctx(owned_quantity=Decimal("1")),
    )
    assert exit_ok.allowed


def test_purification_formula() -> None:
    assert purification_amount(_financials(), Decimal("10")) == Decimal("0.4")
    with pytest.raises(ValueError):
        purification_amount(_financials(), Decimal("-1"))


def test_halal_limits_lock_leverage_and_shorting() -> None:
    limits = halal_risk_limits()
    assert limits.max_gross_leverage == Decimal("1.0")
    assert limits.allow_short is False
    with pytest.raises(ValueError):
        halal_risk_limits(max_gross_leverage=Decimal("2"))


def test_invalid_financials_rejected() -> None:
    with pytest.raises(ValueError):
        _financials(market_cap=Decimal("0"))
    with pytest.raises(ValueError):
        _financials(interest_bearing_debt=Decimal("-1"))


def test_owner_approved_fund_passes_r6_only_for_listed_etfs() -> None:
    from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
    from quant_os_core.shariah import ShariahGate, ShariahOrderContext, ShariahProfile

    def etf(symbol: str, asset: AssetClass = AssetClass.ETF) -> Instrument:
        return Instrument(
            instrument_id=InstrumentId(venue="PAPER", symbol=symbol),
            asset_class=asset,
            instrument_type=InstrumentType.SPOT,
            base_currency=symbol,
            quote_currency="USD",
            price_precision=2,
            quantity_precision=4,
        )

    gate = ShariahGate(ShariahProfile(owner_approved_funds=frozenset({"SPSK"})))
    context = ShariahOrderContext(owned_quantity=Decimal(0), free_cash=Decimal(1000))

    def check(instrument: Instrument) -> tuple[bool, str]:
        d = gate.check_order(
            instrument, is_sell=False, quantity=Decimal(1), notional=Decimal(10), context=context
        )
        return d.allowed, d.rule_id

    assert check(etf("SPSK")) == (True, "R6")
    assert check(etf("SPUS")) == (False, "R6")  # not on the list: still needs screening data
    assert check(etf("SPSK", AssetClass.EQUITY)) == (False, "R6")  # ETFs only
