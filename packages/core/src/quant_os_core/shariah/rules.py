"""Deterministic Shariah compliance gate.

Every rule here is plain code so that it can be audited, tested and signed off by a
scholar. Language models may explain a decision but never make or override one.

Rule identifiers follow ``docs`` section 4 of the research report (R1-R17). Rules marked
"scholar-dependent" are configurable through :class:`ShariahProfile`; consensus rules are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from quant_os_core.instruments import AssetClass, Instrument, InstrumentType
from quant_os_core.risk_engine import RiskLimits


class Madhhab(StrEnum):
    """Compliance profile presets. Names describe scholarly positions, not people."""

    PK_HANAFI = "pk_hanafi"  # follows Darul Uloom Karachi (crypto disabled, 2026 fatwa)
    PERMISSIVE_SPOT_CRYPTO = "permissive_spot_crypto"  # scholars permitting spot crypto


ALLOWED_INSTRUMENT_TYPES: frozenset[InstrumentType] = frozenset(
    {InstrumentType.SPOT, InstrumentType.CASH}
)
BLOCKED_INSTRUMENT_TYPES: frozenset[InstrumentType] = frozenset(
    {
        InstrumentType.PERPETUAL,
        InstrumentType.FUTURE,
        InstrumentType.OPTION,
        InstrumentType.CFD,
        InstrumentType.BINARY,
    }
)
BLOCKED_ASSET_CLASSES: frozenset[AssetClass] = frozenset(
    {
        AssetClass.FUTURE,
        AssetClass.OPTION,
        AssetClass.CFD,
        AssetClass.BINARY,
        AssetClass.PREDICTION,
        AssetClass.INDEX,  # index products are derivatives; trade constituents or ETFs instead
    }
)
CORE_EXCLUDED_SECTORS: frozenset[str] = frozenset(
    {
        "alcohol",
        "pork",
        "conventional_finance",
        "conventional_insurance",
        "gambling",
        "adult_entertainment",
        "tobacco",
    }
)


@dataclass(frozen=True, slots=True)
class ShariahProfile:
    madhhab: Madhhab = Madhhab.PK_HANAFI
    allow_pre_settlement_resale: bool = True  # R9, AAOIFI SS21 clause 3/7
    max_interest_debt_to_market_cap: Decimal = Decimal("0.30")  # R6
    max_interest_cash_to_market_cap: Decimal = Decimal("0.30")  # R6
    max_noncompliant_revenue_ratio: Decimal = Decimal("0.05")  # R6
    min_tangible_asset_ratio: Decimal = Decimal("0.3333")  # R6, SS21 3/17-3/20
    extra_excluded_sectors: frozenset[str] = field(default_factory=frozenset)  # R7 optional

    @property
    def crypto_enabled(self) -> bool:
        return self.madhhab is Madhhab.PERMISSIVE_SPOT_CRYPTO

    @property
    def excluded_sectors(self) -> frozenset[str]:
        return CORE_EXCLUDED_SECTORS | self.extra_excluded_sectors


@dataclass(frozen=True, slots=True)
class CompanyFinancials:
    """Inputs for the AAOIFI Standard 21 ratio screen (R6) and sector screen (R7)."""

    sector: str
    market_cap: Decimal
    interest_bearing_debt: Decimal
    interest_bearing_cash_and_securities: Decimal
    total_revenue: Decimal
    noncompliant_revenue: Decimal
    total_assets: Decimal
    tangible_assets: Decimal
    shares_outstanding: Decimal

    def __post_init__(self) -> None:
        positive = (self.market_cap, self.total_revenue, self.total_assets, self.shares_outstanding)
        if any(value <= 0 for value in positive):
            raise ValueError("market cap, revenue, assets and shares must be positive")
        non_negative = (
            self.interest_bearing_debt,
            self.interest_bearing_cash_and_securities,
            self.noncompliant_revenue,
            self.tangible_assets,
        )
        if any(value < 0 for value in non_negative):
            raise ValueError("balance-sheet inputs cannot be negative")


@dataclass(frozen=True, slots=True)
class ShariahDecision:
    allowed: bool
    rule_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ShariahOrderContext:
    """What the gate needs to know about the account when an order is proposed."""

    owned_quantity: Decimal
    free_cash: Decimal
    margin_enabled: bool = False
    financing_charge_rate: Decimal = Decimal("0")
    venue_licensed: bool = True


_ALLOW = ShariahDecision(True, "OK", "passes all applicable Shariah rules")


class ShariahGate:
    """Runs before the risk engine; any rejection here is final."""

    def __init__(self, profile: ShariahProfile | None = None) -> None:
        self.profile = profile or ShariahProfile()

    def check_instrument(self, instrument: Instrument) -> ShariahDecision:
        if instrument.instrument_type in BLOCKED_INSTRUMENT_TYPES:
            return ShariahDecision(
                False, "R2", f"{instrument.instrument_type} contracts are not permissible"
            )
        if instrument.asset_class in BLOCKED_ASSET_CLASSES:
            return ShariahDecision(False, "R2", f"{instrument.asset_class} products are blocked")
        if instrument.instrument_type not in ALLOWED_INSTRUMENT_TYPES:
            return ShariahDecision(False, "R1", "only spot ownership is permitted")
        if instrument.asset_class is AssetClass.CRYPTO and not self.profile.crypto_enabled:
            return ShariahDecision(
                False, "R13", "crypto is disabled for the selected profile (PK Hanafi default)"
            )
        if instrument.asset_class is AssetClass.FOREX:
            return ShariahDecision(
                False,
                "R10",
                "speculative FX is blocked; use genuine spot conversion at a licensed local venue",
            )
        return _ALLOW

    def screen_company(self, financials: CompanyFinancials) -> ShariahDecision:
        p = self.profile
        if financials.sector.strip().lower() in p.excluded_sectors:
            return ShariahDecision(False, "R7", f"sector '{financials.sector}' is excluded")
        debt_ratio = financials.interest_bearing_debt / financials.market_cap
        if debt_ratio >= p.max_interest_debt_to_market_cap:
            return ShariahDecision(
                False, "R6", f"interest debt/market cap {debt_ratio:.2%} >= limit"
            )
        cash_ratio = financials.interest_bearing_cash_and_securities / financials.market_cap
        if cash_ratio >= p.max_interest_cash_to_market_cap:
            return ShariahDecision(
                False, "R6", f"interest-bearing cash/market cap {cash_ratio:.2%} >= limit"
            )
        revenue_ratio = financials.noncompliant_revenue / financials.total_revenue
        if revenue_ratio >= p.max_noncompliant_revenue_ratio:
            return ShariahDecision(
                False, "R6", f"non-compliant revenue {revenue_ratio:.2%} >= limit"
            )
        tangible_ratio = financials.tangible_assets / financials.total_assets
        if tangible_ratio < p.min_tangible_asset_ratio:
            return ShariahDecision(
                False, "R6", f"tangible assets {tangible_ratio:.2%} below one third"
            )
        return _ALLOW

    def check_order(
        self,
        instrument: Instrument,
        *,
        is_sell: bool,
        quantity: Decimal,
        notional: Decimal,
        context: ShariahOrderContext,
        financials: CompanyFinancials | None = None,
    ) -> ShariahDecision:
        instrument_decision = self.check_instrument(instrument)
        # Selling an already-owned asset to exit is always allowed, even if it was delisted
        # from the compliant list; buying more is not.
        exiting_owned = is_sell and quantity <= context.owned_quantity
        if not instrument_decision.allowed and not exiting_owned:
            return instrument_decision
        if context.margin_enabled:
            return ShariahDecision(False, "R3", "margin accounts are not permissible (riba)")
        if context.financing_charge_rate > 0:
            return ShariahDecision(
                False, "R5", "positions with financing or swap charges are blocked"
            )
        if not context.venue_licensed:
            return ShariahDecision(False, "R14", "venue is not licensed in the user's jurisdiction")
        if is_sell:
            if quantity > context.owned_quantity:
                return ShariahDecision(False, "R4", "short selling is not permissible")
            return _ALLOW
        if notional > context.free_cash:
            return ShariahDecision(
                False, "R3", "order exceeds free cash; leverage is not permitted"
            )
        if instrument.asset_class in {AssetClass.EQUITY, AssetClass.ETF}:
            if financials is None:
                return ShariahDecision(False, "R6", "no screening data; cannot verify compliance")
            screen = self.screen_company(financials)
            if not screen.allowed:
                return screen
        return _ALLOW


def halal_risk_limits(**overrides: Decimal) -> RiskLimits:
    """Risk-engine limits with leverage and short selling switched off (R3, R4)."""

    params: dict[str, Decimal] = {
        "max_position_fraction": Decimal("0.20"),
        "max_daily_loss_fraction": Decimal("0.03"),
        "max_drawdown_fraction": Decimal("0.15"),
    }
    params.update(overrides)
    for locked in ("max_gross_leverage", "max_net_leverage"):
        if locked in params:
            raise ValueError(f"{locked} is fixed at 1.0 in halal mode")
    return RiskLimits(
        max_gross_leverage=Decimal("1.0"),
        max_net_leverage=Decimal("1.0"),
        allow_short=False,
        **params,  # type: ignore[arg-type]
    )


def purification_amount(financials: CompanyFinancials, shares_held: Decimal) -> Decimal:
    """AAOIFI SS21 3/4/6/4: (non-compliant income / shares outstanding) x shares held."""

    if shares_held < 0:
        raise ValueError("shares_held cannot be negative")
    return financials.noncompliant_revenue / financials.shares_outstanding * shares_held
