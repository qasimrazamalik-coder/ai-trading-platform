"""Shariah compliance gate: deterministic rules evaluated before any risk check."""

from quant_os_core.shariah.rules import (
    CompanyFinancials,
    Madhhab,
    ShariahDecision,
    ShariahGate,
    ShariahOrderContext,
    ShariahProfile,
    halal_risk_limits,
    purification_amount,
)

__all__ = [
    "CompanyFinancials",
    "Madhhab",
    "ShariahDecision",
    "ShariahGate",
    "ShariahOrderContext",
    "ShariahProfile",
    "halal_risk_limits",
    "purification_amount",
]
