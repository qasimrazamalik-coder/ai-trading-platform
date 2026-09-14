from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.corporate_actions import CorporateAction, CorporateActionType


def test_split_requires_ratio() -> None:
    with pytest.raises(ValidationError):
        CorporateAction(
            instrument_id=InstrumentId(venue="NASDAQ", symbol="AAPL"),
            action_type=CorporateActionType.SPLIT,
            effective_at=datetime(2026, 1, 1, tzinfo=UTC),
            source="exchange",
        )


def test_cash_dividend_payload() -> None:
    action = CorporateAction(
        instrument_id=InstrumentId(venue="PSX", symbol="OGDC"),
        action_type=CorporateActionType.CASH_DIVIDEND,
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        source="psx",
        cash_amount="3.50",
        currency="PKR",
    )
    assert action.cash_amount == Decimal("3.50")
