from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from quant_os_core.backtesting.engine import StrategyContext
from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import Candle
from quant_os_core.strategies import (
    CanSlimInputs,
    MinerviniInputs,
    ParameterDefinition,
    SingleAssetTargetAdapter,
    StrategyCategory,
    StrategyMetadata,
    StrategySignals,
    avellaneda_stoikov_quote,
    canslim_score,
    cash_and_carry_signal,
    credibility_weighted_event_signal,
    majority_vote,
    microprice,
    minervini_trend_template,
    pairs_zscore_signal,
    weighted_ensemble,
)


def test_metadata_rejects_blank_identity() -> None:
    with pytest.raises(ValueError, match="strategy_id"):
        StrategyMetadata(" ", "Name", StrategyCategory.TREND, "x")
    with pytest.raises(ValueError, match="name"):
        StrategyMetadata("id", " ", StrategyCategory.TREND, "x")


def test_parameter_definition_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="minimum"):
        ParameterDefinition("x", 1, minimum=2, maximum=1)


def test_strategy_signals_reject_bad_diagnostic_shape() -> None:
    metadata = StrategyMetadata("x", "X", StrategyCategory.TREND, "x")
    with pytest.raises(ValueError, match="match positions"):
        StrategySignals(np.zeros(4), metadata, diagnostics={"x": np.zeros(3)})


def test_ensemble_guards() -> None:
    with pytest.raises(ValueError):
        weighted_ensemble([1, 2, 3])
    with pytest.raises(ValueError):
        weighted_ensemble([[1, 0], [0, 1]], weights=[0, 0])
    with pytest.raises(ValueError):
        weighted_ensemble([[2, 0], [0, 1]])
    with pytest.raises(ValueError):
        majority_vote([[1, 0]], minimum_agreement=0)


def test_screening_guards() -> None:
    with pytest.raises(ValueError):
        minervini_trend_template(
            MinerviniInputs(100, 90, 80, 70, 69, 60, 110, 120)
        )
    with pytest.raises(ValueError):
        canslim_score(
            CanSlimInputs(np.nan, 0.2, 0.2, 0.5, 0.5, 90, 0.5, 0.5)
        )


def test_pairs_parameter_guards() -> None:
    x = np.linspace(100, 110, 100)
    y = 2 * x + np.sin(np.arange(100))
    with pytest.raises(ValueError, match="entry_z"):
        pairs_zscore_signal(y, x, window=20, entry_z=1, exit_z=1)
    with pytest.raises(ValueError, match="max_cointegration"):
        pairs_zscore_signal(y, x, window=20, max_cointegration_pvalue=2)


def test_cash_and_carry_guard() -> None:
    with pytest.raises(ValueError):
        cash_and_carry_signal(
            [100],
            [101],
            days_to_expiry=[30],
            min_edge=-1,
        )


def test_microprice_size_length_guard() -> None:
    with pytest.raises(ValueError, match="equal length"):
        microprice([99, 99], [101, 101], [1], [1])


def test_event_threshold_guard() -> None:
    with pytest.raises(ValueError):
        credibility_weighted_event_signal([0.5], [0.5], minimum_credibility=2)


def test_market_maker_additional_guards() -> None:
    with pytest.raises(ValueError):
        avellaneda_stoikov_quote(
            mid_price=100,
            inventory=0,
            volatility=-1,
            risk_aversion=0.1,
            time_remaining=1,
            liquidity_k=1,
        )
    with pytest.raises(ValueError):
        avellaneda_stoikov_quote(
            mid_price=100,
            inventory=0,
            volatility=0.1,
            risk_aversion=0.1,
            time_remaining=-1,
            liquidity_k=1,
        )


def _candle(symbol: str) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return Candle(
        instrument_id=InstrumentId(venue="TEST", symbol=symbol),
        timestamp=start,
        source="test",
        timeframe="1m",
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("10"),
        close_time=start + timedelta(minutes=1),
    )


def _context() -> StrategyContext:
    return StrategyContext(
        event_index=0,
        instrument_bar_index=0,
        cash=Decimal("1000"),
        equity=Decimal("1000"),
        gross_exposure=Decimal("0"),
        net_exposure=Decimal("0"),
    )


def test_event_adapter_constructor_and_multimarket_guards() -> None:
    with pytest.raises(ValueError):
        SingleAssetTargetAdapter(signal_function=lambda _: 0, max_units=Decimal("0"))
    strategy = SingleAssetTargetAdapter(
        signal_function=lambda _: 0.0,
        max_units=Decimal("1"),
        minimum_history=1,
    )
    strategy.on_bar(_candle("A"), _context())
    with pytest.raises(ValueError, match="multiple instruments"):
        strategy.on_bar(_candle("B"), _context())
    strategy.reset()
    assert strategy.on_bar(_candle("B"), _context()) == ()
