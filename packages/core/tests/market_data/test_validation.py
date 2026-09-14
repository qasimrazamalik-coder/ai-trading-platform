from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quant_os_core.market_data.models import Candle, Quote
from quant_os_core.market_data.validation import MarketDataValidator


def test_clean_batch_passes(minute_candles: list[Candle]) -> None:
    report = MarketDataValidator().validate_candles(
        minute_candles,
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert report.is_clean


def test_gap_is_detected(minute_candles: list[Candle]) -> None:
    batch = [minute_candles[0], *minute_candles[2:]]
    report = MarketDataValidator().validate_candles(batch, now=datetime(2026, 1, 2, tzinfo=UTC))
    assert len(report.gaps) == 1
    assert report.gaps[0].missing_intervals == 1


def test_duplicate_and_unsorted_are_detected(minute_candles: list[Candle]) -> None:
    batch = [minute_candles[1], minute_candles[0], minute_candles[0]]
    report = MarketDataValidator().validate_candles(batch, now=datetime(2026, 1, 2, tzinfo=UTC))
    assert report.is_sorted is False
    assert report.duplicate_timestamps == (minute_candles[0].timestamp,)


def test_future_timestamp_is_detected(minute_candles: list[Candle]) -> None:
    report = MarketDataValidator(max_future_drift=timedelta(0)).validate_candles(
        minute_candles,
        now=datetime(2025, 12, 31, tzinfo=UTC),
    )
    assert len(report.future_timestamps) == len(minute_candles)


def test_quote_staleness(btc) -> None:  # type annotation intentionally fixture-agnostic
    quote = Quote(
        instrument_id=btc,
        timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        source="test",
        bid_price="99",
        ask_price="100",
        bid_size="1",
        ask_size="1",
    )
    assert MarketDataValidator.quote_is_stale(
        quote,
        now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        max_age=timedelta(seconds=10),
    )


def test_extreme_return_is_flagged(minute_candles: list[Candle]) -> None:
    # Supply enough ordinary non-identical returns for a non-zero MAD, then inject a shock.
    base = minute_candles[:]
    shock = base[-1].model_copy(
        update={
            "timestamp": base[-1].timestamp + timedelta(minutes=1),
            "close_time": base[-1].close_time + timedelta(minutes=1),
            "open": Decimal("106"),
            "high": Decimal("1002"),
            "low": Decimal("105"),
            "close": Decimal("1000"),
        }
    )
    report = MarketDataValidator().validate_candles(
        [*base, shock],
        now=datetime(2026, 1, 2, tzinfo=UTC),
        outlier_threshold=5,
    )
    assert report.outliers
