"""Market-data integrity checks used before research, backtesting, and live decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import median

from quant_os_core.market_data.models import Candle, Quote
from quant_os_core.market_data.timeframes import Timeframe


@dataclass(frozen=True, slots=True)
class Gap:
    expected_at: datetime
    observed_at: datetime
    missing_intervals: int


@dataclass(frozen=True, slots=True)
class Outlier:
    timestamp: datetime
    log_return: float
    robust_z_score: float


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    record_count: int
    duplicate_timestamps: tuple[datetime, ...]
    gaps: tuple[Gap, ...]
    outliers: tuple[Outlier, ...]
    future_timestamps: tuple[datetime, ...]
    is_sorted: bool

    @property
    def is_clean(self) -> bool:
        return not (
            self.duplicate_timestamps
            or self.gaps
            or self.future_timestamps
            or self.outliers
            or not self.is_sorted
        )


class MarketDataValidator:
    def __init__(self, *, max_future_drift: timedelta = timedelta(seconds=5)) -> None:
        self.max_future_drift = max_future_drift

    def validate_candles(
        self,
        candles: list[Candle],
        *,
        now: datetime | None = None,
        outlier_threshold: float = 8.0,
    ) -> DataQualityReport:
        if not candles:
            return DataQualityReport(0, (), (), (), (), True)

        instrument = candles[0].instrument_id
        timeframe_label = candles[0].timeframe
        if any(item.instrument_id != instrument for item in candles):
            raise ValueError("validation batch contains multiple instruments")
        if any(item.timeframe != timeframe_label for item in candles):
            raise ValueError("validation batch contains multiple timeframes")

        observed_order = [item.timestamp for item in candles]
        is_sorted = observed_order == sorted(observed_order)
        ordered = sorted(candles, key=lambda item: item.timestamp)
        timeframe = Timeframe.parse(timeframe_label)

        counts: dict[datetime, int] = {}
        for item in ordered:
            counts[item.timestamp] = counts.get(item.timestamp, 0) + 1
        duplicates = tuple(timestamp for timestamp, count in counts.items() if count > 1)

        unique = []
        seen: set[datetime] = set()
        for item in ordered:
            if item.timestamp not in seen:
                unique.append(item)
                seen.add(item.timestamp)

        gaps: list[Gap] = []
        for previous, current in zip(unique, unique[1:]):
            elapsed = int((current.timestamp - previous.timestamp).total_seconds())
            if elapsed > timeframe.seconds:
                missing = max(0, elapsed // timeframe.seconds - 1)
                gaps.append(
                    Gap(
                        expected_at=previous.timestamp + timeframe.delta,
                        observed_at=current.timestamp,
                        missing_intervals=missing,
                    )
                )

        reference_now = (now or datetime.now(UTC)).astimezone(UTC)
        latest_allowed = reference_now + self.max_future_drift
        future = tuple(item.timestamp for item in unique if item.timestamp > latest_allowed)
        outliers = self._return_outliers(unique, threshold=outlier_threshold)

        return DataQualityReport(
            record_count=len(candles),
            duplicate_timestamps=duplicates,
            gaps=tuple(gaps),
            outliers=tuple(outliers),
            future_timestamps=future,
            is_sorted=is_sorted,
        )

    @staticmethod
    def quote_is_stale(quote: Quote, *, now: datetime, max_age: timedelta) -> bool:
        reference_now = now.astimezone(UTC)
        return reference_now - quote.timestamp > max_age

    @staticmethod
    def _return_outliers(candles: list[Candle], *, threshold: float) -> list[Outlier]:
        if len(candles) < 4:
            return []

        returns: list[tuple[datetime, float]] = []
        for previous, current in zip(candles, candles[1:]):
            value = math.log(float(current.close / previous.close))
            returns.append((current.timestamp, value))

        values = [value for _, value in returns]
        center = median(values)
        deviations = [abs(value - center) for value in values]
        mad = median(deviations)
        if mad == 0:
            return []

        scale = 0.6744897501960817
        result: list[Outlier] = []
        for timestamp, value in returns:
            robust_z = scale * (value - center) / mad
            if abs(robust_z) >= threshold:
                result.append(Outlier(timestamp, value, robust_z))
        return result
