"""Default entry signal for the paper loop: Donchian breakout with an ATR stop.

Breakouts above the N-bar high in an uptrend are one of the best-documented trend signals.
The stop sits ``atr_mult`` ATRs below entry and the target ``reward_to_risk`` times that
distance above, so every plan carries the exit numbers the pipeline requires. Confidence is a
fixed prior until the meta-labeller supplies calibrated probabilities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from quant_os_core.agents.exits import Bar
from quant_os_core.agents.trade_plan import TradePlan


def atr(bars: Sequence[Bar], window: int = 14) -> Decimal | None:
    if len(bars) < window + 1:
        return None
    ranges = []
    for prev, cur in zip(bars[-window - 1 : -1], bars[-window:], strict=True):
        ranges.append(
            max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))
        )
    return sum(ranges, Decimal("0")) / Decimal(window)


@dataclass(frozen=True, slots=True)
class DonchianBreakout:
    lookback: int = 20
    trend_window: int = 50
    atr_mult: Decimal = Decimal("2")
    reward_to_risk: Decimal = Decimal("2")
    confidence: Decimal = Decimal("0.6")

    def __call__(self, symbol: str, bars: Sequence[Bar]) -> TradePlan | None:
        if len(bars) < max(self.lookback, self.trend_window) + 2:
            return None
        last = bars[-1]
        prior_high = max(b.high for b in bars[-self.lookback - 1 : -1])
        trend = sum((b.close for b in bars[-self.trend_window :]), Decimal("0")) / Decimal(
            self.trend_window
        )
        if last.close <= prior_high or last.close <= trend:
            return None
        volatility = atr(bars)
        if volatility is None or volatility <= 0:
            return None
        risk = volatility * self.atr_mult
        stop = last.close - risk
        if stop <= 0:
            return None
        return TradePlan(
            symbol=symbol,
            entry_price=last.close,
            profit_target=last.close + risk * self.reward_to_risk,
            stop_loss=stop,
            invalidation_condition=f"close back below the {self.lookback}-bar breakout level",
            confidence=self.confidence,
            justification=f"{self.lookback}-bar breakout above the {self.trend_window}-bar average",
        )


@dataclass(frozen=True, slots=True)
class ForecastBreakout:
    """Breakout whose stop comes from a volatility *forecast* over the holding horizon.

    EWMA (RiskMetrics, lambda 0.94) forecasts next-bar variance; the horizon volatility is
    ``sigma * sqrt(horizon)``. Stops scale with expected movement instead of past range, so
    they widen before turbulent periods and tighten in calm ones.
    """

    lookback: int = 20
    trend_window: int = 50
    horizon: int = 10
    stop_sigmas: Decimal = Decimal("1.5")
    reward_to_risk: Decimal = Decimal("2")
    confidence: Decimal = Decimal("0.6")

    def __call__(self, symbol: str, bars: Sequence[Bar]) -> TradePlan | None:
        import math

        import numpy as np

        from quant_os_core.forecasting.volatility import ewma_variance

        if len(bars) < max(self.lookback, self.trend_window, 60) + 2:
            return None
        last = bars[-1]
        prior_high = max(b.high for b in bars[-self.lookback - 1 : -1])
        trend = sum((b.close for b in bars[-self.trend_window :]), Decimal("0")) / Decimal(
            self.trend_window
        )
        if last.close <= prior_high or last.close <= trend:
            return None
        closes = np.array([float(b.close) for b in bars[-250:]])
        sigma = math.sqrt(float(ewma_variance(np.diff(np.log(closes)))[-1]) * self.horizon)
        risk = last.close * Decimal(str(round(sigma, 8))) * self.stop_sigmas
        if risk <= 0 or last.close - risk <= 0:
            return None
        return TradePlan(
            symbol=symbol,
            entry_price=last.close,
            profit_target=last.close + risk * self.reward_to_risk,
            stop_loss=last.close - risk,
            invalidation_condition=f"close back below the {self.lookback}-bar breakout level",
            confidence=self.confidence,
            justification=f"breakout; stop = {self.stop_sigmas} x {self.horizon}-bar EWMA vol",
        )
