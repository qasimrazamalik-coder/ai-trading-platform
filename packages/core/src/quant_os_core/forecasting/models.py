"""Return and price forecasting: AR(p) on returns and Holt linear trend on log prices.

Daily returns are close to unpredictable, so these models must beat the naive forecast out of
sample (see ``evaluation``) before any of them is trusted. Intervals use residual spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Forecast:
    mean: FloatArray
    lower: FloatArray  # ~95% interval
    upper: FloatArray


@dataclass(frozen=True, slots=True)
class ARModel:
    """r_t = c + phi_1 r_(t-1) + ... + phi_p r_(t-p) + e_t, fitted by least squares."""

    intercept: float
    coefficients: FloatArray
    residual_std: float
    history: FloatArray  # the last p observations, newest last

    @classmethod
    def fit(cls, series: npt.ArrayLike, order: int = 1) -> ARModel:
        y = np.asarray(series, dtype=np.float64)
        if order < 1 or y.ndim != 1 or y.size < order + 20 or not np.isfinite(y).all():
            raise ValueError("need a finite 1D series with at least order + 20 values")
        rows = np.column_stack([y[order - k - 1 : y.size - k - 1] for k in range(order)])
        x = np.column_stack([np.ones(rows.shape[0]), rows])
        target = y[order:]
        beta, *_ = np.linalg.lstsq(x, target, rcond=None)
        resid = target - x @ beta
        dof = max(1, target.size - x.shape[1])
        return cls(
            intercept=float(beta[0]),
            coefficients=np.asarray(beta[1:], dtype=np.float64),
            residual_std=float(math.sqrt(float(resid @ resid) / dof)),
            history=y[-order:].copy(),
        )

    def forecast(self, horizon: int = 1) -> Forecast:
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        lags = list(self.history[::-1])  # newest first
        means = []
        for _ in range(horizon):
            value = self.intercept + float(
                np.dot(self.coefficients, lags[: self.coefficients.size])
            )
            means.append(value)
            lags.insert(0, value)
        mean = np.asarray(means)
        # psi-weights for the h-step error variance
        psi = [1.0]
        for h in range(1, horizon):
            psi.append(
                sum(
                    self.coefficients[j] * psi[h - j - 1]
                    for j in range(min(h, self.coefficients.size))
                )
            )
        spread = 1.96 * self.residual_std * np.sqrt(np.cumsum(np.square(psi)))
        return Forecast(mean, mean - spread, mean + spread)


@dataclass(frozen=True, slots=True)
class HoltLinear:
    """Holt's linear trend on log prices; alpha/beta chosen by one-step error on the data."""

    alpha: float
    beta: float
    level: float
    trend: float
    residual_std: float

    @staticmethod
    def _run(y: FloatArray, alpha: float, beta: float) -> tuple[float, float, float]:
        level, trend = float(y[0]), float(y[1] - y[0])
        sse = 0.0
        for value in y[1:]:
            pred = level + trend
            err = value - pred
            sse += err * err
            new_level = alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (new_level - level) + (1 - beta) * trend
            level = new_level
        return level, trend, sse

    @classmethod
    def fit(cls, prices: npt.ArrayLike) -> HoltLinear:
        p = np.asarray(prices, dtype=np.float64)
        if p.ndim != 1 or p.size < 30 or not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError("need at least 30 positive, finite prices")
        y = np.log(p)
        best = None
        for alpha in np.linspace(0.05, 0.95, 19):
            for beta in np.linspace(0.0, 0.5, 11):
                level, trend, sse = cls._run(y, float(alpha), float(beta))
                if best is None or sse < best[0]:
                    best = (sse, float(alpha), float(beta), level, trend)
        assert best is not None
        best_sse, best_alpha, best_beta, level, trend = best
        return cls(
            best_alpha, best_beta, float(level), float(trend), math.sqrt(best_sse / (y.size - 1))
        )

    def forecast(self, horizon: int = 1) -> Forecast:
        """Price forecasts (exp of the log-level path) with widening intervals."""

        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        steps = np.arange(1, horizon + 1)
        log_mean = self.level + steps * self.trend
        spread = 1.96 * self.residual_std * np.sqrt(steps)
        return Forecast(np.exp(log_mean), np.exp(log_mean - spread), np.exp(log_mean + spread))
