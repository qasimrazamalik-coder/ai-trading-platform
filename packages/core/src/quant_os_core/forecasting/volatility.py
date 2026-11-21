"""Volatility forecasting: EWMA (RiskMetrics) and GARCH(1,1) fitted by maximum likelihood.

Volatility is far more forecastable than returns (it clusters), which is why the system uses
forecasts here for stop distances and position sizes rather than for direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize  # type: ignore[import-untyped]

FloatArray = npt.NDArray[np.float64]


def _returns(values: npt.ArrayLike, min_size: int) -> FloatArray:
    r = np.asarray(values, dtype=np.float64)
    if r.ndim != 1 or r.size < min_size or not np.isfinite(r).all():
        raise ValueError(f"returns must be a finite 1D array with at least {min_size} values")
    return r


def ewma_variance(returns: npt.ArrayLike, lam: float = 0.94) -> FloatArray:
    """One-step-ahead variance forecasts: ``out[t]`` is known after return ``t``.

    sigma^2_(t+1) = lam * sigma^2_t + (1 - lam) * r_t^2, seeded with the first 20 returns' variance.
    """

    if not 0 < lam < 1:
        raise ValueError("lam must be in (0, 1)")
    r = _returns(returns, 21)
    out = np.empty_like(r)
    var = float(np.var(r[:20], ddof=1))
    for t, x in enumerate(r):
        var = lam * var + (1 - lam) * x * x
        out[t] = var
    return out


@dataclass(frozen=True, slots=True)
class Garch11:
    """sigma^2_t = omega + alpha * r_(t-1)^2 + beta * sigma^2_(t-1), Gaussian errors."""

    omega: float
    alpha: float
    beta: float
    last_variance: float  # one-step-ahead variance after the last fitted return

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        return self.omega / (1 - self.persistence)

    @staticmethod
    def _filter(r: FloatArray, omega: float, alpha: float, beta: float) -> FloatArray:
        var = np.empty(r.size + 1)
        var[0] = float(np.var(r, ddof=1))
        for t in range(r.size):
            var[t + 1] = omega + alpha * r[t] * r[t] + beta * var[t]
        return var

    @classmethod
    def fit(cls, returns: npt.ArrayLike) -> Garch11:
        r = _returns(returns, 100)
        r = r - r.mean()
        sample_var = float(np.var(r, ddof=1))
        if sample_var <= 0:
            raise ValueError("returns have zero variance")
        scale = 1.0 / math.sqrt(sample_var)  # fit on unit-variance data for stability
        z = r * scale

        def unpack(theta: FloatArray) -> tuple[float, float, float]:
            # alpha, beta in (0,1) with alpha + beta < 1 via a softmax-style transform
            e = np.exp(theta[1:])
            denom = 1.0 + e.sum()
            alpha, beta = float(e[0] / denom), float(e[1] / denom)
            omega = math.exp(float(theta[0]))
            return omega, alpha, beta

        def nll(theta: FloatArray) -> float:
            omega, alpha, beta = unpack(theta)
            var = cls._filter(z, omega, alpha, beta)[:-1]
            if not np.isfinite(var).all() or (var <= 0).any():
                return 1e12
            return float(0.5 * np.sum(np.log(var) + z * z / var))

        start = np.array([math.log(0.05), math.log(0.10 / 0.05), math.log(0.85 / 0.05)])
        result = minimize(
            nll,
            start,
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8},
        )
        omega, alpha, beta = unpack(result.x)
        omega /= scale * scale  # back to the original return scale
        last = float(cls._filter(r, omega, alpha, beta)[-1])
        return cls(omega=omega, alpha=alpha, beta=beta, last_variance=last)

    def conditional_variance(self, returns: npt.ArrayLike) -> FloatArray:
        """One-step-ahead variance after each return (same alignment as ``ewma_variance``)."""

        r = _returns(returns, 2)
        return self._filter(r - r.mean(), self.omega, self.alpha, self.beta)[1:]

    def forecast(self, horizon: int) -> FloatArray:
        """Variance for each of the next ``horizon`` periods (mean-reverts to the long run)."""

        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        steps = np.arange(horizon)
        long_run = self.long_run_variance
        path = long_run + self.persistence**steps * (self.last_variance - long_run)
        return np.asarray(path, dtype=np.float64)


def horizon_volatility(variances: npt.ArrayLike) -> float:
    """Volatility of the summed return over the forecast horizon (sqrt of summed variance)."""

    v = np.asarray(variances, dtype=np.float64)
    if v.ndim != 1 or v.size == 0 or (v < 0).any():
        raise ValueError("variances must be a non-negative 1D array")
    return float(math.sqrt(float(v.sum())))
