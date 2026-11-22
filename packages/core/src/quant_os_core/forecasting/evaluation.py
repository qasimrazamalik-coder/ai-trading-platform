"""Rolling-origin forecast evaluation: every forecast uses only data before its target.

A model earns trust only by beating the naive forecast out of sample; the Diebold-Mariano test
says whether the difference is larger than luck.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
OneStep = Callable[[FloatArray], float]


@dataclass(frozen=True, slots=True)
class ForecastScore:
    forecasts: FloatArray
    actuals: FloatArray
    rmse: float
    mae: float
    directional_accuracy: float
    naive_rmse: float
    skill: float  # 1 - rmse / naive_rmse; > 0 beats the naive forecast
    dm_statistic: float
    dm_p_value: float  # two-sided; small means the difference is unlikely to be luck


def _normal_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def diebold_mariano(errors_a: npt.ArrayLike, errors_b: npt.ArrayLike) -> tuple[float, float]:
    """DM test on squared-error loss for one-step forecasts. Negative stat favours ``a``."""

    a = np.asarray(errors_a, dtype=np.float64)
    b = np.asarray(errors_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 10:
        raise ValueError("need two matching 1D error series of at least 10 values")
    d = a * a - b * b
    sd = float(np.std(d, ddof=1))
    if sd <= 0:
        return 0.0, 1.0
    stat = float(np.mean(d) / (sd / math.sqrt(d.size)))
    return stat, _normal_two_sided(stat)


def rolling_origin(
    series: npt.ArrayLike,
    forecaster: OneStep,
    *,
    min_train: int,
    naive: OneStep | None = None,
    step: int = 1,
) -> ForecastScore:
    """Forecast ``series[t]`` from ``series[:t]`` for every t >= min_train (every ``step``).

    ``naive`` defaults to predicting zero, the right benchmark for returns. For price levels
    pass ``lambda h: h[-1]`` (random walk).
    """

    y = np.asarray(series, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all() or min_train < 2 or y.size < min_train + 10:
        raise ValueError("series too short for rolling-origin evaluation")
    bench = naive or (lambda _h: 0.0)
    preds, bases, actual = [], [], []
    for t in range(min_train, y.size, step):
        history = y[:t]
        preds.append(float(forecaster(history)))
        bases.append(float(bench(history)))
        actual.append(float(y[t]))
    f, n, a = np.asarray(preds), np.asarray(bases), np.asarray(actual)
    err, base_err = a - f, a - n
    rmse = float(np.sqrt(np.mean(err**2)))
    naive_rmse = float(np.sqrt(np.mean(base_err**2)))
    stat, p = diebold_mariano(err, base_err)
    moved = a != 0
    direction = float(np.mean(np.sign(f[moved]) == np.sign(a[moved]))) if moved.any() else 0.5
    return ForecastScore(
        forecasts=f,
        actuals=a,
        rmse=rmse,
        mae=float(np.mean(np.abs(err))),
        directional_accuracy=direction,
        naive_rmse=naive_rmse,
        skill=1 - rmse / naive_rmse if naive_rmse > 0 else 0.0,
        dm_statistic=stat,
        dm_p_value=p,
    )


def qlike(realized_variance: npt.ArrayLike, forecast_variance: npt.ArrayLike) -> float:
    """QLIKE loss for variance forecasts (robust to noisy realized variance); lower is better."""

    rv = np.asarray(realized_variance, dtype=np.float64)
    fv = np.asarray(forecast_variance, dtype=np.float64)
    if rv.shape != fv.shape or (fv <= 0).any() or (rv < 0).any():
        raise ValueError("variance arrays must match and forecasts must be positive")
    ratio = rv / fv
    return float(np.mean(ratio - np.log(np.where(ratio > 0, ratio, 1e-300)) - 1))
