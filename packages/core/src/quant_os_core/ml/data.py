"""Leakage-aware supervised datasets for financial ML."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SupervisedDataset:
    features: FloatArray
    targets: FloatArray
    feature_names: tuple[str, ...]
    timestamps: npt.NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.features, dtype=np.float64)
        y = np.asarray(self.targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size or y.size == 0:
            raise ValueError(
                "features must be 2D and targets must be a matching non-empty 1D array"
            )
        if x.shape[1] != len(self.feature_names):
            raise ValueError("feature_names must match feature columns")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("dataset must contain only finite values")
        if self.timestamps is not None:
            ts = np.asarray(self.timestamps, dtype=np.int64)
            if ts.ndim != 1 or ts.size != y.size or np.any(np.diff(ts) <= 0):
                raise ValueError("timestamps must be strictly increasing and match target length")
            object.__setattr__(self, "timestamps", ts.copy())
        object.__setattr__(self, "features", x.copy())
        object.__setattr__(self, "targets", y.copy())


@dataclass(frozen=True, slots=True)
class Standardizer:
    mean: FloatArray
    scale: FloatArray

    @classmethod
    def fit(cls, x: npt.ArrayLike) -> Standardizer:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError("x must be a finite 2D array with at least two rows")
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=1)
        scale = np.where(scale <= np.finfo(np.float64).eps, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, x: npt.ArrayLike) -> FloatArray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("x column count must match fitted standardizer")
        return (values - self.mean) / self.scale


def lagged_return_dataset(
    prices: npt.ArrayLike,
    *,
    lags: int = 20,
    horizon: int = 1,
    timestamps: npt.ArrayLike | None = None,
) -> SupervisedDataset:
    """Build lagged log-return features using only information available at prediction time."""
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim != 1 or p.size < lags + horizon + 2 or np.any(p <= 0) or not np.isfinite(p).all():
        raise ValueError("prices must be positive finite 1D values with enough history")
    if lags < 1 or horizon < 1:
        raise ValueError("lags and horizon must be positive")
    returns = np.diff(np.log(p))
    rows: list[FloatArray] = []
    targets: list[float] = []
    kept_indices: list[int] = []
    # At return index t, all features end at t; target starts after t and spans horizon returns.
    for t in range(lags - 1, returns.size - horizon):
        rows.append(returns[t - lags + 1 : t + 1])
        targets.append(float(np.sum(returns[t + 1 : t + 1 + horizon])))
        kept_indices.append(t + 1)  # price index corresponding to feature observation time
    ts_out = None
    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=np.int64)
        if ts.ndim != 1 or ts.size != p.size or np.any(np.diff(ts) <= 0):
            raise ValueError("timestamps must strictly increase and match prices")
        ts_out = ts[np.asarray(kept_indices, dtype=np.int64)]
    return SupervisedDataset(
        features=np.vstack(rows),
        targets=np.asarray(targets, dtype=np.float64),
        feature_names=tuple(f"log_return_lag_{offset}" for offset in range(lags - 1, -1, -1)),
        timestamps=ts_out,
    )


def chronological_split(
    dataset: SupervisedDataset,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[SupervisedDataset, SupervisedDataset, SupervisedDataset]:
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("fractions must be inside valid ranges")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must be below 1")
    n = dataset.targets.size
    i = int(np.floor(n * train_fraction))
    j = int(np.floor(n * (train_fraction + validation_fraction)))
    if i < 1 or j <= i or j >= n:
        raise ValueError("dataset is too small for requested split")

    def take(start: int, end: int) -> SupervisedDataset:
        ts = None if dataset.timestamps is None else dataset.timestamps[start:end]
        return SupervisedDataset(
            dataset.features[start:end], dataset.targets[start:end], dataset.feature_names, ts
        )

    return take(0, i), take(i, j), take(j, n)
