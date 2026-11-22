"""Triple-barrier labels and a walk-forward meta-label filter (Lopez de Prado, AFML ch. 3).

A *primary* rule (for example the ensemble v3 trend/momentum signal) proposes entries. The
meta-model learns, from features at entry time, which of those entries reached the profit
barrier before the stop or time barrier. Out of sample, its probability decides whether to take
the trade and feeds the trade plan's confidence. Training is strictly walk-forward with a gap
of at least the holding period, so labels never overlap the test window.

Any estimator with ``fit(X, y)`` and ``predict_proba(X)`` works (LightGBM, sklearn, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from quant_os_core.right_fit.splits import walk_forward_splits

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ProbabilisticClassifier(Protocol):
    def fit(self, x: Any, y: Any) -> Any: ...

    def predict_proba(self, x: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class BarrierSpec:
    profit_mult: float = 2.0  # target = entry * (1 + profit_mult * vol)
    stop_mult: float = 1.0  # stop = entry * (1 - stop_mult * vol)
    max_holding: int = 20  # bars

    def __post_init__(self) -> None:
        if self.profit_mult <= 0 or self.stop_mult <= 0 or self.max_holding < 1:
            raise ValueError("barrier multipliers and holding period must be positive")


def rolling_volatility(close: FloatArray, window: int = 20) -> FloatArray:
    returns = np.diff(np.log(close), prepend=np.nan)
    out = np.full(close.shape, np.nan)
    for i in range(window, close.size):
        out[i] = float(np.std(returns[i - window + 1 : i + 1], ddof=1))
    return out


def triple_barrier_labels(
    close: FloatArray, entries: IntArray, vol: FloatArray, spec: BarrierSpec
) -> tuple[IntArray, IntArray]:
    """Return (label, exit_index) per entry: 1 if the profit barrier is touched first."""

    labels = np.zeros(entries.size, dtype=np.int64)
    exits = np.zeros(entries.size, dtype=np.int64)
    for k, i in enumerate(entries):
        if not np.isfinite(vol[i]) or vol[i] <= 0:
            raise ValueError("volatility must be positive and finite at every entry")
        upper = close[i] * (1 + spec.profit_mult * vol[i])
        lower = close[i] * (1 - spec.stop_mult * vol[i])
        end = min(close.size - 1, i + spec.max_holding)
        exits[k] = end
        for j in range(i + 1, end + 1):
            if close[j] >= upper:
                labels[k], exits[k] = 1, j
                break
            if close[j] <= lower:
                exits[k] = j
                break
    return labels, exits


def basic_features(close: FloatArray, vol: FloatArray, entries: IntArray) -> FloatArray:
    """Momentum (5/20/60), distance from 200-bar average and volatility at entry."""

    def mom(i: int, n: int) -> float:
        return float(close[i] / close[i - n] - 1) if i >= n else 0.0

    rows = []
    for i in entries:
        sma = float(np.mean(close[max(0, i - 199) : i + 1]))
        rows.append([mom(i, 5), mom(i, 20), mom(i, 60), float(close[i] / sma - 1), float(vol[i])])
    return np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MetaLabelResult:
    probabilities: FloatArray  # out-of-sample probability per entry (nan where not tested)
    labels: IntArray
    tested: NDArray[np.bool_]

    def precision_at(self, threshold: float) -> float:
        take = self.tested & (self.probabilities >= threshold)
        return float(self.labels[take].mean()) if take.any() else float("nan")

    @property
    def base_rate(self) -> float:
        return float(self.labels[self.tested].mean()) if self.tested.any() else float("nan")


def walk_forward_meta_label(
    features: FloatArray,
    labels: IntArray,
    make_model: type[ProbabilisticClassifier] | Any,
    *,
    train_size: int,
    test_size: int,
    gap: int,
) -> MetaLabelResult:
    """Fit on each expanding window, predict the next block; ``gap`` must cover the holding."""

    if features.shape[0] != labels.size:
        raise ValueError("features and labels must align")
    probs = np.full(labels.size, np.nan)
    tested = np.zeros(labels.size, dtype=bool)
    for split in walk_forward_splits(
        labels.size, train_size=train_size, test_size=test_size, gap=gap
    ):
        y_train = labels[split.train]
        if np.unique(y_train).size < 2:
            continue
        model = make_model()
        model.fit(features[split.train], y_train)
        probs[split.test] = np.asarray(model.predict_proba(features[split.test]))[:, 1]
        tested[split.test] = True
    return MetaLabelResult(probs, labels, tested)
