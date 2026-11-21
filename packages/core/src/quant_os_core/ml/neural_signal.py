"""Neural confidence layer: learns which of the primary signal's trades reach their target.

1. Replay the primary signal over history; every plan it would have made becomes an event.
2. Label each event the way the exit manager would close it: 1 if the target is hit before
   the stop within the holding limit, else 0 (stop first, same-bar ties count as a loss).
3. Features at the entry bar only (``ml.features``), never after it.
4. Walk-forward: train on past events whose exits finished before the test window starts
   (purged), predict the next window, repeat. Network + logistic baseline, averaged.
5. Calibrate the out-of-sample scores (Platt) so a 0.62 means "about 62% of such trades won".
6. Gates: AUC and Brier against the base rate, skill on most folds, a Deflated Sharpe of the
   filtered trades against every threshold tried, and filtered trades must earn more per trade
   than taking every signal. PBO across thresholds is reported as a diagnostic only: the
   threshold variants are near-copies of each other, so their PBO sits near 0.5 by design.
   A model that fails any gate is not used.

Live, ``NeuralConfidenceSignal`` replaces the primary's fixed confidence with the calibrated
probability and skips trades below a floor - which is exactly what the edge-over-cost filter
and position sizing consume.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import numpy.typing as npt

from quant_os_core.agents.exits import Bar
from quant_os_core.agents.trade_plan import TradePlan
from quant_os_core.ml.features import FEATURE_NAMES, build_features
from quant_os_core.ml.metrics import brier_score, expected_calibration_error
from quant_os_core.ml.nn import ConfidenceModel, MLPClassifier, PlattScaler
from quant_os_core.right_fit.multiple_testing import deflated_sharpe_ratio
from quant_os_core.right_fit.pbo import probability_backtest_overfitting

FloatArray = npt.NDArray[np.float64]
Primary = Callable[[str, Sequence[Bar]], TradePlan | None]


def bars_to_arrays(bars: Sequence[Bar]) -> tuple[FloatArray, FloatArray, FloatArray]:
    return (
        np.array([float(b.high) for b in bars]),
        np.array([float(b.low) for b in bars]),
        np.array([float(b.close) for b in bars]),
    )


@dataclass(frozen=True, slots=True)
class EventDataset:
    """Events in time order. ``entry_index``/``exit_index`` are epoch seconds, so datasets from
    several symbols can be pooled and still purged correctly."""

    entry_index: npt.NDArray[np.int64]
    exit_index: npt.NDArray[np.int64]
    features: FloatArray
    labels: FloatArray
    returns: FloatArray  # realised return per event after round-trip cost

    def __len__(self) -> int:
        return int(self.labels.size)

    @staticmethod
    def pool(datasets: Sequence[EventDataset]) -> EventDataset:
        """Merge several symbols' events and sort by entry time."""

        if not datasets:
            raise ValueError("no datasets to pool")
        entry = np.concatenate([d.entry_index for d in datasets])
        order = np.argsort(entry, kind="mergesort")
        return EventDataset(
            entry[order],
            np.concatenate([d.exit_index for d in datasets])[order],
            np.concatenate([d.features for d in datasets])[order],
            np.concatenate([d.labels for d in datasets])[order],
            np.concatenate([d.returns for d in datasets])[order],
        )


def build_event_dataset(
    symbol: str,
    bars: Sequence[Bar],
    primary: Primary,
    *,
    max_holding: int = 20,
    round_trip_cost: float = 0.002,
    warmup: int = 200,
) -> EventDataset:
    high, low, close = bars_to_arrays(bars)
    features = build_features(high, low, close)
    entries, exits, rows, labels, rets = [], [], [], [], []
    t = warmup
    while t < len(bars) - 1:
        if not np.isfinite(features[t]).all():
            t += 1
            continue
        plan = primary(symbol, bars[: t + 1])
        if plan is None:
            t += 1
            continue
        entry, stop, target = (
            float(plan.entry_price),
            float(plan.stop_loss),
            float(plan.profit_target),
        )
        last = min(len(bars) - 1, t + max_holding)
        outcome, exit_at, exit_price = None, last, close[last]
        for j in range(t + 1, last + 1):
            if low[j] <= stop:  # stop checked first: ties count against the trade
                outcome, exit_at, exit_price = 0, j, stop
                break
            if high[j] >= target:
                outcome, exit_at, exit_price = 1, j, target
                break
        if outcome is None:
            if t + max_holding > len(bars) - 1:
                break  # not enough future bars to know the outcome: stop, don't guess
            outcome = 0  # timed out without reaching the target
        entries.append(int(bars[t].time.timestamp()))
        exits.append(int(bars[exit_at].time.timestamp()))
        rows.append(features[t])
        labels.append(float(outcome))
        rets.append(exit_price / entry - 1 - round_trip_cost)
        t += 1
    return EventDataset(
        np.asarray(entries, dtype=np.int64),
        np.asarray(exits, dtype=np.int64),
        np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_NAMES)),
        np.asarray(labels, dtype=np.float64),
        np.asarray(rets, dtype=np.float64),
    )


def auc(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Area under the ROC curve via ranks (0.5 = no skill)."""

    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size)
    ranks[order] = np.arange(1, s.size + 1)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _log_loss(y: FloatArray, p: FloatArray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass(slots=True)
class TrainingReport:
    events: int
    tested: int
    base_rate: float
    auc_model: float
    auc_network: float
    auc_baseline: float
    brier_model: float
    brier_base_rate: float
    ece_calibrated: float
    precision_at: dict[str, float]
    fold_scores: list[float]  # per fold: base-rate log loss minus model log loss (> 0 is skill)
    deflated_sharpe: float
    pbo: float  # diagnostic only (see module docstring)
    mean_return_all: float
    mean_return_filtered: float
    chosen_threshold: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


def train_confidence_model(
    data: EventDataset,
    *,
    n_folds: int = 5,
    min_train: int = 120,
    thresholds: Sequence[float] = (0.45, 0.5, 0.55, 0.6, 0.65),
    hidden: tuple[int, ...] = (32, 16),
    seed: int = 7,
) -> tuple[ConfidenceModel | None, TrainingReport]:
    n = len(data)
    if n < min_train + n_folds * 20:
        raise ValueError(f"need at least {min_train + n_folds * 20} events, have {n}")
    test_starts = np.linspace(min_train, n, n_folds + 1).astype(int)
    raw = np.full(n, np.nan)
    net_only = np.full(n, np.nan)
    base_only = np.full(n, np.nan)
    rate = np.full(n, np.nan)
    fold_scores: list[float] = []
    for k in range(n_folds):
        lo, hi = int(test_starts[k]), int(test_starts[k + 1])
        first_entry = data.entry_index[lo]
        train = np.flatnonzero(data.exit_index[:lo] < first_entry)  # purge overlapping labels
        y_train = data.labels[train]
        if train.size < 50 or np.unique(y_train).size < 2:
            continue
        net = MLPClassifier(hidden=hidden, seed=seed + k).fit(data.features[train], y_train)
        base = MLPClassifier(hidden=(), seed=seed + k).fit(data.features[train], y_train)
        test = slice(lo, hi)
        net_only[test] = net.predict_proba(data.features[test])
        base_only[test] = base.predict_proba(data.features[test])
        raw[test] = 0.5 * net_only[test] + 0.5 * base_only[test]
        rate[test] = float(y_train.mean())
        y_test = data.labels[test]
        fold_scores.append(_log_loss(y_test, rate[test]) - _log_loss(y_test, raw[test]))

    tested = np.isfinite(raw)
    reasons: list[str] = []
    if tested.sum() < 40:
        raise ValueError("too few out-of-sample predictions; collect more history")
    y, s = data.labels[tested], raw[tested]
    calibrator = PlattScaler().fit(s, y)
    p = calibrator.transform(s)
    precision = {}
    for th in thresholds:
        take = p >= th
        precision[f"{th:.2f}"] = float(y[take].mean()) if take.any() else float("nan")

    oos_returns = data.returns[tested]
    columns = [oos_returns] + [np.where(p >= th, oos_returns, 0.0) for th in thresholds]
    matrix = np.column_stack(columns)
    sharpes = []
    for col in matrix.T:
        sd = float(np.std(col, ddof=1))
        sharpes.append(float(np.mean(col) / sd) if sd > 0 else -math.inf)
    best = int(np.argmax(sharpes[1:])) + 1  # best threshold variant
    chosen = float(thresholds[best - 1])
    try:
        dsr = deflated_sharpe_ratio(
            matrix[:, best], trial_sharpes=[x for x in sharpes if math.isfinite(x)]
        ).probability
    except ValueError:
        dsr = 0.0
    try:
        pbo = probability_backtest_overfitting(matrix, n_slices=8).probability
    except ValueError:
        pbo = 1.0

    report = TrainingReport(
        events=n,
        tested=int(tested.sum()),
        base_rate=float(y.mean()),
        auc_model=auc(y, s),
        auc_network=auc(y, net_only[tested]),
        auc_baseline=auc(y, base_only[tested]),
        brier_model=brier_score(y, p),
        brier_base_rate=brier_score(y, rate[tested]),
        ece_calibrated=expected_calibration_error(y, p),
        precision_at=precision,
        fold_scores=fold_scores,
        deflated_sharpe=float(dsr),
        pbo=float(pbo),
        mean_return_all=float(np.mean(oos_returns)),
        mean_return_filtered=float(np.mean(oos_returns[p >= chosen]))
        if (p >= chosen).any()
        else float("nan"),
        chosen_threshold=chosen,
        passed=False,
    )
    if report.auc_model < 0.55:
        reasons.append(f"AUC {report.auc_model:.3f} < 0.55: no reliable skill")
    if report.brier_model >= report.brier_base_rate:
        reasons.append("does not beat the base rate on Brier score")
    if sum(1 for f in fold_scores if f > 0) < math.ceil(0.6 * len(fold_scores)):
        reasons.append("beats the base rate on too few folds")
    if report.deflated_sharpe < 0.95:
        reasons.append(f"deflated Sharpe {report.deflated_sharpe:.2f} < 0.95")
    if not report.mean_return_filtered > report.mean_return_all:
        reasons.append("filtered trades do not earn more per trade than taking every signal")
    report.passed = not reasons
    report.reasons = reasons
    if not report.passed:
        return None, report

    final_net = MLPClassifier(hidden=hidden, seed=seed).fit(data.features, data.labels)
    final_base = MLPClassifier(hidden=(), seed=seed).fit(data.features, data.labels)
    model = ConfidenceModel(
        network=final_net,
        baseline=final_base,
        calibrator=calibrator,
        feature_names=FEATURE_NAMES,
        metadata={
            "auc": report.auc_model,
            "base_rate": report.base_rate,
            "threshold": chosen,
            "deflated_sharpe": report.deflated_sharpe,
            "pbo": report.pbo,
            "events": n,
        },
    )
    return model, report


@dataclass(frozen=True, slots=True)
class NeuralConfidenceSignal:
    """Primary signal + calibrated neural probability = the plan's confidence."""

    primary: Primary
    model: ConfidenceModel
    min_probability: float = 0.55

    def probability(self, bars: Sequence[Bar]) -> float | None:
        high, low, close = bars_to_arrays(bars)
        row = build_features(high, low, close)[-1]
        if not np.isfinite(row).all():
            return None
        return float(self.model.probability(row[None, :])[0])

    def __call__(self, symbol: str, bars: Sequence[Bar]) -> TradePlan | None:
        plan = self.primary(symbol, bars)
        if plan is None:
            return None
        prob = self.probability(bars)
        if prob is None or prob < self.min_probability:
            return None
        return plan.model_copy(
            update={
                "confidence": Decimal(str(round(prob, 4))),
                "justification": f"{plan.justification}; neural P(target first)={prob:.2f}",
            }
        )
