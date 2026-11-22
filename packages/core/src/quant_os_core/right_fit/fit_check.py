"""Underfit / overfit diagnostics with numbers: learning curves, the generalization gap,
a complexity sweep under the one-standard-error rule, and a plain verdict.

Everything here is chronological. The validation slice is always the most recent data and
training rows always come before it, so no diagnostic can peek at the future.

* **Learning curve**: train and validation error as the training window grows. Validation
  error that keeps falling says more data would help; a large, stable gap says the model
  memorises; both errors stuck at the baseline says it learned nothing.
* **Generalization gap**: how much worse validation error is than training error.
* **Complexity sweep**: score several model sizes on the same walk-forward folds and pick the
  simplest one whose mean error is within one standard error of the best (the 1-SE rule,
  Hastie, Tibshirani and Friedman, *Elements of Statistical Learning*, section 7.10).
* **Verdict**: ``underfit`` (no better than the baseline even in sample), ``overfit``
  (validation clearly worse than training) or ``right_fit``, with the numbers behind it.

A verdict is a diagnostic label, not proof of edge; the deflated Sharpe, PBO and paper period
gates still apply.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Predictor = Callable[[FloatArray], FloatArray]
Trainer = Callable[[FloatArray, FloatArray], Predictor]


class FitVerdict(StrEnum):
    UNDERFIT = "underfit"
    RIGHT_FIT = "right_fit"
    OVERFIT = "overfit"


def mse(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if a.shape != p.shape or a.size == 0:
        raise ValueError("actual and predicted must be non-empty and the same shape")
    return float(np.mean((a - p) ** 2))


def _xy(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    features = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if features.ndim < 2 or target.ndim != 1 or features.shape[0] != target.shape[0]:
        raise ValueError("x must be [n, ...] and y must have n values")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("x and y must be finite")
    return features, target


# ---------------------------------------------------------------- generalization gap


@dataclass(frozen=True, slots=True)
class GeneralizationGap:
    train_error: float
    validation_error: float
    absolute: float  # validation - train
    relative: float  # (validation - train) / train; 0.25 means validation is 25% worse


def generalization_gap(train_error: float, validation_error: float) -> GeneralizationGap:
    if not (math.isfinite(train_error) and math.isfinite(validation_error)):
        raise ValueError("errors must be finite")
    if train_error < 0 or validation_error < 0:
        raise ValueError("errors must be non-negative")
    absolute = validation_error - train_error
    relative = absolute / train_error if train_error > 0 else (math.inf if absolute > 0 else 0.0)
    return GeneralizationGap(train_error, validation_error, absolute, relative)


# ---------------------------------------------------------------- learning curve


@dataclass(frozen=True, slots=True)
class LearningCurvePoint:
    n_train: int
    train_error: float
    validation_error: float


@dataclass(frozen=True, slots=True)
class LearningCurve:
    points: tuple[LearningCurvePoint, ...]
    validation_size: int

    @property
    def final_gap(self) -> GeneralizationGap:
        last = self.points[-1]
        return generalization_gap(last.train_error, last.validation_error)

    @property
    def validation_still_improving(self) -> bool:
        """True when the largest window beats the previous one by more than 1%."""

        if len(self.points) < 2:
            return False
        prev, last = self.points[-2].validation_error, self.points[-1].validation_error
        return last < prev * 0.99


def learning_curve(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    trainer: Trainer,
    *,
    fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    validation_fraction: float = 0.2,
    min_train: int = 20,
) -> LearningCurve:
    """Train on growing windows that end where validation starts; score both sides.

    The validation slice is the last ``validation_fraction`` of rows and never changes, so
    the curve isolates the effect of more training data. Each window is the most recent
    ``fraction`` of the pre-validation rows.
    """

    features, target = _xy(x, y)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not fractions or any(not 0 < f <= 1 for f in fractions):
        raise ValueError("fractions must be in (0, 1]")
    n_val = max(1, round(features.shape[0] * validation_fraction))
    n_pre = features.shape[0] - n_val
    if n_pre < min_train:
        raise ValueError("not enough rows before the validation slice")
    x_val, y_val = features[n_pre:], target[n_pre:]
    points: list[LearningCurvePoint] = []
    for fraction in sorted(set(fractions)):
        n = max(min_train, round(n_pre * fraction))
        x_tr, y_tr = features[n_pre - n : n_pre], target[n_pre - n : n_pre]
        predict = trainer(x_tr, y_tr)
        points.append(
            LearningCurvePoint(
                n_train=n,
                train_error=mse(y_tr, predict(x_tr)),
                validation_error=mse(y_val, predict(x_val)),
            )
        )
    return LearningCurve(tuple(points), n_val)


# ---------------------------------------------------------------- complexity sweep


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    complexity: float  # anything monotone in capacity: parameters, hidden units, lags
    fold_errors: tuple[float, ...]

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.fold_errors))

    @property
    def standard_error(self) -> float:
        k = len(self.fold_errors)
        return float(np.std(self.fold_errors, ddof=1) / math.sqrt(k)) if k > 1 else 0.0


@dataclass(frozen=True, slots=True)
class ComplexitySweep:
    candidates: tuple[Candidate, ...]
    best: Candidate  # lowest mean error
    chosen: Candidate  # simplest within one standard error of the best
    threshold: float  # best.mean_error + best.standard_error


def one_standard_error_rule(candidates: Sequence[Candidate]) -> ComplexitySweep:
    """Pick the simplest candidate whose mean error is within one SE of the best mean."""

    if not candidates:
        raise ValueError("need at least one candidate")
    folds = {len(c.fold_errors) for c in candidates}
    if len(folds) != 1 or 0 in folds:
        raise ValueError("every candidate must be scored on the same non-zero number of folds")
    for c in candidates:
        if not all(math.isfinite(e) and e >= 0 for e in c.fold_errors):
            raise ValueError(f"{c.name}: fold errors must be finite and non-negative")
    best = min(candidates, key=lambda c: c.mean_error)
    threshold = best.mean_error + best.standard_error
    eligible = [c for c in candidates if c.mean_error <= threshold]
    chosen = min(eligible, key=lambda c: (c.complexity, c.mean_error))
    return ComplexitySweep(tuple(candidates), best, chosen, threshold)


def walk_forward_errors(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    trainer: Trainer,
    *,
    folds: int = 4,
    min_train_fraction: float = 0.5,
) -> tuple[float, ...]:
    """Validation MSE on ``folds`` consecutive blocks, each trained on everything before it."""

    features, target = _xy(x, y)
    if folds < 2 or not 0 < min_train_fraction < 1:
        raise ValueError("need folds >= 2 and min_train_fraction in (0, 1)")
    start = int(features.shape[0] * min_train_fraction)
    block = (features.shape[0] - start) // folds
    if start < 10 or block < 2:
        raise ValueError("series too short for the requested folds")
    errors = []
    for k in range(folds):
        lo = start + k * block
        hi = features.shape[0] if k == folds - 1 else lo + block
        predict = trainer(features[:lo], target[:lo])
        errors.append(mse(target[lo:hi], predict(features[lo:hi])))
    return tuple(errors)


def complexity_sweep(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    trainers: Sequence[tuple[str, float, Trainer]],
    *,
    folds: int = 4,
    min_train_fraction: float = 0.5,
) -> ComplexitySweep:
    """Score each ``(name, complexity, trainer)`` on the same folds, then apply the 1-SE rule."""

    candidates = [
        Candidate(
            name,
            complexity,
            walk_forward_errors(x, y, trainer, folds=folds, min_train_fraction=min_train_fraction),
        )
        for name, complexity, trainer in trainers
    ]
    return one_standard_error_rule(candidates)


# ---------------------------------------------------------------- verdict


@dataclass(frozen=True, slots=True)
class FitReport:
    verdict: FitVerdict
    train_error: float
    validation_error: float
    baseline_train_error: float
    baseline_validation_error: float
    gap: GeneralizationGap
    train_skill: float  # 1 - train_error / baseline_train_error
    validation_skill: float  # 1 - validation_error / baseline_validation_error
    reason: str
    curve: LearningCurve | None = None

    def summary(self) -> str:
        text = (
            f"{self.verdict.value}: train {self.train_error:.4g} vs baseline "
            f"{self.baseline_train_error:.4g} (skill {self.train_skill:+.1%}), validation "
            f"{self.validation_error:.4g} (skill {self.validation_skill:+.1%}), gap "
            f"{self.gap.relative:+.1%}; {self.reason}"
        )
        if self.curve is not None and self.curve.validation_still_improving:
            text += "; validation error still falling with more data"
        return text


def _skill(error: float, baseline: float) -> float:
    return 1.0 - error / baseline if baseline > 0 else 0.0


def fit_verdict(
    *,
    train_error: float,
    validation_error: float,
    baseline_train_error: float,
    baseline_validation_error: float,
    max_relative_gap: float = 0.25,
    min_train_skill: float = 0.01,
    curve: LearningCurve | None = None,
) -> FitReport:
    """Label a fitted model from its errors and a naive baseline's errors on the same rows.

    * ``overfit``: validation error is more than ``max_relative_gap`` worse than training
      error *and* the model kept less than half of its in-sample skill over the baseline.
      Requiring both stops a calm-then-volatile split (every model's error rises, the
      baseline's too) from being read as memorisation.
    * ``underfit``: training error is not at least ``min_train_skill`` better than the
      baseline: the model failed to learn even the data it saw.
    * ``right_fit``: it learned something in sample and kept it out of sample.
    """

    values = (train_error, validation_error, baseline_train_error, baseline_validation_error)
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("errors must be finite and non-negative")
    if max_relative_gap < 0 or min_train_skill < 0:
        raise ValueError("thresholds must be non-negative")
    gap = generalization_gap(train_error, validation_error)
    train_skill = _skill(train_error, baseline_train_error)
    val_skill = _skill(validation_error, baseline_validation_error)
    if gap.relative > max_relative_gap and val_skill < train_skill / 2:
        verdict = FitVerdict.OVERFIT
        reason = (
            f"validation error {gap.relative:.0%} above training (limit "
            f"{max_relative_gap:.0%}) and skill fell from {train_skill:+.1%} to "
            f"{val_skill:+.1%}"
        )
    elif train_skill < min_train_skill:
        verdict = FitVerdict.UNDERFIT
        reason = f"in-sample skill {train_skill:+.1%} below {min_train_skill:.0%}: learned little"
    else:
        verdict = FitVerdict.RIGHT_FIT
        reason = "learned in sample and the gap stays within limits"
        if val_skill <= 0:
            reason += ", but it does not beat the baseline out of sample"
    return FitReport(
        verdict=verdict,
        train_error=train_error,
        validation_error=validation_error,
        baseline_train_error=baseline_train_error,
        baseline_validation_error=baseline_validation_error,
        gap=gap,
        train_skill=train_skill,
        validation_skill=val_skill,
        reason=reason,
        curve=curve,
    )


def fit_check(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    trainer: Trainer,
    *,
    baseline: Trainer,
    fractions: Sequence[float] = (0.25, 0.5, 1.0),
    validation_fraction: float = 0.2,
    max_relative_gap: float = 0.25,
    min_train_skill: float = 0.01,
) -> FitReport:
    """Learning curve plus verdict for ``trainer`` against ``baseline`` on the same split."""

    curve = learning_curve(
        x, y, trainer, fractions=fractions, validation_fraction=validation_fraction
    )
    base = learning_curve(x, y, baseline, fractions=(1.0,), validation_fraction=validation_fraction)
    last, ref = curve.points[-1], base.points[-1]
    return fit_verdict(
        train_error=last.train_error,
        validation_error=last.validation_error,
        baseline_train_error=ref.train_error,
        baseline_validation_error=ref.validation_error,
        max_relative_gap=max_relative_gap,
        min_train_skill=min_train_skill,
        curve=curve,
    )


def mean_trainer(x: FloatArray, y: FloatArray) -> Predictor:
    """Baseline that predicts the training mean."""

    level = float(np.mean(y))
    return lambda features: np.full(features.shape[0], level)


def zero_trainer(x: FloatArray, y: FloatArray) -> Predictor:
    """Baseline that predicts zero (the right benchmark for returns)."""

    return lambda features: np.zeros(features.shape[0])
