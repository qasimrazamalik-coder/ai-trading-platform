"""Configurable research promotion gate for anti-overfitting diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from quant_os_core.right_fit.stability import relative_performance_degradation


class RightFitDecision(StrEnum):
    RESEARCH_PASS = "research_pass"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class RightFitPolicy:
    max_pbo: float = 0.40
    min_deflated_sharpe_probability: float = 0.95
    min_parameter_stability: float = 0.60
    max_train_test_degradation: float = 0.35
    min_bootstrap_positive_probability: float = 0.70

    def __post_init__(self) -> None:
        probabilities = (
            self.max_pbo,
            self.min_deflated_sharpe_probability,
            self.min_parameter_stability,
            self.min_bootstrap_positive_probability,
        )
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("probability-like policy thresholds must be between 0 and 1")
        if not np.isfinite(self.max_train_test_degradation):
            raise ValueError("max_train_test_degradation must be finite")
        if self.max_train_test_degradation < 0.0:
            raise ValueError("max_train_test_degradation must be non-negative")


@dataclass(frozen=True, slots=True)
class RightFitReport:
    decision: RightFitDecision
    reasons: tuple[str, ...]
    train_test_degradation: float


def evaluate_right_fit(
    *,
    train_score: float,
    test_score: float,
    pbo: float,
    deflated_sharpe_probability: float,
    parameter_stability: float,
    bootstrap_positive_probability: float,
    leakage_detected: bool,
    policy: RightFitPolicy | None = None,
) -> RightFitReport:
    """Apply a conservative research gate; this does not authorize live trading."""
    active = policy or RightFitPolicy()
    metrics = {
        "pbo": pbo,
        "deflated_sharpe_probability": deflated_sharpe_probability,
        "parameter_stability": parameter_stability,
        "bootstrap_positive_probability": bootstrap_positive_probability,
    }
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in metrics.values()):
        raise ValueError("right-fit probability/stability inputs must be between 0 and 1")

    degradation = relative_performance_degradation(train_score, test_score)
    hard_failures: list[str] = []
    review_flags: list[str] = []

    if leakage_detected:
        hard_failures.append("temporal/data leakage detected")
    if test_score <= 0.0:
        hard_failures.append("out-of-sample score is not positive")
    if pbo > active.max_pbo:
        hard_failures.append("probability of backtest overfitting exceeds policy")
    if deflated_sharpe_probability < active.min_deflated_sharpe_probability:
        hard_failures.append("deflated Sharpe probability is below policy")

    if parameter_stability < active.min_parameter_stability:
        review_flags.append("parameter surface is insufficiently stable")
    if degradation > active.max_train_test_degradation:
        review_flags.append("train-to-test degradation exceeds policy")
    if bootstrap_positive_probability < active.min_bootstrap_positive_probability:
        review_flags.append("bootstrap robustness is below policy")

    if hard_failures:
        return RightFitReport(
            decision=RightFitDecision.REJECT,
            reasons=tuple(hard_failures + review_flags),
            train_test_degradation=degradation,
        )
    if review_flags:
        return RightFitReport(
            decision=RightFitDecision.REVIEW,
            reasons=tuple(review_flags),
            train_test_degradation=degradation,
        )
    return RightFitReport(
        decision=RightFitDecision.RESEARCH_PASS,
        reasons=("all configured right-fit research gates passed",),
        train_test_degradation=degradation,
    )
