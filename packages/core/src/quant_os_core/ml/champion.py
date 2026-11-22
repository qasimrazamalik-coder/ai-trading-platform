"""Champion/challenger gate for weekly retraining.

A retrained model replaces the current one only if, on the same walk-forward test folds it
never trained on, it wins clearly and still passes the overfitting gates. Otherwise the
champion stays and the challenger is logged as rejected, with the reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quant_os_core.ops.state import StateStore


@dataclass(frozen=True, slots=True)
class GatePolicy:
    min_fold_win_share: float = 0.6  # challenger better on at least 60% of folds
    min_mean_improvement: float = 0.0  # mean score must improve (higher is better)
    min_deflated_sharpe: float = 0.95
    max_pbo: float = 0.5
    min_folds: int = 5


@dataclass(frozen=True, slots=True)
class GateResult:
    promote: bool
    reason: str
    fold_win_share: float
    mean_improvement: float


def compare(
    champion_scores: Sequence[float],
    challenger_scores: Sequence[float],
    *,
    deflated_sharpe: float,
    pbo: float,
    policy: GatePolicy | None = None,
) -> GateResult:
    """Scores are per out-of-sample fold, same folds for both, higher is better."""

    p = policy or GatePolicy()
    if len(champion_scores) != len(challenger_scores):
        raise ValueError("both models must be scored on the same folds")
    n = len(challenger_scores)
    if n < p.min_folds:
        return GateResult(False, f"only {n} folds; need {p.min_folds}", 0.0, 0.0)
    wins = sum(1 for a, b in zip(champion_scores, challenger_scores, strict=True) if b > a)
    share = wins / n
    improvement = (sum(challenger_scores) - sum(champion_scores)) / n
    if deflated_sharpe < p.min_deflated_sharpe:
        return GateResult(
            False, f"deflated Sharpe {deflated_sharpe:.2f} below gate", share, improvement
        )
    if pbo >= p.max_pbo:
        return GateResult(False, f"PBO {pbo:.2f} at or above gate", share, improvement)
    if share < p.min_fold_win_share:
        return GateResult(False, f"wins only {share:.0%} of folds", share, improvement)
    if improvement <= p.min_mean_improvement:
        return GateResult(False, "no mean improvement", share, improvement)
    return GateResult(
        True, f"wins {share:.0%} of folds, +{improvement:.4f} mean", share, improvement
    )


def run_gate(
    store: StateStore,
    name: str,
    version: str,
    champion_scores: Sequence[float] | None,
    challenger_scores: Sequence[float],
    *,
    deflated_sharpe: float,
    pbo: float,
    policy: GatePolicy | None = None,
) -> GateResult:
    """Decide, record in the model registry, and return the result.

    With no champion yet, the challenger only needs to pass the overfitting gates.
    """

    if champion_scores is None:
        p = policy or GatePolicy()
        ok = deflated_sharpe >= p.min_deflated_sharpe and pbo < p.max_pbo
        reason = "first model passes overfitting gates" if ok else "fails overfitting gates"
        result = GateResult(ok, reason, 1.0 if ok else 0.0, 0.0)
    else:
        result = compare(
            champion_scores,
            challenger_scores,
            deflated_sharpe=deflated_sharpe,
            pbo=pbo,
            policy=policy,
        )
    metrics = {
        "deflated_sharpe": deflated_sharpe,
        "pbo": pbo,
        "fold_win_share": result.fold_win_share,
        "mean_score": sum(challenger_scores) / max(1, len(challenger_scores)),
        **{f"fold_{i:02d}": float(v) for i, v in enumerate(challenger_scores)},
    }
    store.register_model(
        name, version, "champion" if result.promote else "rejected", metrics, result.reason
    )
    store.log("info", "model", f"{name} {version}: {result.reason}")
    return result
