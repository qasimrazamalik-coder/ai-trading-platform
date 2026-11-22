"""Regime-aware strategy selection among vetted strategies, behind a right-fit guard.

Code decides; a language model may only write the explanation. The selection works like this:

1. Keep only candidates whose marketplace provider is ``vetted``.
2. Walk forward through the history (``right_fit.walk_forward_splits``). In every fold each
   candidate is scored in-sample and out-of-sample on the periods that share the current
   regime label (all periods of the fold when the regime is too rare to measure).
3. Right-fit guard, per candidate:
   * overfit: in-sample to out-of-sample Sharpe degradation above ``max_degradation``;
   * no skill: out-of-sample excess return over the simple baseline is not significant once
     deflated for the number of candidates tried (deflated Sharpe probability below
     ``min_skill_probability``);
   * selection overfit: probability of backtest overfitting across candidates above
     ``max_pbo`` rejects the whole selection.
4. The highest out-of-sample Sharpe that survives the guard wins. If none survives, the
   report selects nothing and the system keeps the baseline.

The report is advisory input to the paper loop; trades still go through the entry pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import numpy.typing as npt

from quant_os_core.agents.marketplace import Marketplace
from quant_os_core.regimes.base import RegimeSnapshot
from quant_os_core.right_fit.multiple_testing import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from quant_os_core.right_fit.pbo import probability_backtest_overfitting
from quant_os_core.right_fit.splits import walk_forward_splits
from quant_os_core.right_fit.stability import relative_performance_degradation

FloatArray = npt.NDArray[np.float64]
Explainer = Callable[["SelectionReport"], str]

MAX_EXPLANATION_CHARS = 2_000


def regime_key(snapshot: RegimeSnapshot) -> str:
    """Default regime label: trend and volatility, e.g. ``bull_trend|low_volatility``."""

    return f"{snapshot.trend.label}|{snapshot.volatility.label}"


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    strategy_id: str
    returns: FloatArray  # per-period paper/backtest returns, aligned with the baseline

    def __post_init__(self) -> None:
        values = np.asarray(self.returns, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"{self.strategy_id}: returns must be a finite 1-D array")
        object.__setattr__(self, "returns", values)


@dataclass(frozen=True, slots=True)
class SelectorPolicy:
    train_size: int = 120
    test_size: int = 40
    min_regime_periods: int = 10
    max_degradation: float = 0.5
    min_skill_probability: float = 0.95
    max_pbo: float = 0.5
    pbo_slices: int = 8

    def __post_init__(self) -> None:
        if self.train_size < 3 or self.test_size < 3:
            raise ValueError("train_size and test_size must be at least 3")
        if self.min_regime_periods < 3:
            raise ValueError("min_regime_periods must be at least 3")
        if self.max_degradation < 0:
            raise ValueError("max_degradation cannot be negative")
        for name in ("min_skill_probability", "max_pbo"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    strategy_id: str
    in_sample_sharpe: float | None
    out_of_sample_sharpe: float | None
    degradation: float | None
    skill_probability: float | None
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectionReport:
    regime: str
    selected: str | None
    candidates: tuple[CandidateEvaluation, ...]
    pbo: float | None
    folds: int
    generated_at: datetime
    explanation: str = ""
    explanation_source: str = "rules"


def _sharpe(values: FloatArray) -> float | None:
    if values.size < 3:
        return None
    std = float(np.std(values, ddof=1))
    if std <= np.finfo(np.float64).eps:
        return None
    return float(np.mean(values)) / std


def _regime_rows(
    indices: npt.NDArray[np.int64], labels: Sequence[str], regime: str, minimum: int
) -> npt.NDArray[np.int64]:
    matched = np.asarray([i for i in indices if labels[int(i)] == regime], dtype=np.int64)
    return matched if matched.size >= minimum else indices


def _rules_explanation(report: SelectionReport) -> str:
    if report.selected is None:
        return (
            f"Regime {report.regime}: no vetted strategy passed the right-fit guard; "
            "keeping the baseline."
        )
    chosen = next(c for c in report.candidates if c.strategy_id == report.selected)
    return (
        f"Regime {report.regime}: selected {report.selected} "
        f"(out-of-sample Sharpe {chosen.out_of_sample_sharpe:.3f}, "
        f"skill probability {chosen.skill_probability:.3f}) over {report.folds} folds."
    )


@dataclass(slots=True)
class StrategySelector:
    marketplace: Marketplace
    policy: SelectorPolicy = field(default_factory=SelectorPolicy)
    explainer: Explainer | None = None  # may only write prose; never changes the choice

    def select(
        self,
        candidates: Sequence[StrategyCandidate],
        *,
        baseline_returns: npt.ArrayLike,
        regime_labels: Sequence[str],
        current: RegimeSnapshot | str,
        now: datetime,
    ) -> SelectionReport:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        baseline = np.asarray(baseline_returns, dtype=np.float64)
        if baseline.ndim != 1 or not np.isfinite(baseline).all():
            raise ValueError("baseline_returns must be a finite 1-D array")
        if len(regime_labels) != baseline.size:
            raise ValueError("regime_labels must align with baseline_returns")
        for candidate in candidates:
            if candidate.returns.size != baseline.size:
                raise ValueError(f"{candidate.strategy_id}: returns must align with baseline")
        regime = current if isinstance(current, str) else regime_key(current)
        p = self.policy
        splits = walk_forward_splits(baseline.size, train_size=p.train_size, test_size=p.test_size)

        evaluations: list[CandidateEvaluation] = []
        eligible: list[StrategyCandidate] = []
        for candidate in candidates:
            if self.marketplace.is_vetted(candidate.strategy_id):
                eligible.append(candidate)
            else:
                evaluations.append(
                    CandidateEvaluation(
                        candidate.strategy_id,
                        None,
                        None,
                        None,
                        None,
                        False,
                        ("not a vetted marketplace provider",),
                    )
                )

        oos_rows = np.concatenate(
            [_regime_rows(s.test, regime_labels, regime, p.min_regime_periods) for s in splits]
        )
        scored: list[tuple[StrategyCandidate, float | None, float | None, FloatArray]] = []
        for candidate in eligible:
            is_scores = [
                _sharpe(
                    candidate.returns[
                        _regime_rows(s.train, regime_labels, regime, p.min_regime_periods)
                    ]
                )
                for s in splits
            ]
            valid_is = [v for v in is_scores if v is not None]
            in_sample = float(np.mean(valid_is)) if valid_is else None
            out_of_sample = _sharpe(candidate.returns[oos_rows])
            excess = candidate.returns[oos_rows] - baseline[oos_rows]
            scored.append((candidate, in_sample, out_of_sample, excess))

        trial_sharpes = [s for s in (_sharpe(item[3]) for item in scored) if s is not None]
        pbo: float | None = None
        if len(eligible) >= 2:
            matrix = np.column_stack([c.returns for c in eligible])
            try:
                pbo = probability_backtest_overfitting(matrix, n_slices=p.pbo_slices).probability
            except ValueError:
                pbo = None

        accepted: list[tuple[float, str]] = []
        for candidate, in_sample, out_of_sample, excess in scored:
            reasons: list[str] = []
            degradation: float | None = None
            skill: float | None = None
            if in_sample is None or out_of_sample is None:
                reasons.append("Sharpe not computable (too few or flat returns)")
            else:
                degradation = relative_performance_degradation(in_sample, out_of_sample)
                if degradation > p.max_degradation:
                    reasons.append(
                        f"overfit: in-sample {in_sample:.3f} vs out-of-sample "
                        f"{out_of_sample:.3f} degrades {degradation:.0%}"
                    )
            if _sharpe(excess) is None:
                reasons.append("no skill: excess return over baseline is flat or too short")
            else:
                if len(trial_sharpes) >= 2:
                    skill = deflated_sharpe_ratio(excess, trial_sharpes=trial_sharpes).probability
                else:
                    skill = probabilistic_sharpe_ratio(excess)
                if skill < p.min_skill_probability:
                    reasons.append(
                        f"no skill: beats baseline with probability {skill:.3f} "
                        f"< {p.min_skill_probability}"
                    )
            if pbo is not None and pbo > p.max_pbo:
                reasons.append(f"selection overfit: PBO {pbo:.2f} > {p.max_pbo}")
            ok = not reasons
            if ok and out_of_sample is not None:
                accepted.append((out_of_sample, candidate.strategy_id))
            evaluations.append(
                CandidateEvaluation(
                    candidate.strategy_id,
                    in_sample,
                    out_of_sample,
                    degradation,
                    skill,
                    ok,
                    tuple(reasons) or ("passes the right-fit guard",),
                )
            )

        selected = max(accepted)[1] if accepted else None
        report = SelectionReport(
            regime=regime,
            selected=selected,
            candidates=tuple(sorted(evaluations, key=lambda e: e.strategy_id)),
            pbo=pbo,
            folds=len(splits),
            generated_at=now,
        )
        return self._explain(report)

    def _explain(self, report: SelectionReport) -> SelectionReport:
        fallback = _rules_explanation(report)
        if self.explainer is None:
            return _with_explanation(report, fallback, "rules")
        try:
            text = self.explainer(report)
        except Exception:  # an offline or broken model counts as "no opinion"
            return _with_explanation(report, fallback, "rules")
        if not isinstance(text, str) or not text.strip():
            return _with_explanation(report, fallback, "rules")
        return _with_explanation(report, text.strip()[:MAX_EXPLANATION_CHARS], "llm")


def _with_explanation(report: SelectionReport, text: str, source: str) -> SelectionReport:
    return SelectionReport(
        regime=report.regime,
        selected=report.selected,
        candidates=report.candidates,
        pbo=report.pbo,
        folds=report.folds,
        generated_at=report.generated_at,
        explanation=text,
        explanation_source=source,
    )
