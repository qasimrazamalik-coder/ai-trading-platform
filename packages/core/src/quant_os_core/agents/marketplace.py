"""Strategy marketplace: signal providers vetted only by their own paper track record.

Providers are internal strategies, analyst-board sources and Discord channels. Nobody pays
or gets paid here and no real money moves: a provider earns ``vetted`` status only from the
paper outcomes of its own signals, graded by this system.

Vetting rules (all must hold, thresholds in :class:`VettingPolicy`):

* at least ``min_signals`` graded signals over at least ``min_track_days`` days;
* deflated Sharpe probability (Bailey and Lopez de Prado), deflated by the number of
  providers competing in the marketplace, at or above ``min_deflated_sharpe``;
* hit rate at or above ``min_hit_rate`` and max drawdown at or below ``max_drawdown``;
* halal compliance rate (share of signals that passed the Shariah gate) at or above
  ``min_halal_compliance``.

Status moves: candidate -> vetted when every rule holds; vetted -> suspended when any rule
breaks on re-vet; suspended -> vetted only when every rule holds again. A manual suspension
is sticky until :meth:`Marketplace.reinstate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

import numpy as np

from quant_os_core.right_fit.multiple_testing import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


class ProviderKind(StrEnum):
    INTERNAL_STRATEGY = "internal_strategy"
    ANALYST_BOARD = "analyst_board"
    DISCORD_CHANNEL = "discord_channel"


class ProviderStatus(StrEnum):
    CANDIDATE = "candidate"
    VETTED = "vetted"
    SUSPENDED = "suspended"


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True)
class SignalOutcome:
    """One provider signal as seen by the paper pipeline.

    ``shariah_passed`` is ``None`` when the pipeline stopped before the Shariah gate (the
    gate never saw it), ``False`` when the gate or intake blocked it as non-compliant.
    ``return_pct`` is the realised paper return as a fraction, set once the trade is closed.
    """

    provider_id: str
    signal_id: str
    at: datetime
    shariah_passed: bool | None
    approved: bool
    return_pct: Decimal | None = None

    def __post_init__(self) -> None:
        _aware(self.at, "at")
        if isinstance(self.return_pct, float):
            raise ValueError("return_pct must be Decimal")


@dataclass(frozen=True, slots=True)
class VettingPolicy:
    min_signals: int = 30
    min_track_days: int = 90
    min_deflated_sharpe: float = 0.95
    min_hit_rate: Decimal = Decimal("0.50")
    max_drawdown: Decimal = Decimal("0.20")
    min_halal_compliance: Decimal = Decimal("0.95")
    revet_interval: timedelta = timedelta(days=7)

    def __post_init__(self) -> None:
        if self.min_signals < 3:
            raise ValueError("min_signals must be at least 3 (Sharpe needs a sample)")
        if self.min_track_days < 0:
            raise ValueError("min_track_days cannot be negative")
        if not 0.0 <= self.min_deflated_sharpe <= 1.0:
            raise ValueError("min_deflated_sharpe is a probability in [0, 1]")
        for name in ("min_hit_rate", "max_drawdown", "min_halal_compliance"):
            value = getattr(self, name)
            if not Decimal("0") <= value <= Decimal("1"):
                raise ValueError(f"{name} must be in [0, 1]")
        if self.revet_interval <= timedelta(0):
            raise ValueError("revet_interval must be positive")


@dataclass(frozen=True, slots=True)
class ProviderMetrics:
    signals: int
    graded: int
    track_days: int
    deflated_sharpe: float | None
    observed_sharpe: float | None
    hit_rate: Decimal | None
    max_drawdown: Decimal | None
    halal_compliance: Decimal | None


@dataclass(frozen=True, slots=True)
class ProviderRecord:
    provider_id: str
    kind: ProviderKind
    display_name: str
    status: ProviderStatus = ProviderStatus.CANDIDATE
    reasons: tuple[str, ...] = ()
    last_vetted: datetime | None = None
    manual_hold: bool = False


@dataclass(frozen=True, slots=True)
class VettingResult:
    provider_id: str
    previous: ProviderStatus
    status: ProviderStatus
    reasons: tuple[str, ...]
    metrics: ProviderMetrics
    vetted_at: datetime

    @property
    def passed(self) -> bool:
        return self.status is ProviderStatus.VETTED


def _periodic_sharpe(returns: list[float]) -> float | None:
    if len(returns) < 3:
        return None
    std = float(np.std(returns, ddof=1))
    if std <= np.finfo(np.float64).eps:
        return None
    return float(np.mean(returns)) / std


def _max_drawdown(returns: list[Decimal]) -> Decimal:
    equity = Decimal("1")
    peak = Decimal("1")
    worst = Decimal("0")
    for value in returns:
        equity *= Decimal("1") + value
        peak = max(peak, equity)
        worst = max(worst, (peak - equity) / peak if peak > 0 else Decimal("1"))
    return worst


@dataclass(slots=True)
class Marketplace:
    policy: VettingPolicy = field(default_factory=VettingPolicy)
    _providers: dict[str, ProviderRecord] = field(default_factory=dict)
    _outcomes: dict[str, dict[str, SignalOutcome]] = field(default_factory=dict)

    # -- providers ---------------------------------------------------------------------

    def add_provider(
        self, provider_id: str, kind: ProviderKind, display_name: str = ""
    ) -> ProviderRecord:
        if not provider_id.strip():
            raise ValueError("provider_id cannot be blank")
        if provider_id in self._providers:
            raise ValueError(f"provider already listed: {provider_id}")
        record = ProviderRecord(provider_id, kind, display_name or provider_id)
        self._providers[provider_id] = record
        self._outcomes[provider_id] = {}
        return record

    def ensure_provider(
        self, provider_id: str, kind: ProviderKind, display_name: str = ""
    ) -> ProviderRecord:
        existing = self._providers.get(provider_id)
        return existing or self.add_provider(provider_id, kind, display_name)

    def provider(self, provider_id: str) -> ProviderRecord:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"unknown provider: {provider_id}") from exc

    def providers(self, status: ProviderStatus | None = None) -> tuple[ProviderRecord, ...]:
        items = sorted(self._providers.values(), key=lambda r: r.provider_id)
        return tuple(r for r in items if status is None or r.status is status)

    def vetted(self, kind: ProviderKind | None = None) -> tuple[ProviderRecord, ...]:
        return tuple(
            r for r in self.providers(ProviderStatus.VETTED) if kind is None or r.kind is kind
        )

    def is_vetted(self, provider_id: str) -> bool:
        record = self._providers.get(provider_id)
        return record is not None and record.status is ProviderStatus.VETTED

    def suspend(self, provider_id: str, reason: str, now: datetime) -> ProviderRecord:
        record = replace(
            self.provider(provider_id),
            status=ProviderStatus.SUSPENDED,
            reasons=(f"manual: {reason}",),
            last_vetted=_aware(now, "now"),
            manual_hold=True,
        )
        self._providers[provider_id] = record
        return record

    def reinstate(self, provider_id: str) -> ProviderRecord:
        """Lift a manual hold; the provider still has to pass the next vet."""

        record = replace(self.provider(provider_id), manual_hold=False)
        self._providers[provider_id] = record
        return record

    # -- track record ------------------------------------------------------------------

    def record(self, outcome: SignalOutcome) -> bool:
        """Store an outcome; returns False for a duplicate signal id (first one wins)."""

        book = self._outcomes.get(outcome.provider_id)
        if book is None:
            raise KeyError(f"unknown provider: {outcome.provider_id}")
        if outcome.signal_id in book:
            return False
        book[outcome.signal_id] = outcome
        return True

    def grade(self, provider_id: str, signal_id: str, return_pct: Decimal) -> SignalOutcome:
        """Attach the realised paper return of an approved signal."""

        if not isinstance(return_pct, Decimal):
            raise ValueError("return_pct must be Decimal")
        if return_pct <= Decimal("-1"):
            raise ValueError("an unlevered spot trade cannot lose more than 100%")
        book = self._outcomes.get(provider_id)
        if book is None or signal_id not in book:
            raise KeyError(f"unknown signal {signal_id} for {provider_id}")
        current = book[signal_id]
        if not current.approved:
            raise ValueError("only approved (paper-traded) signals can be graded")
        updated = replace(current, return_pct=return_pct)
        book[signal_id] = updated
        return updated

    def outcomes(self, provider_id: str) -> tuple[SignalOutcome, ...]:
        book = self._outcomes.get(provider_id)
        if book is None:
            raise KeyError(f"unknown provider: {provider_id}")
        return tuple(sorted(book.values(), key=lambda o: (o.at, o.signal_id)))

    # -- vetting -----------------------------------------------------------------------

    def _graded_returns(self, provider_id: str) -> list[Decimal]:
        return [o.return_pct for o in self.outcomes(provider_id) if o.return_pct is not None]

    def _trial_sharpes(self) -> list[float]:
        sharpes = []
        for provider_id in self._providers:
            value = _periodic_sharpe([float(r) for r in self._graded_returns(provider_id)])
            if value is not None:
                sharpes.append(value)
        return sharpes

    def metrics(self, provider_id: str, now: datetime) -> ProviderMetrics:
        _aware(now, "now")
        history = self.outcomes(provider_id)
        returns = self._graded_returns(provider_id)
        floats = [float(r) for r in returns]
        track_days = (now - history[0].at).days if history else 0
        judged = [o.shariah_passed for o in history if o.shariah_passed is not None]
        compliance = Decimal(sum(judged)) / Decimal(len(judged)) if judged else None
        observed = _periodic_sharpe(floats)
        deflated: float | None = None
        if observed is not None:
            trials = self._trial_sharpes()
            if len(trials) >= 2:
                deflated = deflated_sharpe_ratio(floats, trial_sharpes=trials).probability
            else:  # a single competitor: nothing to deflate for, plain PSR vs zero
                deflated = probabilistic_sharpe_ratio(floats)
        hit_rate = (
            Decimal(sum(1 for r in returns if r > 0)) / Decimal(len(returns)) if returns else None
        )
        return ProviderMetrics(
            signals=len(history),
            graded=len(returns),
            track_days=track_days,
            deflated_sharpe=deflated,
            observed_sharpe=observed,
            hit_rate=hit_rate,
            max_drawdown=_max_drawdown(returns) if returns else None,
            halal_compliance=compliance,
        )

    def _failures(self, metrics: ProviderMetrics) -> list[str]:
        p = self.policy
        reasons: list[str] = []
        if metrics.graded < p.min_signals:
            reasons.append(f"only {metrics.graded} graded signals; need {p.min_signals}")
        if metrics.track_days < p.min_track_days:
            reasons.append(f"track record {metrics.track_days} days; need {p.min_track_days}")
        if metrics.deflated_sharpe is None:
            reasons.append("deflated Sharpe not computable (too few or flat returns)")
        elif metrics.deflated_sharpe < p.min_deflated_sharpe:
            reasons.append(
                f"deflated Sharpe {metrics.deflated_sharpe:.3f} below {p.min_deflated_sharpe}"
            )
        if metrics.hit_rate is None or metrics.hit_rate < p.min_hit_rate:
            reasons.append(f"hit rate {metrics.hit_rate} below {p.min_hit_rate}")
        if metrics.max_drawdown is None or metrics.max_drawdown > p.max_drawdown:
            reasons.append(f"max drawdown {metrics.max_drawdown} above {p.max_drawdown}")
        if metrics.halal_compliance is None or metrics.halal_compliance < p.min_halal_compliance:
            reasons.append(
                f"halal compliance {metrics.halal_compliance} below {p.min_halal_compliance}"
            )
        return reasons

    def vet(self, provider_id: str, now: datetime) -> VettingResult:
        record = self.provider(provider_id)
        metrics = self.metrics(provider_id, now)
        reasons = self._failures(metrics)
        if record.manual_hold:
            status = ProviderStatus.SUSPENDED
            reasons = [*record.reasons, *reasons]
        elif not reasons:
            status = ProviderStatus.VETTED
        elif record.status is ProviderStatus.CANDIDATE:
            status = ProviderStatus.CANDIDATE
        else:
            status = ProviderStatus.SUSPENDED
        shown = tuple(reasons) or ("meets every vetting rule",)
        self._providers[provider_id] = replace(
            record, status=status, reasons=shown, last_vetted=now
        )
        return VettingResult(provider_id, record.status, status, shown, metrics, now)

    def due_for_revet(self, now: datetime) -> tuple[str, ...]:
        _aware(now, "now")
        return tuple(
            r.provider_id
            for r in self.providers()
            if r.last_vetted is None or now - r.last_vetted >= self.policy.revet_interval
        )

    def revet_due(self, now: datetime) -> tuple[VettingResult, ...]:
        """Scheduled job: re-vet every provider whose last vet is older than the interval."""

        return tuple(self.vet(provider_id, now) for provider_id in self.due_for_revet(now))
