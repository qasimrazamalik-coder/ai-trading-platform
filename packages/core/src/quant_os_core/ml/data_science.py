"""Data science checks before any model sees the data: quality, distribution, feature value.

* ``data_quality``: non-positive or missing prices, stale runs, gaps, OHLC violations and
  outlier returns by robust z-score (median/MAD, so one crash does not hide the next).
* ``return_profile``: mean, volatility, skew, excess kurtosis, percentiles, worst/best days,
  autocorrelation of returns and of squared returns (volatility clustering).
* ``feature_diagnostics``: each feature's rank correlation with the label (information
  coefficient), its stability across time halves, and highly redundant feature pairs.
* ``render_report``: all of it as Markdown for the docs folder or the chatbot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class QualityReport:
    rows: int
    bad_prices: int
    ohlc_violations: int
    stale_runs: int  # runs of 5+ identical closes
    gaps: int  # time gaps more than 3x the median spacing
    outliers: tuple[int, ...]  # row indices with |robust z| > 8
    ok: bool


def data_quality(
    times: Sequence[datetime],
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
) -> QualityReport:
    h, lo, c = (np.asarray(v, dtype=np.float64) for v in (high, low, close))
    n = c.size
    if not (len(times) == n == h.size == lo.size):
        raise ValueError("times and prices must have the same length")
    bad = int(np.sum(~np.isfinite(c) | (c <= 0)))
    violations = int(np.sum((h < lo) | (c > h + 1e-12) | (c < lo - 1e-12)))
    stale, run = 0, 1
    for t in range(1, n):
        run = run + 1 if c[t] == c[t - 1] else 1
        if run == 5:
            stale += 1
    gaps = 0
    if n > 2:
        spacing = np.array([(times[t] - times[t - 1]).total_seconds() for t in range(1, n)])
        med = float(np.median(spacing))
        if med > 0:
            # weekends/holidays: only flag gaps far beyond a normal long weekend
            limit = max(3 * med, timedelta(days=4).total_seconds()) if med >= 86_000 else 3 * med
            gaps = int(np.sum(spacing > limit))
    outliers: tuple[int, ...] = ()
    if n > 30 and bad == 0:
        r = np.diff(np.log(c))
        mad = float(np.median(np.abs(r - np.median(r)))) * 1.4826
        if mad > 0:
            z = np.abs(r - np.median(r)) / mad
            outliers = tuple(int(i) + 1 for i in np.flatnonzero(z > 8))
    ok = bad == 0 and violations == 0 and gaps == 0 and stale == 0
    return QualityReport(n, bad, violations, stale, gaps, outliers, ok)


@dataclass(frozen=True, slots=True)
class ReturnProfile:
    periods: int
    mean: float
    volatility: float
    skew: float
    excess_kurtosis: float
    percentiles: dict[str, float]
    worst: float
    best: float
    autocorr_1: float
    autocorr_sq_1: float  # > 0.1 usually means volatility clusters (GARCH territory)


def _autocorr(x: FloatArray, lag: int = 1) -> float:
    if x.size <= lag + 2:
        return 0.0
    a, b = x[:-lag] - x[:-lag].mean(), x[lag:] - x[lag:].mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / denom) if denom > 0 else 0.0


def return_profile(close: npt.ArrayLike) -> ReturnProfile:
    c = np.asarray(close, dtype=np.float64)
    if c.ndim != 1 or c.size < 30 or (c <= 0).any():
        raise ValueError("need at least 30 positive closes")
    r = np.diff(np.log(c))
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    z = (r - mu) / sd if sd > 0 else np.zeros_like(r)
    return ReturnProfile(
        periods=int(r.size),
        mean=mu,
        volatility=sd,
        skew=float(np.mean(z**3)),
        excess_kurtosis=float(np.mean(z**4) - 3),
        percentiles={f"p{q}": float(np.percentile(r, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
        worst=float(r.min()),
        best=float(r.max()),
        autocorr_1=_autocorr(r),
        autocorr_sq_1=_autocorr(r * r),
    )


def _rank(x: FloatArray) -> FloatArray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    ranks[order] = np.arange(x.size)
    return ranks


def _spearman(x: FloatArray, y: FloatArray) -> float:
    if x.size < 5 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


@dataclass(frozen=True, slots=True)
class FeatureDiagnostic:
    name: str
    ic: float  # Spearman correlation with the label over all rows
    ic_first_half: float
    ic_second_half: float

    @property
    def stable(self) -> bool:
        """Same sign in both halves and not trivially small."""

        return self.ic_first_half * self.ic_second_half > 0 and abs(self.ic) >= 0.02


@dataclass(frozen=True, slots=True)
class FeatureReport:
    diagnostics: tuple[FeatureDiagnostic, ...]
    redundant_pairs: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)


def feature_diagnostics(
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    names: Sequence[str],
    redundancy: float = 0.9,
) -> FeatureReport:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[1] != len(names):
        raise ValueError("features [n, k], labels [n] and k names are required")
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[keep], y[keep]
    half = x.shape[0] // 2
    diags = tuple(
        FeatureDiagnostic(
            name,
            _spearman(x[:, j], y),
            _spearman(x[:half, j], y[:half]),
            _spearman(x[half:, j], y[half:]),
        )
        for j, name in enumerate(names)
    )
    pairs = []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            rho = _spearman(x[:, i], x[:, j])
            if abs(rho) >= redundancy:
                pairs.append((names[i], names[j], round(rho, 3)))
    ordered = tuple(sorted(diags, key=lambda d: abs(d.ic), reverse=True))
    return FeatureReport(ordered, tuple(pairs))


def render_report(
    title: str,
    quality: QualityReport,
    profile: ReturnProfile,
    features: FeatureReport | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Data quality",
        "",
        f"- Rows: {quality.rows}; status: {'OK' if quality.ok else 'NEEDS ATTENTION'}",
        f"- Bad prices: {quality.bad_prices}; OHLC violations: {quality.ohlc_violations}",
        f"- Stale runs (5+ equal closes): {quality.stale_runs}; time gaps: {quality.gaps}",
        f"- Outlier returns (|robust z| > 8): {len(quality.outliers)}",
        "",
        "## Return profile (log returns per bar)",
        "",
        f"- Mean {profile.mean:.5f}, volatility {profile.volatility:.5f}",
        f"- Skew {profile.skew:.2f}, excess kurtosis {profile.excess_kurtosis:.2f}"
        + (" (fat tails)" if profile.excess_kurtosis > 1 else ""),
        f"- Worst {profile.worst:.2%}, best {profile.best:.2%}",
        f"- Autocorrelation: returns {profile.autocorr_1:.3f}, squared returns "
        f"{profile.autocorr_sq_1:.3f}"
        + (" (volatility clusters: use GARCH/EWMA)" if profile.autocorr_sq_1 > 0.1 else ""),
        "- Percentiles: " + ", ".join(f"{k} {v:.2%}" for k, v in profile.percentiles.items()),
    ]
    if features is not None:
        lines += [
            "",
            "## Feature value (rank correlation with the label)",
            "",
            "| Feature | IC | 1st half | 2nd half | Stable |",
            "|---|---|---|---|---|",
        ]
        for d in features.diagnostics:
            lines.append(
                f"| {d.name} | {d.ic:+.3f} | {d.ic_first_half:+.3f} | {d.ic_second_half:+.3f} "
                f"| {'yes' if d.stable else 'no'} |"
            )
        if features.redundant_pairs:
            lines += [
                "",
                "Redundant pairs (|rho| >= 0.9): "
                + "; ".join(f"{a} ~ {b} ({r})" for a, b, r in features.redundant_pairs),
            ]
    return "\n".join(lines) + "\n"
