"""Multiple-testing-aware Sharpe diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import e

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import kurtosis, norm, skew

from quant_os_core.quant._validation import as_float_array

_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    probability: float
    observed_sharpe: float
    benchmark_sharpe: float
    trial_count: int


def _periodic_sharpe(returns: ArrayLike) -> tuple[np.ndarray, float]:
    values = as_float_array(returns, name="returns", min_size=3)
    volatility = float(np.std(values, ddof=1))
    if volatility <= np.finfo(np.float64).eps:
        raise ValueError("Sharpe diagnostics require non-zero return volatility")
    return values, float(np.mean(values) / volatility)


def probabilistic_sharpe_ratio(
    returns: ArrayLike,
    *,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probability that the observed periodic Sharpe exceeds a benchmark.

    The finite-sample adjustment accounts for skewness and Pearson kurtosis.
    Both observed and benchmark Sharpe ratios use the return series' native
    periodicity; no annualization is applied inside this statistic.
    """
    values, observed = _periodic_sharpe(returns)
    if not np.isfinite(benchmark_sharpe):
        raise ValueError("benchmark_sharpe must be finite")
    sample_skew = float(skew(values, bias=False))
    sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    denominator_sq = (
        1.0
        - sample_skew * observed
        + ((sample_kurtosis - 1.0) / 4.0) * observed**2
    )
    if denominator_sq <= 0.0 or not np.isfinite(denominator_sq):
        raise ValueError("probabilistic Sharpe denominator is not positive and finite")
    z_score = (
        (observed - float(benchmark_sharpe))
        * np.sqrt(values.size - 1.0)
        / np.sqrt(denominator_sq)
    )
    return float(norm.cdf(z_score))


def expected_maximum_sharpe(trial_sharpes: ArrayLike) -> float:
    """Approximate the expected best Sharpe created by multiple trials."""
    values = as_float_array(trial_sharpes, name="trial_sharpes", min_size=2)
    count = values.size
    dispersion = float(np.std(values, ddof=1))
    if dispersion <= np.finfo(np.float64).eps:
        return 0.0
    first = norm.ppf(1.0 - 1.0 / count)
    second = norm.ppf(1.0 - 1.0 / (count * e))
    expected = dispersion * (
        (1.0 - _EULER_MASCHERONI) * first + _EULER_MASCHERONI * second
    )
    return float(max(0.0, expected))


def deflated_sharpe_ratio(
    returns: ArrayLike,
    *,
    trial_sharpes: ArrayLike,
) -> DeflatedSharpeResult:
    """Deflate an observed Sharpe for non-normality and strategy selection trials."""
    _, observed = _periodic_sharpe(returns)
    trials = as_float_array(trial_sharpes, name="trial_sharpes", min_size=2)
    benchmark = expected_maximum_sharpe(trials)
    probability = probabilistic_sharpe_ratio(returns, benchmark_sharpe=benchmark)
    return DeflatedSharpeResult(
        probability=probability,
        observed_sharpe=observed,
        benchmark_sharpe=benchmark,
        trial_count=int(trials.size),
    )


def sidak_adjusted_pvalue(p_value: float, *, n_tests: int) -> float:
    """Sidak family-wise adjustment for independent hypothesis tests."""
    parsed = float(p_value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError("p_value must be between 0 and 1")
    tests = int(n_tests)
    if tests != n_tests or tests <= 0:
        raise ValueError("n_tests must be a positive integer")
    return float(1.0 - (1.0 - parsed) ** tests)
