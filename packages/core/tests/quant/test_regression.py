import numpy as np
import pytest

from quant_os_core.quant.regression import (
    alpha,
    beta,
    engle_granger,
    information_ratio,
    mean_reversion_half_life,
    ordinary_least_squares,
    tracking_error,
)


def test_ols_recovers_line() -> None:
    x = np.arange(1.0, 11.0)
    y = 2.0 + 3.0 * x
    result = ordinary_least_squares(y, x)
    assert result.intercept == pytest.approx(2.0)
    assert result.slope == pytest.approx(3.0)
    assert result.r_squared == pytest.approx(1.0)
    np.testing.assert_allclose(result.residuals, 0.0, atol=1e-12)


def test_beta_and_alpha_known_relationship() -> None:
    benchmark = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    asset = 0.001 + 1.5 * benchmark
    assert beta(asset, benchmark) == pytest.approx(1.5)
    assert alpha(asset, benchmark) == pytest.approx(0.001)


def test_tracking_error_and_information_ratio() -> None:
    benchmark = [0.01, 0.00, -0.01, 0.02]
    asset = [0.012, 0.001, -0.009, 0.023]
    te = tracking_error(asset, benchmark, periods_per_year=12)
    ir = information_ratio(asset, benchmark, periods_per_year=12)
    assert te > 0
    assert np.isfinite(ir)


def test_engle_granger_recognizes_constructed_pair() -> None:
    rng = np.random.default_rng(123)
    x = np.cumsum(rng.normal(size=500)) + 100.0
    stationary_noise = rng.normal(scale=0.3, size=500)
    y = 5.0 + 1.7 * x + stationary_noise
    result = engle_granger(y, x)
    assert result.hedge_ratio == pytest.approx(1.7, rel=0.02)
    assert result.p_value < 0.05
    assert result.is_stationary_5pct


def test_half_life_for_mean_reverting_process() -> None:
    rng = np.random.default_rng(5)
    series = np.zeros(800)
    phi = 0.8
    for idx in range(1, series.size):
        series[idx] = phi * series[idx - 1] + rng.normal(scale=0.2)
    half_life = mean_reversion_half_life(series)
    theoretical = -np.log(2.0) / (phi - 1.0)
    assert half_life == pytest.approx(theoretical, rel=0.35)


def test_regression_risk_free_and_annualization_guards() -> None:
    benchmark = [-0.01, 0.00, 0.01]
    asset = [-0.02, 0.00, 0.02]
    with pytest.raises(ValueError):
        alpha(asset, benchmark, risk_free_period_return=float("nan"))
    with pytest.raises(ValueError):
        tracking_error(asset, benchmark, periods_per_year=float("inf"))
    with pytest.raises(ValueError):
        information_ratio(asset, benchmark, periods_per_year=0.0)
