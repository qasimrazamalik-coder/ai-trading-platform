"""Portfolio construction and allocation algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from quant_os_core.quant.portfolio import nearest_psd, portfolio_volatility

FloatArray = NDArray[np.float64]


def _vector(values: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _covariance(matrix: ArrayLike, n_assets: int) -> NDArray[np.float64]:
    cov = np.asarray(matrix, dtype=np.float64)
    if cov.shape != (n_assets, n_assets):
        raise ValueError("covariance shape must match number of assets")
    if not np.isfinite(cov).all():
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    return nearest_psd(cov, floor=1e-12)


def _normalize_long_only(weights: ArrayLike) -> FloatArray:
    w = _vector(weights, name="weights")
    if np.any(w < 0.0):
        raise ValueError("long-only weights cannot be negative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return w / total


def inverse_volatility_weights(covariance: ArrayLike) -> FloatArray:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.isfinite(cov).all() or np.any(np.diag(cov) <= 0.0):
        raise ValueError("asset variances must be finite and positive")
    cov = _covariance(cov, cov.shape[0])
    vol = np.sqrt(np.diag(cov))
    if np.any(vol <= 0.0):
        raise ValueError("asset volatility must be positive")
    return _normalize_long_only(1.0 / vol)


def minimum_variance_weights(
    covariance: ArrayLike,
    *,
    max_weight: float = 1.0,
) -> FloatArray:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    n = cov.shape[0]
    cov = _covariance(cov, n)
    if not np.isfinite(max_weight) or not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must be in (0, 1]")
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError("max_weight makes full investment infeasible")
    initial = np.full(n, 1.0 / n)
    result = minimize(
        lambda w: float(w @ cov @ w),
        initial,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"minimum-variance optimization failed: {result.message}")
    return _normalize_long_only(result.x)


def maximum_sharpe_weights(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    *,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> FloatArray:
    mu = _vector(expected_returns, name="expected_returns")
    cov = _covariance(covariance, mu.size)
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    if not np.isfinite(max_weight) or not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must be in (0, 1]")
    if max_weight * mu.size < 1.0 - 1e-12:
        raise ValueError("max_weight makes full investment infeasible")
    excess = mu - risk_free_rate

    def objective(w: FloatArray) -> float:
        vol = portfolio_volatility(w, cov)
        if vol <= 1e-15:
            return 1e12
        return -float(w @ excess) / vol

    result = minimize(
        objective,
        np.full(mu.size, 1.0 / mu.size),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * mu.size,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"maximum-Sharpe optimization failed: {result.message}")
    return _normalize_long_only(result.x)


def risk_parity_weights(
    covariance: ArrayLike,
    *,
    target_budgets: ArrayLike | None = None,
) -> FloatArray:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    n = cov.shape[0]
    cov = _covariance(cov, n)
    budgets = (
        np.full(n, 1.0 / n)
        if target_budgets is None
        else _normalize_long_only(target_budgets)
    )
    if budgets.size != n:
        raise ValueError("target_budgets must match number of assets")

    def objective(w: FloatArray) -> float:
        variance = float(w @ cov @ w)
        if variance <= 1e-18:
            return 1e12
        marginal = cov @ w
        contributions = w * marginal / variance
        return float(np.sum((contributions - budgets) ** 2))

    result = minimize(
        objective,
        inverse_volatility_weights(cov),
        method="SLSQP",
        bounds=[(1e-10, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        options={"ftol": 1e-14, "maxiter": 3000},
    )
    if not result.success:
        raise RuntimeError(f"risk-parity optimization failed: {result.message}")
    return _normalize_long_only(result.x)


def cvar_minimization_weights(
    scenario_returns: ArrayLike,
    *,
    confidence: float = 0.95,
    max_weight: float = 1.0,
) -> FloatArray:
    scenarios = np.asarray(scenario_returns, dtype=np.float64)
    if scenarios.ndim != 2 or scenarios.shape[0] < 2 or scenarios.shape[1] < 1:
        raise ValueError("scenario_returns must be observations x assets")
    if not np.isfinite(scenarios).all():
        raise ValueError("scenario_returns must contain only finite values")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    n = scenarios.shape[1]
    if not np.isfinite(max_weight) or not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must be in (0, 1]")
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError("max_weight makes full investment infeasible")

    def objective(w: FloatArray) -> float:
        losses = -(scenarios @ w)
        var = float(np.quantile(losses, confidence))
        tail = losses[losses >= var]
        return float(np.mean(tail)) if tail.size else var

    result = minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"CVaR optimization failed: {result.message}")
    return _normalize_long_only(result.x)


def black_litterman_posterior(
    prior_returns: ArrayLike,
    covariance: ArrayLike,
    pick_matrix: ArrayLike,
    view_returns: ArrayLike,
    *,
    tau: float = 0.05,
    view_uncertainty: ArrayLike | None = None,
) -> tuple[FloatArray, NDArray[np.float64]]:
    pi = _vector(prior_returns, name="prior_returns")
    cov = _covariance(covariance, pi.size)
    p = np.asarray(pick_matrix, dtype=np.float64)
    q = _vector(view_returns, name="view_returns")
    if p.ndim != 2 or p.shape != (q.size, pi.size):
        raise ValueError("pick_matrix must have shape views x assets")
    if not np.isfinite(p).all():
        raise ValueError("pick_matrix must contain only finite values")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    tau_cov = tau * cov
    if view_uncertainty is None:
        omega = np.diag(np.maximum(np.diag(p @ tau_cov @ p.T), 1e-12))
    else:
        omega = np.asarray(view_uncertainty, dtype=np.float64)
        if omega.shape != (q.size, q.size):
            raise ValueError("view_uncertainty must be square with one row per view")
        if not np.isfinite(omega).all():
            raise ValueError("view_uncertainty must contain only finite values")
    middle = np.linalg.pinv(p @ tau_cov @ p.T + omega)
    posterior_mean = pi + tau_cov @ p.T @ middle @ (q - p @ pi)
    posterior_cov = cov + tau_cov - tau_cov @ p.T @ middle @ p @ tau_cov
    return posterior_mean, nearest_psd(posterior_cov, floor=1e-12)


def _cluster_variance(cov: NDArray[np.float64], indices: list[int]) -> float:
    sub = cov[np.ix_(indices, indices)]
    ivp = inverse_volatility_weights(sub)
    return float(ivp @ sub @ ivp)


def hierarchical_risk_parity_weights(covariance: ArrayLike) -> FloatArray:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2:
        raise ValueError("covariance must be square with at least two assets")
    cov = _covariance(cov, cov.shape[0])
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    condensed = squareform(distance, checks=False)
    order = list(map(int, leaves_list(linkage(condensed, method="single"))))
    weights = np.ones(cov.shape[0], dtype=np.float64)
    clusters: list[list[int]] = [order]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left, right = cluster[:split], cluster[split:]
        left_var = _cluster_variance(cov, left)
        right_var = _cluster_variance(cov, right)
        denom = left_var + right_var
        alpha = 0.5 if denom <= 1e-18 else 1.0 - left_var / denom
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        clusters.extend([left, right])
    return _normalize_long_only(weights)


def hierarchical_equal_risk_contribution_weights(covariance: ArrayLike) -> FloatArray:
    """Hierarchical equal-risk allocation across recursively split correlation clusters."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2:
        raise ValueError("covariance must be square with at least two assets")
    cov = _covariance(cov, cov.shape[0])
    std = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(std, std), -1.0, 1.0)
    distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    condensed = squareform(distance, checks=False)
    order = list(map(int, leaves_list(linkage(condensed, method="single"))))
    weights = np.ones(cov.shape[0], dtype=np.float64)
    clusters: list[list[int]] = [order]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left, right = cluster[:split], cluster[split:]
        left_vol = np.sqrt(max(_cluster_variance(cov, left), 1e-18))
        right_vol = np.sqrt(max(_cluster_variance(cov, right), 1e-18))
        alpha = right_vol / (left_vol + right_vol)
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        clusters.extend([left, right])
    return _normalize_long_only(weights)


@dataclass(frozen=True, slots=True)
class AllocationSummary:
    weights: FloatArray
    expected_return: float
    volatility: float
    sharpe: float


def summarize_allocation(
    weights: ArrayLike,
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    *,
    risk_free_rate: float = 0.0,
) -> AllocationSummary:
    w = _normalize_long_only(weights)
    mu = _vector(expected_returns, name="expected_returns")
    if w.size != mu.size:
        raise ValueError("weights and expected_returns must have equal length")
    cov = _covariance(covariance, w.size)
    expected = float(w @ mu)
    vol = portfolio_volatility(w, cov)
    sharpe = 0.0 if vol <= 1e-15 else (expected - risk_free_rate) / vol
    return AllocationSummary(w, expected, vol, float(sharpe))
