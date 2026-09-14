"""Portfolio covariance and risk-decomposition mathematics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quant_os_core.quant._validation import FloatArray, as_float_array

Matrix = NDArray[np.float64]


def covariance_matrix(returns: ArrayLike) -> Matrix:
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("returns must be a 2-D array with observations x assets")
    if not np.isfinite(matrix).all():
        raise ValueError("returns must contain only finite values")
    return np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))


def correlation_matrix(returns: ArrayLike) -> Matrix:
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("returns must be a 2-D array with >=2 observations and >=2 assets")
    if not np.isfinite(matrix).all():
        raise ValueError("returns must contain only finite values")
    stds = np.std(matrix, axis=0, ddof=1)
    if np.any(stds == 0.0):
        raise ValueError("correlation is undefined for constant asset series")
    return np.corrcoef(matrix, rowvar=False)


def _validated_weights_covariance(
    weights: ArrayLike, covariance: ArrayLike
) -> tuple[FloatArray, Matrix]:
    w = as_float_array(weights, name="weights", min_size=1)
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape != (w.size, w.size):
        raise ValueError("covariance must be square and match number of weights")
    if not np.isfinite(cov).all():
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(cov)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    return w, cov


def portfolio_variance(weights: ArrayLike, covariance: ArrayLike) -> float:
    w, cov = _validated_weights_covariance(weights, covariance)
    value = float(w @ cov @ w)
    return max(0.0, value)


def portfolio_volatility(weights: ArrayLike, covariance: ArrayLike) -> float:
    return float(np.sqrt(portfolio_variance(weights, covariance)))


def marginal_risk_contribution(weights: ArrayLike, covariance: ArrayLike) -> FloatArray:
    w, cov = _validated_weights_covariance(weights, covariance)
    vol = portfolio_volatility(w, cov)
    if vol == 0.0:
        raise ValueError("marginal risk contribution undefined for zero-volatility portfolio")
    return cov @ w / vol


def component_risk_contribution(weights: ArrayLike, covariance: ArrayLike) -> FloatArray:
    w, cov = _validated_weights_covariance(weights, covariance)
    return w * marginal_risk_contribution(w, cov)


def risk_contribution_percentages(weights: ArrayLike, covariance: ArrayLike) -> FloatArray:
    components = component_risk_contribution(weights, covariance)
    total = float(np.sum(components))
    if total == 0.0:
        raise ValueError("risk contribution percentages undefined when total risk is zero")
    return components / total


def nearest_psd(matrix: ArrayLike, *, floor: float = 0.0) -> Matrix:
    """Symmetrize and clip eigenvalues to obtain a positive-semidefinite matrix."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("matrix must be square")
    if not np.isfinite(x).all():
        raise ValueError("matrix must contain only finite values")
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("floor must be finite and >= 0")
    symmetric = (x + x.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, floor)
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    return (repaired + repaired.T) / 2.0
