"""Dependency-light diagonal Gaussian-mixture regime detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp


@dataclass(frozen=True, slots=True)
class GaussianMixtureFit:
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    converged: bool
    iterations: int
    log_likelihood: float


class DiagonalGaussianMixture:
    def __init__(
        self,
        n_components: int = 3,
        *,
        max_iter: int = 100,
        tolerance: float = 1e-5,
        variance_floor: float = 1e-6,
        random_state: int = 7,
    ) -> None:
        if n_components < 2:
            raise ValueError("n_components must be at least 2")
        if max_iter < 1 or tolerance <= 0.0 or variance_floor <= 0.0:
            raise ValueError("invalid Gaussian-mixture optimization parameters")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.variance_floor = variance_floor
        self.random_state = random_state
        self.fit_: GaussianMixtureFit | None = None

    @staticmethod
    def _matrix(features: npt.ArrayLike, *, minimum: int = 1) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] < minimum:
            raise ValueError(f"features must be 2D with at least {minimum} observations")
        if not np.isfinite(values).all():
            raise ValueError("features must be finite")
        return values

    def _log_prob(self, x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - means[None, :, :]
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * variances[None, :, :])
            + diff * diff / variances[None, :, :],
            axis=2,
        )

    def fit(self, features: npt.ArrayLike) -> GaussianMixtureFit:
        x = self._matrix(features, minimum=4)
        if x.shape[0] < self.n_components:
            raise ValueError("observations must be >= n_components")
        rng = np.random.default_rng(self.random_state)
        quantiles = np.linspace(0.0, 1.0, self.n_components + 2)[1:-1]
        projection = x[:, 0]
        centers = np.quantile(projection, quantiles)
        means = np.empty((self.n_components, x.shape[1]), dtype=np.float64)
        for index, center in enumerate(centers):
            anchor = int(np.argmin(np.abs(projection - center)))
            means[index] = x[anchor]
        means += rng.normal(0.0, 1e-8, size=means.shape)
        base_variance = np.maximum(np.var(x, axis=0, ddof=1), self.variance_floor)
        variances = np.tile(base_variance, (self.n_components, 1))
        weights = np.full(self.n_components, 1.0 / self.n_components)

        previous = -np.inf
        converged = False
        current = -np.inf
        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            log_joint = self._log_prob(x, means, variances) + np.log(weights)[None, :]
            log_norm = logsumexp(log_joint, axis=1)
            current = float(np.sum(log_norm))
            responsibilities = np.exp(log_joint - log_norm[:, None])
            mass = responsibilities.sum(axis=0) + 1e-12
            weights = mass / np.sum(mass)
            means = (responsibilities.T @ x) / mass[:, None]
            diff = x[:, None, :] - means[None, :, :]
            variances = np.sum(responsibilities[:, :, None] * diff * diff, axis=0) / mass[:, None]
            variances = np.maximum(variances, self.variance_floor)
            converged_now = abs(current - previous) <= self.tolerance * (1.0 + abs(previous))
            if np.isfinite(previous) and converged_now:
                converged = True
                break
            previous = current

        self.fit_ = GaussianMixtureFit(
            weights=weights.copy(),
            means=means.copy(),
            variances=variances.copy(),
            converged=converged,
            iterations=iterations,
            log_likelihood=current,
        )
        return self.fit_

    def predict_proba(self, features: npt.ArrayLike) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("model must be fit before prediction")
        x = self._matrix(features)
        fit = self.fit_
        log_joint = self._log_prob(x, fit.means, fit.variances) + np.log(fit.weights)[None, :]
        return np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])

    def predict(self, features: npt.ArrayLike) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1).astype(np.int64)
