"""Small diagonal Gaussian HMM for unsupervised market-regime discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp


@dataclass(frozen=True, slots=True)
class GaussianHMMFit:
    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    converged: bool
    iterations: int
    log_likelihood: float


class GaussianHMMRegimeDetector:
    def __init__(
        self,
        n_states: int = 3,
        *,
        max_iter: int = 100,
        tolerance: float = 1e-5,
        variance_floor: float = 1e-6,
        transition_prior: float = 1e-2,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be at least 2")
        if max_iter < 1 or tolerance <= 0.0 or variance_floor <= 0.0:
            raise ValueError("invalid HMM optimization parameters")
        if transition_prior <= 0.0:
            raise ValueError("transition_prior must be positive")
        self.n_states = n_states
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.variance_floor = variance_floor
        self.transition_prior = transition_prior
        self.fit_: GaussianHMMFit | None = None

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

    def _emissions(self, x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - means[None, :, :]
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * variances[None, :, :])
            + diff * diff / variances[None, :, :],
            axis=2,
        )

    @staticmethod
    def _forward(
        log_emission: np.ndarray,
        log_initial: np.ndarray,
        log_transition: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        count, states = log_emission.shape
        alpha = np.empty((count, states), dtype=np.float64)
        alpha[0] = log_initial + log_emission[0]
        for t in range(1, count):
            alpha[t] = log_emission[t] + logsumexp(alpha[t - 1][:, None] + log_transition, axis=0)
        return alpha, float(logsumexp(alpha[-1]))

    @staticmethod
    def _backward(log_emission: np.ndarray, log_transition: np.ndarray) -> np.ndarray:
        count, states = log_emission.shape
        beta = np.zeros((count, states), dtype=np.float64)
        for t in range(count - 2, -1, -1):
            beta[t] = logsumexp(
                log_transition + log_emission[t + 1][None, :] + beta[t + 1][None, :],
                axis=1,
            )
        return beta

    def _initialize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ordering = np.argsort(x[:, 0])
        chunks = np.array_split(ordering, self.n_states)
        means = np.vstack([np.mean(x[index], axis=0) for index in chunks])
        base_variance = np.maximum(np.var(x, axis=0, ddof=1), self.variance_floor)
        variances = np.tile(base_variance, (self.n_states, 1))
        initial = np.full(self.n_states, 1.0 / self.n_states)
        transition = np.full(
            (self.n_states, self.n_states),
            (1.0 - 0.80) / max(self.n_states - 1, 1),
        )
        np.fill_diagonal(transition, 0.80)
        transition /= transition.sum(axis=1, keepdims=True)
        return initial, transition, means, variances

    def fit(self, features: npt.ArrayLike) -> GaussianHMMFit:
        x = self._matrix(features, minimum=4)
        if x.shape[0] < self.n_states * 2:
            raise ValueError("HMM requires at least two observations per state")
        initial, transition, means, variances = self._initialize(x)
        previous = -np.inf
        converged = False
        current = -np.inf
        iterations = 0

        for iterations in range(1, self.max_iter + 1):
            log_emission = self._emissions(x, means, variances)
            log_initial = np.log(np.maximum(initial, 1e-300))
            log_transition = np.log(np.maximum(transition, 1e-300))
            alpha, current = self._forward(log_emission, log_initial, log_transition)
            beta = self._backward(log_emission, log_transition)
            log_gamma = alpha + beta - current
            gamma = np.exp(log_gamma)
            gamma /= gamma.sum(axis=1, keepdims=True)

            xi_sum = np.zeros_like(transition)
            for t in range(x.shape[0] - 1):
                log_xi = (
                    alpha[t][:, None]
                    + log_transition
                    + log_emission[t + 1][None, :]
                    + beta[t + 1][None, :]
                    - current
                )
                xi = np.exp(log_xi)
                xi_sum += xi / max(float(np.sum(xi)), 1e-300)

            initial = gamma[0] / np.sum(gamma[0])
            transition = xi_sum + self.transition_prior
            transition /= transition.sum(axis=1, keepdims=True)
            mass = gamma.sum(axis=0) + 1e-12
            means = (gamma.T @ x) / mass[:, None]
            diff = x[:, None, :] - means[None, :, :]
            variances = np.sum(gamma[:, :, None] * diff * diff, axis=0) / mass[:, None]
            variances = np.maximum(variances, self.variance_floor)

            converged_now = abs(current - previous) <= self.tolerance * (1.0 + abs(previous))
            if np.isfinite(previous) and converged_now:
                converged = True
                break
            previous = current

        self.fit_ = GaussianHMMFit(
            initial_probabilities=initial.copy(),
            transition_matrix=transition.copy(),
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
        emissions = self._emissions(x, fit.means, fit.variances)
        alpha, likelihood = self._forward(
            emissions,
            np.log(np.maximum(fit.initial_probabilities, 1e-300)),
            np.log(np.maximum(fit.transition_matrix, 1e-300)),
        )
        beta = self._backward(emissions, np.log(np.maximum(fit.transition_matrix, 1e-300)))
        probabilities = np.exp(alpha + beta - likelihood)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, features: npt.ArrayLike) -> np.ndarray:
        """Return Viterbi most-likely state path."""
        if self.fit_ is None:
            raise RuntimeError("model must be fit before prediction")
        x = self._matrix(features)
        fit = self.fit_
        emissions = self._emissions(x, fit.means, fit.variances)
        log_transition = np.log(np.maximum(fit.transition_matrix, 1e-300))
        delta = np.empty((x.shape[0], self.n_states), dtype=np.float64)
        pointers = np.zeros((x.shape[0], self.n_states), dtype=np.int64)
        delta[0] = np.log(np.maximum(fit.initial_probabilities, 1e-300)) + emissions[0]
        for t in range(1, x.shape[0]):
            candidates = delta[t - 1][:, None] + log_transition
            pointers[t] = np.argmax(candidates, axis=0)
            delta[t] = emissions[t] + np.max(candidates, axis=0)
        states = np.empty(x.shape[0], dtype=np.int64)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(x.shape[0] - 2, -1, -1):
            states[t] = pointers[t + 1, states[t + 1]]
        return states
