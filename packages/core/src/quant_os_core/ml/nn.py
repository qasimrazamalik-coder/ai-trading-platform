"""A small neural network in NumPy: runs anywhere, no GPU or PyTorch needed.

Multi-layer perceptron for binary probabilities ("will this trade hit its target before its
stop?"): He-initialised ReLU layers, dropout, L2 weight decay, Adam, mini-batches, and early
stopping on the most recent slice of the training data (chronological, never shuffled across
time). Inputs are standardised with statistics from the training rows only. Hidden layers set
to ``()`` give plain logistic regression, the baseline the network has to beat.

The PyTorch sequence models in ``ml.deep`` (LSTM, TCN, Transformer, PatchTST) remain available
for research when PyTorch is installed; this module is what the live bot uses.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _sigmoid(z: FloatArray) -> FloatArray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable


@dataclass(slots=True)
class MLPClassifier:
    hidden: tuple[int, ...] = (32, 16)
    learning_rate: float = 1e-3
    l2: float = 1e-4
    dropout: float = 0.1
    epochs: int = 300
    batch_size: int = 64
    patience: int = 25
    validation_fraction: float = 0.2
    seed: int = 7
    weights: list[FloatArray] = field(default_factory=list)
    biases: list[FloatArray] = field(default_factory=list)
    mean: FloatArray | None = None
    scale: FloatArray | None = None
    history: list[tuple[float, float]] = field(default_factory=list)  # (train, val) loss

    # ---- core maths ---------------------------------------------------------------------
    def _forward(
        self, x: FloatArray, rng: np.random.Generator | None
    ) -> tuple[FloatArray, list[FloatArray], list[FloatArray | None]]:
        activations, masks = [x], []
        a = x
        for w, b in zip(self.weights[:-1], self.biases[:-1], strict=True):
            a = np.maximum(a @ w + b, 0.0)
            mask = None
            if rng is not None and self.dropout > 0:
                mask = (rng.random(a.shape) >= self.dropout) / (1 - self.dropout)
                a = a * mask
            activations.append(a)
            masks.append(mask)
        logits = a @ self.weights[-1] + self.biases[-1]
        return _sigmoid(logits[:, 0]), activations, masks

    @staticmethod
    def _loss(p: FloatArray, y: FloatArray) -> float:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _standardise(self, x: FloatArray) -> FloatArray:
        assert self.mean is not None and self.scale is not None
        return (x - self.mean) / self.scale

    # ---- public API ---------------------------------------------------------------------
    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> MLPClassifier:
        features = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if features.ndim != 2 or target.shape != (features.shape[0],):
            raise ValueError("x must be [n, k] and y must have n labels")
        if not np.isfinite(features).all() or not set(np.unique(target)) <= {0.0, 1.0}:
            raise ValueError("x must be finite and y binary")
        if features.shape[0] < 30 or np.unique(target).size < 2:
            raise ValueError("need at least 30 rows containing both classes")
        rng = np.random.default_rng(self.seed)
        n_val = max(5, int(features.shape[0] * self.validation_fraction))
        x_train, y_train = features[:-n_val], target[:-n_val]
        x_val, y_val = features[-n_val:], target[-n_val:]  # most recent rows, never shuffled in
        self.mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        xt, xv = self._standardise(x_train), self._standardise(x_val)

        sizes = [features.shape[1], *self.hidden, 1]
        self.weights = [
            rng.normal(0.0, math.sqrt(2.0 / fan_in), (fan_in, fan_out))
            for fan_in, fan_out in pairwise(sizes)
        ]
        self.biases = [np.zeros(fan_out) for fan_out in sizes[1:]]
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0

        best = (math.inf, [w.copy() for w in self.weights], [b.copy() for b in self.biases])
        waited = 0
        self.history = []
        for _ in range(self.epochs):
            order = rng.permutation(xt.shape[0])  # shuffling within the training slice only
            for start in range(0, order.size, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb, yb = xt[idx], y_train[idx]
                p, acts, masks = self._forward(xb, rng)
                delta = ((p - yb) / idx.size)[:, None]  # dLoss/dlogit for sigmoid + BCE
                grads_w, grads_b = [], []
                for layer in range(len(self.weights) - 1, -1, -1):
                    grads_w.append(acts[layer].T @ delta + self.l2 * self.weights[layer])
                    grads_b.append(delta.sum(axis=0))
                    if layer > 0:
                        delta = delta @ self.weights[layer].T
                        delta = delta * (acts[layer] > 0)
                        mask = masks[layer - 1]
                        if mask is not None:
                            delta = delta * mask
                grads_w.reverse()
                grads_b.reverse()
                step += 1
                for i in range(len(self.weights)):
                    for param, grad, m, v in (
                        (self.weights, grads_w, m_w, v_w),
                        (self.biases, grads_b, m_b, v_b),
                    ):
                        m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2
                        m_hat = m[i] / (1 - beta1**step)
                        v_hat = v[i] / (1 - beta2**step)
                        param[i] = param[i] - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            train_loss = self._loss(self._forward(xt, None)[0], y_train)
            val_loss = self._loss(self._forward(xv, None)[0], y_val)
            self.history.append((train_loss, val_loss))
            if val_loss < best[0] - 1e-5:
                best = (val_loss, [w.copy() for w in self.weights], [b.copy() for b in self.biases])
                waited = 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.weights, self.biases = best[1], best[2]
        return self

    def predict_proba(self, x: npt.ArrayLike) -> FloatArray:
        if not self.weights or self.mean is None:
            raise RuntimeError("fit the model first")
        features = np.asarray(x, dtype=np.float64)
        if features.ndim == 1:
            features = features[None, :]
        if features.shape[1] != self.mean.size or not np.isfinite(features).all():
            raise ValueError("features must be finite with the trained column count")
        return self._forward(self._standardise(features), None)[0]

    # ---- persistence --------------------------------------------------------------------
    def to_arrays(self, prefix: str = "") -> dict[str, FloatArray]:
        if self.mean is None or self.scale is None:
            raise RuntimeError("fit the model first")
        out = {f"{prefix}mean": self.mean, f"{prefix}scale": self.scale}
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            out[f"{prefix}w{i}"], out[f"{prefix}b{i}"] = w, b
        return out

    @classmethod
    def from_arrays(cls, arrays: dict[str, FloatArray], prefix: str = "") -> MLPClassifier:
        model = cls()
        model.mean, model.scale = arrays[f"{prefix}mean"], arrays[f"{prefix}scale"]
        i = 0
        while f"{prefix}w{i}" in arrays:
            model.weights.append(arrays[f"{prefix}w{i}"])
            model.biases.append(arrays[f"{prefix}b{i}"])
            i += 1
        model.hidden = tuple(w.shape[1] for w in model.weights[:-1])
        return model


@dataclass(slots=True)
class PlattScaler:
    """p = sigmoid(a * logit(score) + b), fitted by Newton's method on held-out scores."""

    a: float = 1.0
    b: float = 0.0

    def fit(self, scores: npt.ArrayLike, labels: npt.ArrayLike) -> PlattScaler:
        s = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
        y = np.asarray(labels, dtype=np.float64)
        if s.shape != y.shape or s.size < 20 or np.unique(y).size < 2:
            raise ValueError("need 20+ held-out scores with both classes")
        z = np.log(s / (1 - s))
        a, b = 1.0, 0.0
        for _ in range(50):
            p = _sigmoid(a * z + b)
            g = np.array([np.sum((p - y) * z), np.sum(p - y)])
            w = p * (1 - p) + 1e-9
            hess = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]])
            hess += np.eye(2) * 1e-6
            delta = np.linalg.solve(hess, g)
            a, b = a - float(delta[0]), b - float(delta[1])
            if np.abs(delta).max() < 1e-8:
                break
        self.a, self.b = a, b
        return self

    def transform(self, scores: npt.ArrayLike) -> FloatArray:
        s = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
        return _sigmoid(self.a * np.log(s / (1 - s)) + self.b)


@dataclass(slots=True)
class ConfidenceModel:
    """Network + logistic baseline ensemble + calibration, saved as one ``.npz`` bundle."""

    network: MLPClassifier
    baseline: MLPClassifier
    calibrator: PlattScaler
    feature_names: tuple[str, ...]
    network_weight: float = 0.5
    metadata: dict[str, object] = field(default_factory=dict)

    def raw(self, x: npt.ArrayLike) -> FloatArray:
        w = self.network_weight
        return w * self.network.predict_proba(x) + (1 - w) * self.baseline.predict_proba(x)

    def probability(self, x: npt.ArrayLike) -> FloatArray:
        return self.calibrator.transform(self.raw(x))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {**self.network.to_arrays("nn_"), **self.baseline.to_arrays("lr_")}
        meta = {
            "feature_names": list(self.feature_names),
            "network_weight": self.network_weight,
            "platt": [self.calibrator.a, self.calibrator.b],
            "metadata": self.metadata,
        }
        tmp = path.with_suffix(".tmp.npz")
        payload = cast(dict[str, Any], {"meta": np.array(json.dumps(meta)), **arrays})
        np.savez(str(tmp), **payload)
        tmp.replace(path)  # atomic: a crash never leaves a half-written model

    @classmethod
    def load(cls, path: Path) -> ConfidenceModel:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        meta = json.loads(str(arrays.pop("meta")))
        return cls(
            network=MLPClassifier.from_arrays(arrays, "nn_"),
            baseline=MLPClassifier.from_arrays(arrays, "lr_"),
            calibrator=PlattScaler(*meta["platt"]),
            feature_names=tuple(meta["feature_names"]),
            network_weight=float(meta["network_weight"]),
            metadata=dict(meta.get("metadata", {})),
        )
