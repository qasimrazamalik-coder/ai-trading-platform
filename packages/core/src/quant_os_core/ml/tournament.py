"""Leakage-aware walk-forward model tournaments."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from quant_os_core.ml.data import Standardizer, SupervisedDataset
from quant_os_core.ml.metrics import (
    directional_accuracy,
    information_coefficient,
    mean_squared_error,
)
from quant_os_core.right_fit.splits import walk_forward_splits


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    model_id: str
    factory: Callable[[], object]
    standardize: bool = True


@dataclass(frozen=True, slots=True)
class TournamentResult:
    model_id: str
    fold_mse: tuple[float, ...]
    fold_directional_accuracy: tuple[float, ...]
    fold_information_coefficient: tuple[float, ...]

    @property
    def mean_mse(self) -> float:
        return float(np.mean(self.fold_mse))

    @property
    def mean_directional_accuracy(self) -> float:
        return float(np.mean(self.fold_directional_accuracy))

    @property
    def mean_information_coefficient(self) -> float:
        return float(np.mean(self.fold_information_coefficient))


def walk_forward_tournament(
    dataset: SupervisedDataset,
    candidates: tuple[ModelCandidate, ...],
    *,
    train_size: int,
    test_size: int,
    gap: int = 1,
    expanding: bool = True,
) -> tuple[TournamentResult, ...]:
    if not candidates:
        raise ValueError("at least one model candidate is required")
    if len({item.model_id for item in candidates}) != len(candidates):
        raise ValueError("model candidate ids must be unique")
    splits = walk_forward_splits(
        dataset.targets.size,
        train_size=train_size,
        test_size=test_size,
        gap=gap,
        expanding=expanding,
    )
    results: list[TournamentResult] = []
    for candidate in candidates:
        fold_mse: list[float] = []
        fold_da: list[float] = []
        fold_ic: list[float] = []
        for split in splits:
            x_train = dataset.features[split.train]
            x_test = dataset.features[split.test]
            if candidate.standardize:
                scaler = Standardizer.fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            y_train = dataset.targets[split.train]
            y_test = dataset.targets[split.test]
            model = candidate.factory()
            fit = getattr(model, "fit", None)
            predict = getattr(model, "predict", None)
            if fit is None or predict is None:
                raise TypeError("candidate factory must return an object with fit and predict")
            fit(x_train, y_train)
            predictions = np.asarray(predict(x_test), dtype=np.float64)
            fold_mse.append(mean_squared_error(y_test, predictions))
            fold_da.append(directional_accuracy(y_test, predictions))
            fold_ic.append(information_coefficient(y_test, predictions))
        results.append(
            TournamentResult(
                model_id=candidate.model_id,
                fold_mse=tuple(fold_mse),
                fold_directional_accuracy=tuple(fold_da),
                fold_information_coefficient=tuple(fold_ic),
            )
        )
    return tuple(sorted(results, key=lambda item: item.mean_mse))
