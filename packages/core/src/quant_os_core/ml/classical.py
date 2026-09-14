"""Classical ML model adapters with optional third-party boosters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class Regressor(Protocol):
    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> Any: ...
    def predict(self, x: npt.ArrayLike) -> npt.ArrayLike: ...


@dataclass(slots=True)
class SklearnRegressorAdapter:
    model: Regressor
    name: str

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> SklearnRegressorAdapter:
        features = np.asarray(x, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64)
        if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.size:
            raise ValueError("invalid training arrays")
        self.model.fit(features, targets)
        return self

    def predict(self, x: npt.ArrayLike) -> np.ndarray:
        features = np.asarray(x, dtype=np.float64)
        if self.name == "lightgbm" and hasattr(self.model, "booster_"):
            raw = getattr(self.model, "booster_").predict(features)
        else:
            raw = self.model.predict(features)
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("model returned invalid predictions")
        return values


def ridge_regressor(*, alpha: float = 1.0) -> SklearnRegressorAdapter:
    from sklearn.linear_model import Ridge

    return SklearnRegressorAdapter(Ridge(alpha=alpha), "ridge")


def random_forest_regressor(
    *, random_state: int = 7, n_estimators: int = 200
) -> SklearnRegressorAdapter:
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=3,
    )
    return SklearnRegressorAdapter(model, "random_forest")


def xgboost_regressor(*, random_state: int = 7) -> SklearnRegressorAdapter:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("xgboost is not installed; install quant-os[ml]") from exc
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    return SklearnRegressorAdapter(model, "xgboost")


def lightgbm_regressor(*, random_state: int = 7) -> SklearnRegressorAdapter:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("lightgbm is not installed; install quant-os[ml]") from exc
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=random_state,
        verbosity=-1,
    )
    return SklearnRegressorAdapter(model, "lightgbm")


def catboost_regressor(*, random_state: int = 7) -> SklearnRegressorAdapter:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("catboost is not installed; install quant-os[ml]") from exc
    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.03,
        random_seed=random_state,
        verbose=False,
    )
    return SklearnRegressorAdapter(model, "catboost")


@dataclass(slots=True)
class SklearnClassifierAdapter:
    model: Any
    name: str

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> SklearnClassifierAdapter:
        features = np.asarray(x, dtype=np.float64)
        targets = np.asarray(y, dtype=np.int64)
        if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.size:
            raise ValueError("invalid training arrays")
        if not set(np.unique(targets)).issubset({0, 1}) or np.unique(targets).size < 2:
            raise ValueError("classification targets must contain both binary classes")
        self.model.fit(features, targets)
        return self

    def predict_proba(self, x: npt.ArrayLike) -> np.ndarray:
        features = np.asarray(x, dtype=np.float64)
        probabilities = np.asarray(self.model.predict_proba(features)[:, 1], dtype=np.float64)
        if probabilities.ndim != 1 or np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("model returned invalid probabilities")
        return probabilities


def logistic_classifier(*, c: float = 1.0) -> SklearnClassifierAdapter:
    from sklearn.linear_model import LogisticRegression

    if c <= 0.0:
        raise ValueError("c must be positive")
    return SklearnClassifierAdapter(LogisticRegression(C=c, max_iter=1_000), "logistic")
