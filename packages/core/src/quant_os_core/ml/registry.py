"""Versioned in-process model registry metadata."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class ModelFamily(StrEnum):
    BASELINE = "baseline"
    LINEAR = "linear"
    TREE = "tree"
    BOOSTING = "boosting"
    DEEP = "deep"
    FOUNDATION = "foundation"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True, slots=True)
class ModelVersion:
    model_id: str
    version: int
    family: ModelFamily
    created_at: datetime
    artifact_uri: str | None = None
    training_data_hash: str | None = None


class ModelRegistry:
    def __init__(self) -> None:
        self._versions: dict[str, list[ModelVersion]] = {}

    def register(
        self,
        model_id: str,
        family: ModelFamily,
        *,
        artifact_uri: str | None = None,
        training_data_hash: str | None = None,
    ) -> ModelVersion:
        if not model_id.strip():
            raise ValueError("model_id cannot be blank")
        versions = self._versions.setdefault(model_id, [])
        version = ModelVersion(
            model_id=model_id,
            version=len(versions) + 1,
            family=family,
            created_at=datetime.now(UTC),
            artifact_uri=artifact_uri,
            training_data_hash=training_data_hash,
        )
        versions.append(version)
        return version

    def latest(self, model_id: str) -> ModelVersion:
        versions = self._versions.get(model_id)
        if not versions:
            raise KeyError(model_id)
        return versions[-1]

    def versions(self, model_id: str) -> tuple[ModelVersion, ...]:
        return tuple(self._versions.get(model_id, ()))
