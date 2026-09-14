"""Minimal local experiment tracking with JSONL output."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    model_id: str
    metrics: dict[str, float]
    parameters: dict[str, Any]
    created_at: str


class JsonlExperimentTracker:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(
        self,
        experiment_id: str,
        model_id: str,
        *,
        metrics: dict[str, float],
        parameters: dict[str, Any],
    ) -> ExperimentRecord:
        if not experiment_id.strip() or not model_id.strip():
            raise ValueError("experiment_id and model_id cannot be blank")
        record = ExperimentRecord(
            experiment_id=experiment_id,
            model_id=model_id,
            metrics={key: float(value) for key, value in metrics.items()},
            parameters=parameters,
            created_at=datetime.now(UTC).isoformat(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True, default=str) + "\n")
        return record
