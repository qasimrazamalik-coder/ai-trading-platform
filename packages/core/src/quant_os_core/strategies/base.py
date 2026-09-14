"""Shared strategy contracts and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class StrategyCategory(StrEnum):
    TREND = "trend"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    STAT_ARB = "stat_arb"
    CARRY = "carry"
    MICROSTRUCTURE = "microstructure"
    FUNDAMENTAL = "fundamental"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    name: str
    default: float | int
    minimum: float | int | None = None
    maximum: float | int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name cannot be blank")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("parameter minimum cannot exceed maximum")
        if self.minimum is not None and self.default < self.minimum:
            raise ValueError("default cannot be below minimum")
        if self.maximum is not None and self.default > self.maximum:
            raise ValueError("default cannot exceed maximum")


@dataclass(frozen=True, slots=True)
class StrategyMetadata:
    strategy_id: str
    name: str
    category: StrategyCategory
    description: str
    supported_markets: tuple[str, ...] = ("multi_asset",)
    parameters: tuple[ParameterDefinition, ...] = ()
    known_failure_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id cannot be blank")
        if not self.name.strip():
            raise ValueError("strategy name cannot be blank")


@dataclass(frozen=True, slots=True)
class StrategySignals:
    """Normalized target exposure in [-1, 1] plus optional diagnostics."""

    positions: FloatArray
    metadata: StrategyMetadata
    diagnostics: dict[str, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.positions, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("positions must be a non-empty one-dimensional array")
        if not np.isfinite(values).all():
            raise ValueError("positions must contain only finite values")
        if np.any(np.abs(values) > 1.0 + 1e-12):
            raise ValueError("positions must stay inside [-1, 1]")
        object.__setattr__(self, "positions", values.copy())
        for name, diagnostic in self.diagnostics.items():
            parsed = np.asarray(diagnostic, dtype=np.float64)
            if parsed.ndim != 1 or parsed.size != values.size:
                raise ValueError(f"diagnostic {name!r} must match positions length")
            if not np.isfinite(parsed).all():
                raise ValueError(f"diagnostic {name!r} must contain finite values")


class VectorStrategy(Protocol):
    metadata: StrategyMetadata

    def generate(self, prices: npt.ArrayLike, **kwargs: npt.ArrayLike) -> StrategySignals: ...
