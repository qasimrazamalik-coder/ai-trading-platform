"""Foundation-model catalog and lazy adapters; no checkpoint downloads occur implicitly."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec


class UsageTier(StrEnum):
    RESEARCH_ONLY = "research_only"
    CHECK_LICENSE = "check_license"
    PERMISSIVE = "permissive"


@dataclass(frozen=True, slots=True)
class FoundationModelSpec:
    model_id: str
    package: str
    domain: str
    usage: UsageTier
    notes: str

    @property
    def package_available(self) -> bool:
        return find_spec(self.package) is not None


DEFAULT_FOUNDATION_MODELS: tuple[FoundationModelSpec, ...] = (
    FoundationModelSpec(
        "amazon_chronos",
        "chronos",
        "general_time_series",
        UsageTier.CHECK_LICENSE,
        "Lazy adapter only; user must supply compatible local checkpoint/package.",
    ),
    FoundationModelSpec(
        "google_timesfm",
        "timesfm",
        "general_time_series",
        UsageTier.RESEARCH_ONLY,
        "Treat checkpoint licensing separately from source-code licensing.",
    ),
    FoundationModelSpec(
        "kronos_financial",
        "kronos",
        "financial_ohlcv",
        UsageTier.CHECK_LICENSE,
        "Lazy adapter only; never downloads weights automatically.",
    ),
)


def foundation_model_catalog() -> tuple[FoundationModelSpec, ...]:
    return DEFAULT_FOUNDATION_MODELS
