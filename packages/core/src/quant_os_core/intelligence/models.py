"""Typed market-intelligence records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceTier(StrEnum):
    PRIMARY = "primary"
    AUTHORITATIVE = "authoritative"
    ESTABLISHED = "established"
    COMMUNITY = "community"
    UNKNOWN = "unknown"


class NewsItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str = Field(min_length=1, max_length=160)
    headline: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=160)
    source_tier: SourceTier = SourceTier.UNKNOWN
    published_at: datetime
    entities: tuple[str, ...] = ()
    url: str | None = None
    is_primary_source: bool = False

    @field_validator("headline", "source")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class CredibilityAssessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    corroborating_sources: int = Field(ge=0)
    contradictory_sources: int = Field(ge=0)
    reasons: tuple[str, ...]


class IntelligenceSignal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    sentiment: float = Field(ge=-1, le=1)
    novelty: float = Field(ge=0, le=1)
    market_relevance: float = Field(ge=0, le=1)
    credibility: CredibilityAssessment

    @property
    def risk_adjusted_impact(self) -> float:
        return (
            self.sentiment
            * self.novelty
            * self.market_relevance
            * self.credibility.score
        )
