"""News, credibility and financial-intelligence primitives."""

from quant_os_core.intelligence.credibility import assess_credibility
from quant_os_core.intelligence.dedup import deduplicate_news, headline_similarity
from quant_os_core.intelligence.models import (
    CredibilityAssessment,
    IntelligenceSignal,
    NewsItem,
    SourceTier,
)
from quant_os_core.intelligence.sentiment import headline_sentiment

__all__ = [
    "CredibilityAssessment",
    "IntelligenceSignal",
    "NewsItem",
    "SourceTier",
    "assess_credibility",
    "deduplicate_news",
    "headline_sentiment",
    "headline_similarity",
]
