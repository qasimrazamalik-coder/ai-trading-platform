"""Transparent source/corroboration credibility scoring."""

from __future__ import annotations

from quant_os_core.intelligence.models import CredibilityAssessment, NewsItem, SourceTier

_BASE = {
    SourceTier.PRIMARY: 0.90,
    SourceTier.AUTHORITATIVE: 0.84,
    SourceTier.ESTABLISHED: 0.72,
    SourceTier.COMMUNITY: 0.48,
    SourceTier.UNKNOWN: 0.35,
}


def assess_credibility(
    item: NewsItem,
    *,
    corroborating_sources: int = 0,
    contradictory_sources: int = 0,
) -> CredibilityAssessment:
    if corroborating_sources < 0 or contradictory_sources < 0:
        raise ValueError("source counts cannot be negative")
    score = _BASE[item.source_tier]
    reasons = [f"source tier: {item.source_tier.value}"]
    if item.is_primary_source:
        score += 0.07
        reasons.append("primary-source document")
    if corroborating_sources:
        boost = min(0.15, 0.035 * corroborating_sources)
        score += boost
        reasons.append(f"{corroborating_sources} independent corroboration(s)")
    if contradictory_sources:
        penalty = min(0.35, 0.09 * contradictory_sources)
        score -= penalty
        reasons.append(f"{contradictory_sources} contradiction(s)")
    score = min(1.0, max(0.0, score))
    evidence_count = corroborating_sources + contradictory_sources
    confidence = min(1.0, 0.55 + min(0.35, evidence_count * 0.07))
    if item.is_primary_source:
        confidence = min(1.0, confidence + 0.10)
    return CredibilityAssessment(
        score=score,
        confidence=confidence,
        corroborating_sources=corroborating_sources,
        contradictory_sources=contradictory_sources,
        reasons=tuple(reasons),
    )
