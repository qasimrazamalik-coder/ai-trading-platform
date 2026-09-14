from datetime import UTC, datetime

import pytest

from quant_os_core.intelligence import NewsItem, SourceTier, assess_credibility


def item(tier: SourceTier, primary: bool = False) -> NewsItem:
    return NewsItem(
        item_id="1",
        headline="Company files audited results",
        source="Exchange",
        source_tier=tier,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        is_primary_source=primary,
    )


def test_primary_corroborated_source_scores_high() -> None:
    result = assess_credibility(item(SourceTier.PRIMARY, True), corroborating_sources=3)
    assert result.score > 0.95
    assert result.confidence > 0.8


def test_contradictions_reduce_score() -> None:
    clean = assess_credibility(item(SourceTier.ESTABLISHED), corroborating_sources=2)
    contradicted = assess_credibility(
        item(SourceTier.ESTABLISHED), corroborating_sources=2, contradictory_sources=3
    )
    assert contradicted.score < clean.score
    assert any("contradiction" in reason for reason in contradicted.reasons)


def test_negative_counts_rejected() -> None:
    with pytest.raises(ValueError):
        assess_credibility(item(SourceTier.UNKNOWN), corroborating_sources=-1)
