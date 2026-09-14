from datetime import UTC, datetime, timedelta

import pytest

from quant_os_core.intelligence import (
    NewsItem,
    deduplicate_news,
    headline_sentiment,
    headline_similarity,
)


def news(item_id: str, headline: str, minutes: int = 0) -> NewsItem:
    return NewsItem(
        item_id=item_id,
        headline=headline,
        source="Source",
        published_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes),
    )


def test_sentiment_baseline_and_negation() -> None:
    assert headline_sentiment("Company beats estimates with record profit") > 0
    assert headline_sentiment("Exchange hack warning after collapse") < 0
    assert headline_sentiment("Company did not miss estimates") > 0
    assert headline_sentiment("plain factual update") == 0


def test_similarity_and_dedup() -> None:
    a = "Bitcoin surges after ETF approval"
    b = "Bitcoin surges following ETF approval"
    assert headline_similarity(a, b) > 0.5
    items = (news("1", a), news("2", b, 1), news("3", "Oil inventories unexpectedly fall", 2))
    result = deduplicate_news(items, threshold=0.5)
    assert [value.item_id for value in result] == ["1", "3"]


def test_bad_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        deduplicate_news((), threshold=1.1)
