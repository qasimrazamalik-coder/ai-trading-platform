"""Simple headline de-duplication for multi-source feeds."""

from __future__ import annotations

import re

from quant_os_core.intelligence.models import NewsItem


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def headline_similarity(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def deduplicate_news(
    items: tuple[NewsItem, ...],
    *,
    threshold: float = 0.72,
) -> tuple[NewsItem, ...]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    kept: list[NewsItem] = []
    for item in sorted(items, key=lambda value: value.published_at):
        if any(headline_similarity(item.headline, other.headline) >= threshold for other in kept):
            continue
        kept.append(item)
    return tuple(kept)
