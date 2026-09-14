"""Dependency-free financial headline sentiment baseline.

This is deliberately a transparent baseline. Learned NLP models can replace it through the
same interface, but the platform always retains an auditable non-LLM reference model.
"""

from __future__ import annotations

import re

_POSITIVE = {
    "beat": 1.0,
    "beats": 1.0,
    "growth": 0.7,
    "upgrade": 0.8,
    "surge": 0.8,
    "record": 0.5,
    "approval": 0.7,
    "profit": 0.6,
    "bullish": 0.8,
    "recovery": 0.6,
}
_NEGATIVE = {
    "miss": -1.0,
    "misses": -1.0,
    "downgrade": -0.8,
    "fraud": -1.0,
    "hack": -0.9,
    "lawsuit": -0.7,
    "loss": -0.6,
    "default": -1.0,
    "bearish": -0.8,
    "collapse": -1.0,
    "warning": -0.5,
}
_NEGATORS = {"not", "no", "never", "without"}


def headline_sentiment(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return 0.0
    score = 0.0
    hits = 0
    for index, token in enumerate(tokens):
        weight = _POSITIVE.get(token, _NEGATIVE.get(token))
        if weight is None:
            continue
        if index > 0 and tokens[index - 1] in _NEGATORS:
            weight *= -1
        score += weight
        hits += 1
    if hits == 0:
        return 0.0
    return max(-1.0, min(1.0, score / hits))
