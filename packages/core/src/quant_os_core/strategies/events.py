"""Event/news feature-to-signal transforms.

These functions do not determine whether a story is true. Credibility and provenance are
expected to be produced by the later intelligence layer and supplied as explicit inputs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_os_core.quant._validation import as_float_array, paired_float_arrays


def credibility_weighted_event_signal(
    sentiment: npt.ArrayLike,
    credibility: npt.ArrayLike,
    *,
    minimum_credibility: float = 0.7,
    activation_threshold: float = 0.2,
) -> npt.NDArray[np.float64]:
    sentiments, credibility_scores = paired_float_arrays(
        sentiment,
        credibility,
        left_name="sentiment",
        right_name="credibility",
        min_size=1,
    )
    if np.any(np.abs(sentiments) > 1.0):
        raise ValueError("sentiment must stay inside [-1, 1]")
    if np.any((credibility_scores < 0.0) | (credibility_scores > 1.0)):
        raise ValueError("credibility must stay inside [0, 1]")
    minimum = float(minimum_credibility)
    threshold = float(activation_threshold)
    if not 0.0 <= minimum <= 1.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("thresholds must stay inside [0, 1]")
    weighted = sentiments * credibility_scores
    signal = np.zeros(weighted.size, dtype=np.float64)
    trusted = credibility_scores >= minimum
    signal[trusted & (weighted > threshold)] = 1.0
    signal[trusted & (weighted < -threshold)] = -1.0
    return signal


def decay_event_signal(
    impulses: npt.ArrayLike,
    *,
    decay: float = 0.8,
) -> npt.NDArray[np.float64]:
    values = as_float_array(impulses, name="impulses")
    if np.any(np.abs(values) > 1.0):
        raise ValueError("impulses must stay inside [-1, 1]")
    factor = float(decay)
    if not 0.0 <= factor < 1.0:
        raise ValueError("decay must be in [0, 1)")
    result = np.zeros(values.size, dtype=np.float64)
    state = 0.0
    for index, impulse in enumerate(values):
        state = state * factor + impulse
        result[index] = float(np.clip(state, -1.0, 1.0))
    return result
