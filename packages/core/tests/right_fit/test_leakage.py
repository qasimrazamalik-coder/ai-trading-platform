import numpy as np
import pytest

from quant_os_core.right_fit import (
    assert_monotonic_unique_timestamps,
    interval_overlap_mask,
    purge_label_overlap,
    train_test_overlap,
)


def test_timestamp_guard_accepts_strict_order() -> None:
    assert_monotonic_unique_timestamps([1, 2, 3, 4])


def test_timestamp_guard_rejects_duplicate() -> None:
    with pytest.raises(ValueError):
        assert_monotonic_unique_timestamps([1, 2, 2, 3])


def test_train_test_overlap_reports_shared_indices() -> None:
    shared = train_test_overlap([0, 1, 2], [2, 3, 4])
    assert np.array_equal(shared, np.array([2]))


def test_interval_overlap_mask() -> None:
    result = interval_overlap_mask([0, 3, 8], [2, 6, 10], test_start=5, test_end=8)
    assert result.tolist() == [False, True, True]


def test_purge_label_overlap_removes_future_crossing_samples() -> None:
    train = np.arange(0, 7)
    label_ends = np.array([1, 2, 4, 5, 5, 6, 7, 8, 9, 9])
    test = np.array([5, 6])
    purged = purge_label_overlap(train, label_ends, test)
    assert np.array_equal(purged, np.array([0, 1, 2]))


def test_purge_label_overlap_applies_embargo() -> None:
    train = np.array([0, 1, 7, 8])
    label_ends = np.arange(10)
    test = np.array([4, 5])
    purged = purge_label_overlap(train, label_ends, test, embargo=2)
    assert np.array_equal(purged, np.array([0, 1, 8]))


def test_timestamp_guard_preserves_nanosecond_integer_precision() -> None:
    base = 1_800_000_000_000_000_000
    assert_monotonic_unique_timestamps([base, base + 1, base + 2])
    with pytest.raises(ValueError):
        assert_monotonic_unique_timestamps([base, base + 1, base + 1])
