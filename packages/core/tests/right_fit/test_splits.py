import numpy as np
import pytest

from quant_os_core.right_fit import (
    combinatorial_purged_splits,
    purged_kfold,
    walk_forward_splits,
)


def test_purged_kfold_respects_gaps() -> None:
    splits = purged_kfold(20, n_splits=4, purge=2, embargo=1)
    first = splits[0]
    assert np.array_equal(first.test, np.arange(0, 5))
    assert np.array_equal(first.train, np.arange(6, 20))
    middle = splits[1]
    assert np.array_equal(middle.test, np.arange(5, 10))
    assert not np.intersect1d(middle.train, np.arange(3, 11)).size


def test_purged_kfold_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        purged_kfold(5, n_splits=1)
    with pytest.raises(ValueError):
        purged_kfold(5, n_splits=6)


def test_walk_forward_expanding() -> None:
    splits = walk_forward_splits(20, train_size=8, test_size=4, gap=1)
    assert len(splits) == 2
    assert np.array_equal(splits[0].train, np.arange(0, 8))
    assert np.array_equal(splits[0].test, np.arange(9, 13))
    assert np.array_equal(splits[1].train, np.arange(0, 12))
    assert np.array_equal(splits[1].test, np.arange(13, 17))


def test_walk_forward_rolling() -> None:
    splits = walk_forward_splits(
        20,
        train_size=6,
        test_size=3,
        step_size=3,
        gap=1,
        expanding=False,
    )
    assert np.array_equal(splits[0].train, np.arange(0, 6))
    assert np.array_equal(splits[1].train, np.arange(3, 9))


def test_combinatorial_purged_split_count() -> None:
    splits = combinatorial_purged_splits(24, n_groups=6, test_groups=2)
    assert len(splits) == 15
    assert all(not np.intersect1d(split.train, split.test).size for split in splits)


def test_cpcv_honors_combination_guard() -> None:
    with pytest.raises(ValueError):
        combinatorial_purged_splits(40, n_groups=8, test_groups=4, max_combinations=10)
