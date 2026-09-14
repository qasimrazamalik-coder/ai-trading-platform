import numpy as np
import pytest

from quant_os_core.right_fit import (
    FitDiagnosis,
    RightFitDecision,
    RightFitPolicy,
    TimeSeriesSplit,
    assert_monotonic_unique_timestamps,
    combinatorial_purged_splits,
    complexity_penalized_score,
    diagnose_fit,
    evaluate_right_fit,
    expected_maximum_sharpe,
    interval_overlap_mask,
    parameter_surface_stability,
    probability_backtest_overfitting,
    purge_label_overlap,
    purged_kfold,
    sidak_adjusted_pvalue,
    train_test_overlap,
    walk_forward_parameter_selection,
    walk_forward_splits,
)


def test_index_and_timestamp_validation_guards() -> None:
    with pytest.raises(ValueError):
        train_test_overlap([[1, 2]], [3, 4])
    with pytest.raises(ValueError):
        train_test_overlap([], [1])
    with pytest.raises(ValueError):
        train_test_overlap([1.5], [2])
    with pytest.raises(ValueError):
        train_test_overlap([-1], [2])
    with pytest.raises(ValueError):
        assert_monotonic_unique_timestamps([1.0, np.inf])
    with pytest.raises(ValueError):
        assert_monotonic_unique_timestamps([1])
    assert_monotonic_unique_timestamps(
        np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]")
    )


def test_interval_and_label_purge_validation_guards() -> None:
    with pytest.raises(ValueError):
        interval_overlap_mask([0, 1], [0], test_start=1, test_end=2)
    with pytest.raises(ValueError):
        interval_overlap_mask([0, 2], [1, 1], test_start=1, test_end=2)
    with pytest.raises(ValueError):
        interval_overlap_mask([0], [1], test_start=2, test_end=1)
    with pytest.raises(ValueError):
        purge_label_overlap([0], [1], [2], embargo=-1)
    with pytest.raises(ValueError):
        purge_label_overlap([5], [0, 1], [2])


def test_split_dataclass_and_splitter_guards() -> None:
    with pytest.raises(ValueError):
        TimeSeriesSplit(train=np.array([[0]]), test=np.array([1]))
    with pytest.raises(ValueError):
        TimeSeriesSplit(train=np.array([], dtype=int), test=np.array([1]))
    with pytest.raises(ValueError):
        TimeSeriesSplit(train=np.array([0, 1]), test=np.array([1, 2]))
    with pytest.raises(ValueError):
        purged_kfold(4, n_splits=2, purge=10)
    with pytest.raises(ValueError):
        walk_forward_splits(10, train_size=8, test_size=3)
    with pytest.raises(ValueError):
        combinatorial_purged_splits(10, n_groups=1)
    with pytest.raises(ValueError):
        combinatorial_purged_splits(10, n_groups=4, test_groups=4)
    with pytest.raises(ValueError):
        combinatorial_purged_splits(8, n_groups=2, test_groups=1, purge=8)


def test_multiple_testing_guards() -> None:
    assert expected_maximum_sharpe([0.1, 0.1, 0.1]) == 0.0
    with pytest.raises(ValueError):
        sidak_adjusted_pvalue(-0.1, n_tests=2)
    with pytest.raises(ValueError):
        sidak_adjusted_pvalue(0.1, n_tests=0)


def test_pbo_guards() -> None:
    matrix = np.ones((10, 2))
    with pytest.raises(ValueError):
        probability_backtest_overfitting(matrix, n_slices=3)
    with pytest.raises(ValueError):
        probability_backtest_overfitting(matrix, n_slices=12)
    bad = matrix.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        probability_backtest_overfitting(bad)
    with pytest.raises(ValueError):
        probability_backtest_overfitting(np.ones((8, 2)), n_slices=8, max_combinations=10)


def test_stability_and_fit_guards() -> None:
    with pytest.raises(ValueError):
        complexity_penalized_score(1.0, complexity=-1.0, penalty_strength=0.1)
    with pytest.raises(ValueError):
        parameter_surface_stability([[1.0, np.nan], [1.0, 1.0]])
    with pytest.raises(ValueError):
        parameter_surface_stability([[1.0, 1.0], [1.0, 1.0]], tolerance=1.0)
    with pytest.raises(ValueError):
        parameter_surface_stability(
            [[1.0, 1.0], [1.0, 1.0]], target_plateau_fraction=0.0
        )
    with pytest.raises(ValueError):
        diagnose_fit(train_score=np.inf, validation_score=1.0)
    with pytest.raises(ValueError):
        diagnose_fit(train_score=1.0, validation_score=1.0, min_train_score=-1.0)
    assert diagnose_fit(train_score=0.1, validation_score=0.3) is FitDiagnosis.INCONCLUSIVE


def test_walk_forward_selection_guards() -> None:
    split = TimeSeriesSplit(train=np.array([0, 1]), test=np.array([2, 3]))
    with pytest.raises(ValueError):
        walk_forward_parameter_selection([0.1, 0.2, 0.3], [split])
    with pytest.raises(ValueError):
        walk_forward_parameter_selection(np.ones((4, 2)), [])
    out_of_bounds = TimeSeriesSplit(train=np.array([0, 1]), test=np.array([4, 5]))
    with pytest.raises(ValueError):
        walk_forward_parameter_selection(np.ones((4, 2)), [out_of_bounds])
    too_short = TimeSeriesSplit(train=np.array([0]), test=np.array([1]))
    with pytest.raises(ValueError):
        walk_forward_parameter_selection(np.ones((4, 2)), [too_short])


def test_right_fit_policy_and_failure_paths() -> None:
    with pytest.raises(ValueError):
        RightFitPolicy(max_pbo=1.2)
    with pytest.raises(ValueError):
        RightFitPolicy(max_train_test_degradation=np.inf)
    with pytest.raises(ValueError):
        RightFitPolicy(max_train_test_degradation=-0.1)
    with pytest.raises(ValueError):
        evaluate_right_fit(
            train_score=1.0,
            test_score=0.8,
            pbo=1.2,
            deflated_sharpe_probability=0.99,
            parameter_stability=0.8,
            bootstrap_positive_probability=0.9,
            leakage_detected=False,
        )
    result = evaluate_right_fit(
        train_score=1.0,
        test_score=-0.1,
        pbo=0.2,
        deflated_sharpe_probability=0.5,
        parameter_stability=0.2,
        bootstrap_positive_probability=0.2,
        leakage_detected=False,
    )
    assert result.decision is RightFitDecision.REJECT
    assert len(result.reasons) >= 4
