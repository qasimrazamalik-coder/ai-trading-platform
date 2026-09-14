import numpy as np

from quant_os_core.right_fit import (
    FitDiagnosis,
    diagnose_fit,
    walk_forward_parameter_selection,
    walk_forward_splits,
)


def test_walk_forward_selection_uses_training_only() -> None:
    x = np.linspace(0.0, 12.0, 120)
    base = 0.002 * np.sin(x)
    matrix = np.column_stack([base.copy(), base.copy(), base.copy()])
    matrix[:60, 0] += 0.01
    matrix[60:, 1] += 0.01
    matrix[:, 2] -= 0.002
    splits = walk_forward_splits(
        120,
        train_size=40,
        test_size=20,
        step_size=20,
        expanding=False,
    )
    result = walk_forward_parameter_selection(matrix, splits)
    assert result.selected_candidate_indices[0] == 0
    assert result.selected_candidate_indices.shape == result.test_scores.shape
    assert 0.0 <= result.positive_test_fraction <= 1.0


def test_diagnose_underfit() -> None:
    assert diagnose_fit(train_score=0.1, validation_score=0.08) is FitDiagnosis.UNDERFIT


def test_diagnose_overfit() -> None:
    assert diagnose_fit(train_score=1.5, validation_score=0.2) is FitDiagnosis.OVERFIT


def test_diagnose_right_fit() -> None:
    assert diagnose_fit(train_score=1.2, validation_score=1.0) is FitDiagnosis.RIGHT_FIT
