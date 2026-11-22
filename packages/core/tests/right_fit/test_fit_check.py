import math

import numpy as np
import pytest

from quant_os_core.right_fit.fit_check import (
    Candidate,
    FitVerdict,
    complexity_sweep,
    fit_check,
    fit_verdict,
    generalization_gap,
    learning_curve,
    mean_trainer,
    mse,
    one_standard_error_rule,
    walk_forward_errors,
    zero_trainer,
)


def _linear_data(n=300, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = x @ np.array([1.0, -0.5, 0.25]) + noise * rng.normal(size=n)
    return x, y


def ols_trainer(x, y):
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return lambda f: np.column_stack([np.ones(len(f)), f]) @ beta


def memoriser(x, y):
    """1-nearest-neighbour: perfect in sample, noisy out of sample."""

    def predict(f):
        d = ((f[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
        return y[d.argmin(axis=1)]

    return predict


def test_mse_and_shape_checks():
    assert mse([1.0, 2.0], [1.0, 4.0]) == 2.0
    with pytest.raises(ValueError):
        mse([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        mse([], [])


def test_generalization_gap_values_and_guards():
    gap = generalization_gap(1.0, 1.5)
    assert gap.absolute == pytest.approx(0.5)
    assert gap.relative == pytest.approx(0.5)
    assert generalization_gap(0.0, 1.0).relative == math.inf
    assert generalization_gap(0.0, 0.0).relative == 0.0
    with pytest.raises(ValueError):
        generalization_gap(-1.0, 1.0)
    with pytest.raises(ValueError):
        generalization_gap(math.nan, 1.0)


def test_learning_curve_uses_fixed_recent_validation_slice():
    x, y = _linear_data()
    seen = []

    def spy_trainer(xt, yt):
        seen.append(len(xt))
        return ols_trainer(xt, yt)

    curve = learning_curve(x, y, spy_trainer, fractions=(0.25, 0.5, 1.0))
    assert curve.validation_size == 60
    assert [p.n_train for p in curve.points] == [60, 120, 240] == seen
    assert curve.final_gap.train_error == curve.points[-1].train_error
    assert all(p.validation_error < 0.05 for p in curve.points)


def test_learning_curve_improving_flag():
    x, y = _linear_data(noise=1.0)
    curve = learning_curve(x, y, memoriser, fractions=(0.1, 1.0), min_train=5)
    assert isinstance(curve.validation_still_improving, bool)
    single = learning_curve(x, y, ols_trainer, fractions=(1.0,))
    assert single.validation_still_improving is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_fraction": 0.0},
        {"fractions": ()},
        {"fractions": (1.5,)},
        {"min_train": 1000},
    ],
)
def test_learning_curve_rejects_bad_settings(kwargs):
    x, y = _linear_data()
    with pytest.raises(ValueError):
        learning_curve(x, y, ols_trainer, **kwargs)


def test_learning_curve_rejects_bad_data():
    with pytest.raises(ValueError):
        learning_curve(np.ones(10), np.ones(10), ols_trainer)
    x = np.ones((30, 2))
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        learning_curve(x, np.ones(30), ols_trainer)


def test_one_standard_error_rule_prefers_simpler_model():
    sweep = one_standard_error_rule(
        [
            Candidate("small", 1, (1.02, 1.00, 0.98, 1.05)),
            Candidate("medium", 2, (0.99, 0.97, 0.96, 1.00)),
            Candidate("large", 3, (0.90, 1.10, 0.80, 1.05)),
        ]
    )
    assert sweep.best.name == "large"
    assert sweep.chosen.name == "small"
    assert sweep.threshold == pytest.approx(sweep.best.mean_error + sweep.best.standard_error)


def test_one_standard_error_rule_guards():
    with pytest.raises(ValueError):
        one_standard_error_rule([])
    with pytest.raises(ValueError):
        one_standard_error_rule([Candidate("a", 1, (1.0,)), Candidate("b", 2, (1.0, 2.0))])
    with pytest.raises(ValueError):
        one_standard_error_rule([Candidate("a", 1, (-1.0,))])
    single = one_standard_error_rule([Candidate("a", 1, (1.0,))])
    assert single.chosen.standard_error == 0.0


def test_walk_forward_errors_and_complexity_sweep():
    x, y = _linear_data()
    errors = walk_forward_errors(x, y, ols_trainer, folds=3)
    assert len(errors) == 3 and all(e < 0.05 for e in errors)
    sweep = complexity_sweep(
        x,
        y,
        [("mean", 0, mean_trainer), ("ols", 4, ols_trainer), ("1nn", 300, memoriser)],
        folds=3,
    )
    assert sweep.chosen.name == "ols"
    with pytest.raises(ValueError):
        walk_forward_errors(x, y, ols_trainer, folds=1)
    with pytest.raises(ValueError):
        walk_forward_errors(x[:15], y[:15], ols_trainer, folds=4)


def test_verdicts():
    right = fit_verdict(
        train_error=0.5,
        validation_error=0.55,
        baseline_train_error=1.0,
        baseline_validation_error=1.0,
    )
    assert right.verdict is FitVerdict.RIGHT_FIT
    assert "right_fit" in right.summary()

    over = fit_verdict(
        train_error=0.1,
        validation_error=1.2,
        baseline_train_error=1.0,
        baseline_validation_error=1.0,
    )
    assert over.verdict is FitVerdict.OVERFIT
    assert over.validation_skill < 0

    under = fit_verdict(
        train_error=1.0,
        validation_error=1.0,
        baseline_train_error=1.0,
        baseline_validation_error=1.0,
    )
    assert under.verdict is FitVerdict.UNDERFIT

    weak = fit_verdict(
        train_error=0.9,
        validation_error=1.05,
        baseline_train_error=1.0,
        baseline_validation_error=1.0,
    )
    assert weak.verdict is FitVerdict.RIGHT_FIT
    assert "does not beat the baseline" in weak.reason

    zero_base = fit_verdict(
        train_error=0.0,
        validation_error=0.0,
        baseline_train_error=0.0,
        baseline_validation_error=0.0,
    )
    assert zero_base.train_skill == 0.0


def test_verdict_guards():
    with pytest.raises(ValueError):
        fit_verdict(
            train_error=-1,
            validation_error=1,
            baseline_train_error=1,
            baseline_validation_error=1,
        )
    with pytest.raises(ValueError):
        fit_verdict(
            train_error=1,
            validation_error=1,
            baseline_train_error=1,
            baseline_validation_error=1,
            max_relative_gap=-0.1,
        )


def test_fit_check_end_to_end():
    x, y = _linear_data(noise=0.5)
    good = fit_check(x, y, ols_trainer, baseline=mean_trainer)
    assert good.verdict is FitVerdict.RIGHT_FIT
    assert good.curve is not None and len(good.curve.points) == 3

    noise_x, noise_y = _linear_data(noise=5.0, seed=3)
    noise_y = np.random.default_rng(1).normal(size=noise_y.size)
    bad = fit_check(noise_x, noise_y, memoriser, baseline=zero_trainer)
    assert bad.verdict is FitVerdict.OVERFIT

    flat = fit_check(x, y, mean_trainer, baseline=mean_trainer)
    assert flat.verdict is FitVerdict.UNDERFIT


def test_summary_mentions_improving_curve():
    x, y = _linear_data(n=400, noise=0.3)
    report = fit_check(x, y, memoriser, baseline=mean_trainer, fractions=(0.05, 1.0))
    if report.curve is not None and report.curve.validation_still_improving:
        assert "still falling" in report.summary()
