import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quant_os_core.forecasting.deep_forecast import (  # noqa: E402
    Comparison,
    Fold,
    ModelScore,
    NBeatsForecaster,
    Target,
    TCNForecaster,
    TrainConfig,
    baseline_forecasts,
    compare_forecasters,
    complexity_from_runs,
    default_specs,
    lstm_spec,
    make_trainer,
    make_windows,
    nbeats_spec,
    plan_folds,
    tcn_spec,
    train_deep,
    walk_forward_deep,
)
from quant_os_core.right_fit.fit_check import FitVerdict, learning_curve  # noqa: E402

FAST = TrainConfig(lookback=8, epochs=3, batch_size=64, patience=2, max_train=300)


def _garch_like(n=600, seed=0):
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    var = 1e-4
    for t in range(n):
        r[t] = math.sqrt(var) * rng.standard_normal()
        var = 1e-6 + 0.1 * r[t] ** 2 + 0.85 * var
    return r


def test_windows_never_include_target_day():
    r = np.arange(1, 61, dtype=float) / 1000
    x, y = make_windows(r, 5, Target.RETURN)
    assert x.shape == (55, 5, 2) and y.shape == (55,)
    assert x[0, -1, 0] == r[4] and y[0] == r[5]
    assert np.allclose(x[..., 1], np.abs(x[..., 0]))
    _, yv = make_windows(r, 5, Target.VARIANCE)
    assert yv[0] == pytest.approx(r[5] ** 2)
    with pytest.raises(ValueError):
        make_windows(r, 0, Target.RETURN)
    with pytest.raises(ValueError):
        make_windows(np.ones(10), 3, Target.RETURN)


def test_model_shapes():
    x = torch.randn(4, 8, 2)
    assert TCNForecaster(2, channels=4, levels=2)(x).shape == (4,)
    assert TCNForecaster(4, channels=4, levels=1)(torch.randn(3, 8, 4)).shape == (3,)
    assert NBeatsForecaster(2, 8, hidden=4, blocks=2)(x).shape == (4,)
    with pytest.raises(ValueError):
        TCNForecaster(2, kernel_size=1)
    with pytest.raises(ValueError):
        NBeatsForecaster(2, 0)


def test_tcn_is_causal():
    torch.manual_seed(0)
    model = TCNForecaster(1, channels=4, levels=2)
    model.eval()
    x = torch.randn(1, 10, 1)
    hidden = model.net(x.transpose(1, 2))
    x2 = x.clone()
    x2[0, -1, 0] += 5.0  # changing the last step must not change earlier outputs
    hidden2 = model.net(x2.transpose(1, 2))
    assert torch.allclose(hidden[:, :, :-1], hidden2[:, :, :-1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookback": 1},
        {"learning_rate": 0.0},
        {"validation_fraction": 0.6},
        {"max_train": 5},
    ],
)
def test_config_guards(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_training_is_seeded_and_scales_on_train_rows():
    r = _garch_like()
    x, y = make_windows(r, 8, Target.VARIANCE)
    a = train_deep(tcn_spec(4, 1), x, y, Target.VARIANCE, FAST)
    b = train_deep(tcn_spec(4, 1), x, y, Target.VARIANCE, FAST)
    assert np.allclose(a(x[-20:]), b(x[-20:]))
    assert (a(x[-20:]) > 0).all()
    assert 1 <= a.epochs_run <= FAST.epochs and len(a.history) == a.epochs_run
    # scaler excludes the most recent (validation) rows: corrupting them changes nothing
    x_bad = x[-300:].copy()
    x_bad[-10:] *= 100.0
    c = train_deep(tcn_spec(4, 1), x[-300:], y[-300:], Target.VARIANCE, FAST)
    d = train_deep(tcn_spec(4, 1), x_bad, y[-300:], Target.VARIANCE, FAST)
    assert np.allclose(c.x_mean, d.x_mean)


def test_early_stopping_and_default_specs():
    x = np.random.default_rng(2).normal(size=(200, 8, 2))
    y = np.random.default_rng(3).normal(size=200)
    cfg = TrainConfig(lookback=8, epochs=50, batch_size=32, patience=1, learning_rate=0.05)
    model = train_deep(nbeats_spec(16, 2), x, y, Target.RETURN, cfg)
    assert model.epochs_run < 50
    assert [s.name for s in default_specs()] == ["LSTM-16", "TCN-16", "NBEATS-32"]


def test_train_deep_guards():
    with pytest.raises(ValueError):
        train_deep(lstm_spec(4), np.ones((30, 8)), np.ones(30), Target.RETURN, FAST)
    with pytest.raises(ValueError):
        train_deep(lstm_spec(4), np.ones((20, 8, 2)), np.ones(20), Target.RETURN, FAST)


def test_constant_target_does_not_divide_by_zero():
    x = np.random.default_rng(0).normal(size=(80, 8, 2))
    model = train_deep(nbeats_spec(4, 1), x, np.zeros(80), Target.RETURN, FAST)
    assert np.isfinite(model(x)).all()


def test_plan_folds():
    folds = plan_folds(200, min_train=100, folds=3)
    assert folds[0] == Fold(100, 133) and folds[-1].test_end == 200
    with pytest.raises(ValueError):
        plan_folds(100, min_train=95, folds=2)


def test_baselines_use_only_past_data():
    r = _garch_like(400)
    folds = plan_folds(r.size, min_train=200, folds=2)
    var = baseline_forecasts(r, Target.VARIANCE, folds)
    assert set(var) == {"naive-last", "EWMA", "GARCH(1,1)"}
    assert all(v.size == 200 for v in var.values())
    assert var["naive-last"][0] == pytest.approx(r[199] ** 2)
    # changing a future return leaves earlier forecasts unchanged
    r2 = r.copy()
    r2[350] = 0.5
    var2 = baseline_forecasts(r2, Target.VARIANCE, folds)
    for name in var:
        assert np.allclose(var[name][: 350 - 200 + 1], var2[name][: 350 - 200 + 1])
    ret = baseline_forecasts(r, Target.RETURN, folds)
    assert set(ret) == {"naive-zero", "AR(1)"} and (ret["naive-zero"] == 0).all()


def test_walk_forward_and_fit_report():
    r = _garch_like(500)
    folds = plan_folds(r.size, min_train=300, folds=2)
    run = walk_forward_deep(r, tcn_spec(4, 1), Target.VARIANCE, folds, FAST)
    assert run.index[0] == 300 and run.index[-1] == 499 and run.forecasts.size == 200
    assert len(run.folds) == 2 and run.parameters > 0 and run.seconds >= 0
    report = run.fit_report()
    assert report.verdict in set(FitVerdict)
    with pytest.raises(ValueError):
        walk_forward_deep(r, tcn_spec(4, 1), Target.VARIANCE, [Fold(20, 40)], FAST)
    with pytest.raises(ValueError):
        walk_forward_deep(r, tcn_spec(4, 1), Target.VARIANCE, [Fold(300, 900)], FAST)


@pytest.mark.parametrize("target", [Target.RETURN, Target.VARIANCE])
def test_compare_forecasters_table(target):
    r = _garch_like(520, seed=1)
    comp = compare_forecasters(
        r,
        target,
        asset="SYN",
        specs=[lstm_spec(4), tcn_spec(4, 1), nbeats_spec(4, 1)],
        config=FAST,
        folds=2,
        min_train=300,
    )
    assert isinstance(comp, Comparison) and comp.n_forecasts == 220
    kinds = [s.kind for s in comp.scores]
    assert kinds.count("deep") == 3 and kinds.count("naive") == 1
    table = comp.table()
    assert "SYN" in table and "verdict:" in table and "LSTM-4" in table
    naive = next(s for s in comp.scores if s.kind == "naive")
    assert naive.skill == 0.0 and math.isnan(naive.directional_accuracy)
    if target is Target.RETURN:
        assert all(math.isnan(s.qlike) for s in comp.scores)
    else:
        assert all(s.qlike > 0 for s in comp.scores)
    assert isinstance(comp.deep_beats_baselines(), bool)
    sweep = complexity_from_runs(comp.runs)
    assert sweep.chosen.complexity <= sweep.best.complexity or sweep.chosen is sweep.best


def _score(name, kind, rmse, dm=None):
    return ModelScore(name, kind, rmse, rmse, 0.5, math.nan, 0.0, (0.0, 1.0), dm)


def test_deep_beats_baselines_requires_significance():
    base = [_score("naive-zero", "naive", 1.0), _score("AR(1)", "baseline", 0.9)]
    win = Comparison(
        "A", Target.RETURN, (*base, _score("D", "deep", 0.5, (-3.0, 0.001))), (), 100, "AR(1)"
    )
    assert win.deep_beats_baselines()
    assert "beats every baseline" in win.table()
    lucky = Comparison(
        "A", Target.RETURN, (*base, _score("D", "deep", 0.8, (-1.0, 0.3))), (), 100, "AR(1)"
    )
    assert not lucky.deep_beats_baselines()


def test_trainer_adapter_works_with_learning_curve():
    r = _garch_like(400)
    x, y = make_windows(r, 8, Target.VARIANCE)
    curve = learning_curve(
        x,
        y,
        make_trainer(tcn_spec(4, 1), Target.VARIANCE, FAST),
        fractions=(0.5, 1.0),
        min_train=60,
    )
    assert len(curve.points) == 2
