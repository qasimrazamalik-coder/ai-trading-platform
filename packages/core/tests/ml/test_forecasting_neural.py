import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from quant_os_core.agents.exits import Bar
from quant_os_core.forecasting import (
    ARModel,
    Garch11,
    HoltLinear,
    diebold_mariano,
    ewma_variance,
    horizon_volatility,
    qlike,
    rolling_origin,
)
from quant_os_core.ml.data_science import (
    data_quality,
    feature_diagnostics,
    render_report,
    return_profile,
)
from quant_os_core.ml.features import FEATURE_NAMES, build_features, ready_rows
from quant_os_core.ml.neural_signal import (
    EventDataset,
    NeuralConfidenceSignal,
    auc,
    build_event_dataset,
    train_confidence_model,
)
from quant_os_core.ml.nn import ConfidenceModel, MLPClassifier, PlattScaler
from quant_os_core.ops.signals import DonchianBreakout, ForecastBreakout

RNG = np.random.default_rng


def _garch_series(n: int, seed: int) -> np.ndarray:
    rng, (w, a, b) = RNG(seed), (2e-6, 0.08, 0.90)
    r, v = np.empty(n), w / (1 - a - b)
    for t in range(n):
        r[t] = np.sqrt(v) * rng.standard_normal()
        v = w + a * r[t] ** 2 + b * v
    return r


def test_garch_recovers_parameters_and_beats_ewma_and_constant() -> None:
    r = _garch_series(3000, 1)
    g = Garch11.fit(r)
    assert g.alpha == pytest.approx(0.08, abs=0.03) and g.beta == pytest.approx(0.90, abs=0.04)
    realized = r[1:] ** 2
    garch_loss = qlike(realized, g.conditional_variance(r)[:-1])
    assert garch_loss < qlike(realized, ewma_variance(r)[:-1])
    assert garch_loss < qlike(realized, np.full(realized.size, r.var()))
    path = g.forecast(250)
    assert path[-1] == pytest.approx(g.long_run_variance, rel=0.05)  # mean reversion
    assert horizon_volatility([1e-4] * 4) == pytest.approx(0.02)


def test_ar_finds_real_dependence_and_no_skill_on_noise() -> None:
    rng = RNG(2)
    y = np.zeros(2000)
    for t in range(1, y.size):
        y[t] = 0.3 * y[t - 1] + 0.01 * rng.standard_normal()
    fit = ARModel.fit(y, 1)
    assert fit.coefficients[0] == pytest.approx(0.3, abs=0.06)

    def model(h: np.ndarray) -> float:
        return float(ARModel.fit(h[-300:], 1).forecast(1).mean[0])

    real = rolling_origin(y, model, min_train=300, step=5)
    assert real.skill > 0 and real.dm_p_value < 0.05
    noise = rolling_origin(0.01 * rng.standard_normal(1500), model, min_train=300, step=5)
    assert noise.skill < 0.02 and noise.dm_p_value > 0.01
    band = fit.forecast(5)
    assert (band.upper - band.lower)[-1] > (band.upper - band.lower)[0]
    with pytest.raises(ValueError):
        diebold_mariano([1.0] * 5, [1.0] * 5)


def test_holt_tracks_a_trend() -> None:
    prices = 100 * np.exp(0.001 * np.arange(300) + 0.002 * RNG(3).standard_normal(300))
    forecast = HoltLinear.fit(prices).forecast(10)
    assert forecast.mean[-1] > prices[-1] and forecast.lower[0] < forecast.mean[0]


def _market(n: int, seed: int, learnable: bool) -> list[Bar]:
    rng, p, regime, out = RNG(seed), 100.0, 0, []
    t0 = datetime(2010, 1, 1, tzinfo=UTC)
    for t in range(n):
        if rng.random() < 0.02:
            regime = 1 - regime
        mu, sd = ((0.0025, 0.008) if regime else (-0.0008, 0.02)) if learnable else (0.0003, 0.014)
        r = mu + sd * rng.standard_normal()
        new = p * np.exp(r)
        hi = max(p * np.exp(abs(r) + sd * 0.5 * rng.random()), p, new)
        lo = min(p, new) * np.exp(-sd * 0.5 * rng.random())
        p = new
        out.append(
            Bar(
                t0 + timedelta(days=t),
                Decimal(f"{hi:.4f}"),
                Decimal(f"{lo:.4f}"),
                Decimal(f"{p:.4f}"),
            )
        )
    return out


def test_features_never_look_ahead() -> None:
    bars = _market(400, 4, True)
    h, lo, c = (np.array([float(getattr(b, k)) for b in bars]) for k in ("high", "low", "close"))
    full = build_features(h, lo, c)
    c2, h2, lo2 = c.copy(), h.copy(), lo.copy()
    c2[300:] *= 1.5  # rewrite the future
    h2[300:] *= 1.5
    lo2[300:] *= 1.5
    changed = build_features(h2, lo2, c2)
    assert np.allclose(full[:300], changed[:300], equal_nan=True)
    assert full.shape[1] == len(FEATURE_NAMES) and ready_rows(full)[260:].all()


def test_network_gradients_learning_and_persistence(tmp_path: Path) -> None:
    rng = RNG(0)
    m = MLPClassifier(hidden=(5, 3), dropout=0.0, l2=0.0)
    x, y = rng.normal(size=(8, 4)), (rng.random(8) > 0.5).astype(float)
    sizes = [4, 5, 3, 1]
    m.weights = [rng.normal(size=(a, b)) * 0.5 for a, b in pairwise(sizes)]
    m.biases = [rng.normal(size=b) * 0.1 for b in sizes[1:]]
    p, acts, _ = m._forward(x, None)
    delta = ((p - y) / 8)[:, None]
    grads = []
    for layer in range(2, -1, -1):
        grads.append(acts[layer].T @ delta)
        if layer:
            delta = (delta @ m.weights[layer].T) * (acts[layer] > 0)
    numeric = np.zeros((4, 5))
    for i in range(4):
        for j in range(5):
            m.weights[0][i, j] += 1e-6
            up = m._loss(m._forward(x, None)[0], y)
            m.weights[0][i, j] -= 2e-6
            down = m._loss(m._forward(x, None)[0], y)
            m.weights[0][i, j] += 1e-6
            numeric[i, j] = (up - down) / 2e-6
    assert np.abs(numeric - grads[-1]).max() < 1e-7  # backprop matches finite differences

    n = 3000
    xs = rng.normal(size=(n, 6))
    logit = 2.5 * xs[:, 0] * xs[:, 1] + xs[:, 2]  # an interaction a linear model cannot see
    ys = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    net = MLPClassifier().fit(xs[:2400], ys[:2400])
    linear = MLPClassifier(hidden=()).fit(xs[:2400], ys[:2400])
    auc_net = auc(ys[2400:], net.predict_proba(xs[2400:]))
    assert auc_net > auc(ys[2400:], linear.predict_proba(xs[2400:])) + 0.08

    bundle = ConfidenceModel(net, linear, PlattScaler(), FEATURE_NAMES[:6])
    bundle.save(tmp_path / "m" / "model.npz")
    loaded = ConfidenceModel.load(tmp_path / "m" / "model.npz")
    assert np.allclose(loaded.probability(xs[:5]), bundle.probability(xs[:5]))
    with pytest.raises(ValueError):
        MLPClassifier().fit(xs[:10], ys[:10])


def test_platt_scaler_fixes_overconfidence() -> None:
    rng = RNG(5)
    true_p = rng.uniform(0.2, 0.8, 4000)
    y = (rng.random(4000) < true_p).astype(float)
    overconfident = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))
    fixed = PlattScaler().fit(overconfident, y).transform(overconfident)
    assert np.abs(fixed - true_p).mean() < np.abs(overconfident - true_p).mean() / 2


def test_data_science_checks() -> None:
    bars = _market(600, 6, True)
    times = [b.time for b in bars]
    h, lo, c = (np.array([float(getattr(b, k)) for b in bars]) for k in ("high", "low", "close"))
    assert data_quality(times, h, lo, c).ok
    c_bad = c.copy()
    c_bad[100:106] = c_bad[100]  # stale run
    c_bad[300] = c_bad[299] * 3  # spike
    report = data_quality(times, np.maximum(h, c_bad), np.minimum(lo, c_bad), c_bad)
    assert report.stale_runs == 1 and 300 in report.outliers and not report.ok
    profile = return_profile(c)
    rng = RNG(7)
    x = rng.normal(size=(500, 3))
    labels = (x[:, 0] + 0.3 * rng.normal(size=500) > 0).astype(float)
    x[:, 2] = x[:, 0] * 2 + 0.01 * rng.normal(size=500)
    diag = feature_diagnostics(x, labels, ["signal", "noise", "copy"])
    assert diag.diagnostics[0].name in {"signal", "copy"} and diag.diagnostics[0].stable
    assert ("signal", "copy", pytest.approx(1.0, abs=0.01)) in [
        (a, b, r) for a, b, r in diag.redundant_pairs
    ]
    text = render_report("Test", data_quality(times, h, lo, c), profile, diag)
    assert "## Feature value" in text and "| signal |" in text


@pytest.fixture(scope="module")
def learnable() -> EventDataset:
    return build_event_dataset("X", _market(9000, 3, True), DonchianBreakout(), max_holding=20)


def test_neural_layer_learns_real_structure(learnable: EventDataset) -> None:
    model, report = train_confidence_model(learnable)
    assert report.passed and model is not None, report.reasons
    assert report.auc_model > 0.6 and report.brier_model < report.brier_base_rate
    best_floor = report.precision_at[f"{report.chosen_threshold:.2f}"]
    assert best_floor > report.base_rate + 0.03  # filtered trades win more often
    assert report.mean_return_filtered > report.mean_return_all
    assert np.all(np.diff(learnable.entry_index) >= 0)  # time ordered


def test_neural_layer_refuses_pure_noise() -> None:
    data = build_event_dataset("N", _market(9000, 3, False), DonchianBreakout(), max_holding=20)
    model, report = train_confidence_model(data)
    assert model is None and not report.passed and report.reasons


def test_live_signal_uses_calibrated_probability(learnable: EventDataset) -> None:
    model, _ = train_confidence_model(learnable)
    assert model is not None
    bars = _market(9000, 3, True)
    signal = NeuralConfidenceSignal(DonchianBreakout(), model, min_probability=0.0)
    produced = [signal("X", bars[: t + 1]) for t in range(8500, 9000)]
    plans = [p for p in produced if p is not None]
    assert plans and all(Decimal("0") < p.confidence < Decimal("1") for p in plans)
    assert all("neural P(target first)" in p.justification for p in plans)
    strict = NeuralConfidenceSignal(DonchianBreakout(), model, min_probability=0.999)
    assert all(strict("X", bars[: t + 1]) is None for t in range(8500, 9000))


def test_forecast_breakout_sets_stops_from_volatility_forecast() -> None:
    calm = _market(400, 8, True)
    plan = None
    for t in range(300, 400):
        plan = ForecastBreakout()("X", calm[: t + 1]) or plan
    assert plan is not None and plan.stop_loss < plan.entry_price < plan.profit_target
    ratio = (plan.profit_target - plan.entry_price) / (plan.entry_price - plan.stop_loss)
    assert ratio == pytest.approx(Decimal("2"), abs=Decimal("0.001"))


def test_trainer_cli_saves_a_gated_champion_and_runner_loads_it(tmp_path: Path) -> None:
    from quant_os_core.ml.train_confidence import main as train
    from quant_os_core.ops.run_paper import _signal

    csv = tmp_path / "SYN.csv"
    rows = ["time,high,low,close"] + [
        f"{b.time.isoformat()},{b.high},{b.low},{b.close}" for b in _market(9000, 3, True)
    ]
    csv.write_text("\n".join(rows) + "\n")
    out, state, report = tmp_path / "models/c.npz", tmp_path / "s.db", tmp_path / "r.md"
    code = train(
        ["--csv", str(csv), "--out", str(out), "--state", str(state), "--report", str(report)]
    )
    assert code == 0 and out.exists()
    text = report.read_text()
    assert "PASSED" in text and "## Feature value across all symbols" in text
    signal = _signal({"model": {"path": str(out)}})
    assert isinstance(signal, NeuralConfidenceSignal)
    # retraining on the same data does not beat the champion by a margin -> champion kept
    second = train(
        [
            "--csv",
            str(csv),
            "--out",
            str(tmp_path / "models/c2.npz"),
            "--state",
            str(state),
            "--report",
            str(report),
        ]
    )
    assert second == 0 and not (tmp_path / "models/c2.npz").exists()
    assert "kept the current champion" in report.read_text()
    assert json.loads(json.dumps({"ok": True}))["ok"]
