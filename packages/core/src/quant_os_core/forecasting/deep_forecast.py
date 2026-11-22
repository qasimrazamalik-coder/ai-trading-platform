"""Deep-learning return and volatility forecasters, judged walk-forward against simple baselines.

Three small PyTorch models read the last ``lookback`` daily returns (two channels: the return
and its absolute value) and forecast the next day's return or squared return (a noisy but
unbiased variance proxy):

* ``LSTM``: the recurrent forecaster from ``ml.deep``.
* ``TCN``: a temporal convolutional network, causal dilated convolutions with residuals.
* ``N-BEATS``: an N-BEATS-style MLP, stacked blocks that each emit a backcast (removed from
  the input) and a partial forecast (summed), after Oreshkin et al. (2020).

Discipline, so a win would mean something:

1. Walk-forward: the test period is cut into consecutive folds; each fold is forecast by a
   model trained only on targets that come before it. No forecast sees its own future.
2. Scaling (inputs and target) is fitted on the training rows only, never on validation or
   test rows.
3. Early stopping watches the most recent slice of the training data, never the test fold.
4. Seeds fix weight initialisation and batch order, so a rerun gives the same numbers.
5. Every model is compared on the same days with naive forecasts (zero return; yesterday's
   squared return) and the existing baselines (AR(1) for returns; EWMA and GARCH(1,1) for
   variance): MAE, RMSE, directional accuracy, QLIKE for variance and the Diebold-Mariano test.

Daily returns are close to unpredictable. Expect the deep models to lose to the naive zero
forecast on returns and to struggle against EWMA/GARCH on variance; the report says so
plainly when they do. Research only: nothing here sends or sizes an order.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "PyTorch is required for quant_os_core.forecasting.deep_forecast; install quant-os[dl]"
    ) from exc

from quant_os_core.forecasting.evaluation import diebold_mariano, qlike
from quant_os_core.forecasting.models import ARModel
from quant_os_core.forecasting.volatility import Garch11, ewma_variance
from quant_os_core.ml.deep import LSTMForecaster
from quant_os_core.right_fit.fit_check import (
    Candidate,
    ComplexitySweep,
    FitReport,
    fit_verdict,
    one_standard_error_rule,
)

FloatArray = npt.NDArray[np.float64]
VARIANCE_FLOOR = 1e-10


class Target(StrEnum):
    RETURN = "return"
    VARIANCE = "variance"


# ---------------------------------------------------------------- models


class _CausalBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(ch_in, ch_out, kernel_size, dilation=dilation)
        self.skip: nn.Module = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = nn.functional.pad(x, (self.pad, 0))  # left pad only: causal
        conv: torch.Tensor = self.conv(padded)
        skip: torch.Tensor = self.skip(x)
        return torch.relu(conv) + skip


class TCNForecaster(nn.Module):
    """Causal dilated convolutions (dilation 1, 2, 4, ...) and a linear head on the last step."""

    def __init__(
        self, input_size: int, channels: int = 16, levels: int = 3, kernel_size: int = 3
    ) -> None:
        super().__init__()
        if channels < 1 or levels < 1 or kernel_size < 2:
            raise ValueError("channels and levels must be >= 1 and kernel_size >= 2")
        blocks = [
            _CausalBlock(input_size if i == 0 else channels, channels, kernel_size, 2**i)
            for i in range(levels)
        ]
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden: torch.Tensor = self.net(x.transpose(1, 2))
        out: torch.Tensor = self.head(hidden[:, :, -1])
        return out.squeeze(-1)


class _NBeatsBlock(nn.Module):
    def __init__(self, width: int, hidden: int, layers: int) -> None:
        super().__init__()
        stack: list[nn.Module] = []
        size = width
        for _ in range(layers):
            stack += [nn.Linear(size, hidden), nn.ReLU()]
            size = hidden
        self.body = nn.Sequential(*stack)
        self.backcast = nn.Linear(hidden, width)
        self.forecast = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h: torch.Tensor = self.body(x)
        back: torch.Tensor = self.backcast(h)
        fore: torch.Tensor = self.forecast(h)
        return back, fore


class NBeatsForecaster(nn.Module):
    """N-BEATS-style generic stack: each block explains part of the input and forecasts."""

    def __init__(
        self, input_size: int, lookback: int, hidden: int = 32, blocks: int = 3, layers: int = 2
    ) -> None:
        super().__init__()
        if lookback < 1 or hidden < 1 or blocks < 1 or layers < 1:
            raise ValueError("lookback, hidden, blocks and layers must be positive")
        width = input_size * lookback
        self.blocks = nn.ModuleList(_NBeatsBlock(width, hidden, layers) for _ in range(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.flatten(1)
        total = torch.zeros(x.shape[0], 1, dtype=x.dtype)
        for block in self.blocks:
            back, fore = block(residual)
            residual = residual - back
            total = total + fore
        return total.squeeze(-1)


Builder = Callable[[int, int], nn.Module]  # (input channels, lookback) -> module


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    build: Builder

    def parameter_count(self, input_size: int = 2, lookback: int = 20) -> int:
        module = self.build(input_size, lookback)
        return int(sum(p.numel() for p in module.parameters()))


def lstm_spec(hidden: int = 16) -> ModelSpec:
    return ModelSpec(f"LSTM-{hidden}", lambda c, _l: LSTMForecaster(c, hidden_size=hidden))


def tcn_spec(channels: int = 16, levels: int = 3) -> ModelSpec:
    return ModelSpec(f"TCN-{channels}", lambda c, _l: TCNForecaster(c, channels, levels))


def nbeats_spec(hidden: int = 32, blocks: int = 3) -> ModelSpec:
    return ModelSpec(
        f"NBEATS-{hidden}", lambda c, lookback: NBeatsForecaster(c, lookback, hidden, blocks)
    )


def default_specs() -> tuple[ModelSpec, ...]:
    return (lstm_spec(), tcn_spec(), nbeats_spec())


# ---------------------------------------------------------------- data


@dataclass(frozen=True, slots=True)
class TrainConfig:
    lookback: int = 20
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 6
    validation_fraction: float = 0.2
    max_train: int | None = 2000  # most recent training windows per fold (None = all)
    seed: int = 7

    def __post_init__(self) -> None:
        if self.lookback < 2 or self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("lookback >= 2; epochs, batch_size and patience >= 1")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be > 0 and weight_decay >= 0")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.max_train is not None and self.max_train < 20:
            raise ValueError("max_train must be at least 20")


def _returns(values: npt.ArrayLike) -> FloatArray:
    r = np.asarray(values, dtype=np.float64)
    if r.ndim != 1 or r.size < 50 or not np.isfinite(r).all():
        raise ValueError("returns must be a finite 1D array with at least 50 values")
    return r


def target_values(returns: npt.ArrayLike, target: Target) -> FloatArray:
    r = _returns(returns)
    return r.copy() if target is Target.RETURN else r * r


def make_windows(
    returns: npt.ArrayLike, lookback: int, target: Target
) -> tuple[FloatArray, FloatArray]:
    """``x[i]`` holds returns ``t-lookback .. t-1`` (and their absolute values); ``y[i]`` is
    the target at ``t = lookback + i``. Inputs never include the target day."""

    r = _returns(returns)
    if lookback < 1 or lookback >= r.size:
        raise ValueError("lookback must be in [1, len(returns))")
    view = np.lib.stride_tricks.sliding_window_view(r, lookback)[:-1]
    x = np.stack([view, np.abs(view)], axis=-1)
    y = target_values(r, target)[lookback:]
    return np.ascontiguousarray(x), y


# ---------------------------------------------------------------- training


class FittedDeep:
    """A trained network plus the train-only scalers; callable on raw windows."""

    def __init__(
        self,
        module: nn.Module,
        x_mean: FloatArray,
        x_scale: FloatArray,
        y_mean: float,
        y_scale: float,
        target: Target,
        epochs_run: int,
        history: tuple[tuple[float, float], ...],
    ) -> None:
        self.module = module
        self.x_mean, self.x_scale = x_mean, x_scale
        self.y_mean, self.y_scale = y_mean, y_scale
        self.target = target
        self.epochs_run = epochs_run
        self.history = history  # (train, validation) loss per epoch, scaled units

    def __call__(self, x: FloatArray) -> FloatArray:
        features = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_scale
        self.module.eval()
        with torch.no_grad():
            raw: torch.Tensor = self.module(torch.as_tensor(features, dtype=torch.float32))
        pred = raw.numpy().astype(np.float64) * self.y_scale + self.y_mean
        if self.target is Target.VARIANCE:
            # a variance forecast must be positive; below 1% of the training mean is noise
            pred = np.maximum(pred, max(VARIANCE_FLOOR, 0.01 * self.y_mean))
        return np.asarray(pred, dtype=np.float64)


def train_deep(
    spec: ModelSpec, x: npt.ArrayLike, y: npt.ArrayLike, target: Target, config: TrainConfig
) -> FittedDeep:
    """Fit on the older rows, early-stop on the newest ``validation_fraction`` of them."""

    features = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    if features.ndim != 3 or labels.shape != (features.shape[0],):
        raise ValueError("x must be [n, lookback, channels] and y must have n values")
    if config.max_train is not None and features.shape[0] > config.max_train:
        features, labels = features[-config.max_train :], labels[-config.max_train :]
    n_val = max(5, int(features.shape[0] * config.validation_fraction))
    if features.shape[0] - n_val < 20:
        raise ValueError("need at least 20 training windows after the validation slice")
    x_tr, y_tr = features[:-n_val], labels[:-n_val]
    x_va, y_va = features[-n_val:], labels[-n_val:]
    x_mean = x_tr.mean(axis=(0, 1))
    x_std = x_tr.std(axis=(0, 1))
    x_scale = np.where(x_std > 1e-12, x_std, 1.0)
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std())
    y_scale = y_std if y_std > 1e-12 else 1.0

    def tensor(a: FloatArray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32)

    xt, yt = tensor((x_tr - x_mean) / x_scale), tensor((y_tr - y_mean) / y_scale)
    xv, yv = tensor((x_va - x_mean) / x_scale), tensor((y_va - y_mean) / y_scale)

    torch.manual_seed(config.seed)
    module = spec.build(features.shape[2], features.shape[1])
    optimizer = torch.optim.Adam(
        module.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(config.seed)
    best_loss, best_state, stale = math.inf, None, 0
    history: list[tuple[float, float]] = []
    for _ in range(config.epochs):
        module.train()
        order = torch.randperm(xt.shape[0], generator=gen)  # shuffles training rows only
        total = 0.0
        for start in range(0, xt.shape[0], config.batch_size):
            idx = order[start : start + config.batch_size]
            optimizer.zero_grad()
            pred: torch.Tensor = module(xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * idx.numel()
        module.eval()
        with torch.no_grad():
            val_pred: torch.Tensor = module(xv)
            val_loss = float(loss_fn(val_pred, yv))
        history.append((total / xt.shape[0], val_loss))
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    return FittedDeep(
        module, x_mean, x_scale, y_mean, y_scale, target, len(history), tuple(history)
    )


def make_trainer(
    spec: ModelSpec, target: Target, config: TrainConfig
) -> Callable[[FloatArray, FloatArray], FittedDeep]:
    """Adapter with the ``right_fit.fit_check.Trainer`` signature (learning curves, sweeps)."""

    return lambda x, y: train_deep(spec, x, y, target, config)


# ---------------------------------------------------------------- walk-forward


@dataclass(frozen=True, slots=True)
class Fold:
    test_start: int  # index into the return series; training targets end before it
    test_end: int  # exclusive


def plan_folds(n_returns: int, *, min_train: int, folds: int) -> tuple[Fold, ...]:
    if folds < 1 or min_train < 50 or n_returns - min_train < folds * 10:
        raise ValueError("need min_train >= 50 and at least 10 test days per fold")
    block = (n_returns - min_train) // folds
    out = []
    for k in range(folds):
        lo = min_train + k * block
        out.append(Fold(lo, n_returns if k == folds - 1 else lo + block))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class FoldFit:
    train_mse: float
    test_mse: float
    baseline_train_mse: float
    baseline_test_mse: float
    epochs_run: int


@dataclass(frozen=True, slots=True)
class DeepRun:
    name: str
    target: Target
    index: npt.NDArray[np.int64]  # return-series positions of the forecasts
    forecasts: FloatArray
    folds: tuple[FoldFit, ...]
    parameters: int
    seconds: float

    def fit_report(self) -> FitReport:
        """Verdict from fold-averaged train error vs out-of-sample error, each against the
        naive baseline (zero for returns, the training mean for variance) on the same rows."""

        return fit_verdict(
            train_error=float(np.mean([f.train_mse for f in self.folds])),
            validation_error=float(np.mean([f.test_mse for f in self.folds])),
            baseline_train_error=float(np.mean([f.baseline_train_mse for f in self.folds])),
            baseline_validation_error=float(np.mean([f.baseline_test_mse for f in self.folds])),
        )


def walk_forward_deep(
    returns: npt.ArrayLike,
    spec: ModelSpec,
    target: Target,
    folds: Sequence[Fold],
    config: TrainConfig | None = None,
) -> DeepRun:
    cfg = config or TrainConfig()
    r = _returns(returns)
    x, y = make_windows(r, cfg.lookback, target)
    started = time.perf_counter()
    preds: list[FloatArray] = []
    index: list[npt.NDArray[np.int64]] = []
    fits: list[FoldFit] = []
    for fold in folds:
        if fold.test_start - cfg.lookback < 30 or fold.test_end > r.size:
            raise ValueError("fold leaves too little training data or runs past the series")
        lo, hi = fold.test_start - cfg.lookback, fold.test_end - cfg.lookback
        model = train_deep(spec, x[:lo], y[:lo], target, cfg)
        n_seen = min(lo, cfg.max_train) if cfg.max_train is not None else lo
        n_fit = n_seen - max(5, int(n_seen * cfg.validation_fraction))
        y_fit = y[lo - n_seen : lo - n_seen + n_fit]
        in_sample = model(x[lo - n_seen : lo - n_seen + n_fit])
        out = model(x[lo:hi])
        base_level = 0.0 if target is Target.RETURN else float(y_fit.mean())
        fits.append(
            FoldFit(
                train_mse=float(np.mean((y_fit - in_sample) ** 2)),
                test_mse=float(np.mean((y[lo:hi] - out) ** 2)),
                baseline_train_mse=float(np.mean((y_fit - base_level) ** 2)),
                baseline_test_mse=float(np.mean((y[lo:hi] - base_level) ** 2)),
                epochs_run=model.epochs_run,
            )
        )
        preds.append(out)
        index.append(np.arange(fold.test_start, fold.test_end, dtype=np.int64))
    return DeepRun(
        name=spec.name,
        target=target,
        index=np.concatenate(index),
        forecasts=np.concatenate(preds),
        folds=tuple(fits),
        parameters=spec.parameter_count(x.shape[2], cfg.lookback),
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------- baselines


def baseline_forecasts(
    returns: npt.ArrayLike, target: Target, folds: Sequence[Fold]
) -> dict[str, FloatArray]:
    """Forecasts for every test day, each made from data before that day.

    Returns: ``naive`` (zero) and ``AR(1)`` refitted per fold on prior data. Variance:
    ``naive`` (yesterday's squared return), ``EWMA`` (lambda 0.94) and ``GARCH(1,1)``
    refitted per fold and filtered forward with the fold's own parameters and mean.
    """

    r = _returns(returns)
    out: dict[str, list[float]] = {}
    if target is Target.RETURN:
        out = {"naive-zero": [], "AR(1)": []}
        for fold in folds:
            ar = ARModel.fit(r[: fold.test_start], order=1)
            for t in range(fold.test_start, fold.test_end):
                out["naive-zero"].append(0.0)
                out["AR(1)"].append(ar.intercept + float(ar.coefficients[0]) * r[t - 1])
    else:
        ewma = ewma_variance(r)  # ewma[t-1] uses returns up to t-1 only
        out = {"naive-last": [], "EWMA": [], "GARCH(1,1)": []}
        for fold in folds:
            train = r[: fold.test_start]
            garch = Garch11.fit(train)
            mu = float(train.mean())
            var = garch.last_variance  # forecast for test_start
            for t in range(fold.test_start, fold.test_end):
                out["naive-last"].append(max(r[t - 1] ** 2, VARIANCE_FLOOR))
                out["EWMA"].append(float(ewma[t - 1]))
                out["GARCH(1,1)"].append(var)
                var = garch.omega + garch.alpha * (r[t] - mu) ** 2 + garch.beta * var
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# ---------------------------------------------------------------- scoring


@dataclass(frozen=True, slots=True)
class ModelScore:
    name: str
    kind: str  # "naive", "baseline" or "deep"
    mae: float
    rmse: float
    directional_accuracy: float  # nan when the forecast never takes a side
    qlike: float  # variance target only; nan for returns
    skill: float  # 1 - rmse / naive rmse
    dm_vs_naive: tuple[float, float]  # (stat, p); negative stat favours this model
    dm_vs_best_baseline: tuple[float, float] | None


def _direction(forecasts: FloatArray, actual: FloatArray, reference: FloatArray) -> float:
    f, a = forecasts - reference, actual - reference
    mask = (f != 0) & (a != 0)
    return float(np.mean(np.sign(f[mask]) == np.sign(a[mask]))) if mask.any() else math.nan


def score_forecasts(
    name: str,
    kind: str,
    forecasts: FloatArray,
    actual: FloatArray,
    naive: FloatArray,
    reference: FloatArray,
    target: Target,
    best_baseline: FloatArray | None = None,
) -> ModelScore:
    err = actual - forecasts
    rmse = float(np.sqrt(np.mean(err**2)))
    naive_err = actual - naive
    naive_rmse = float(np.sqrt(np.mean(naive_err**2)))
    dm_best = None
    if best_baseline is not None:
        dm_best = diebold_mariano(err, actual - best_baseline)
    return ModelScore(
        name=name,
        kind=kind,
        mae=float(np.mean(np.abs(err))),
        rmse=rmse,
        directional_accuracy=_direction(forecasts, actual, reference),
        qlike=qlike(actual, forecasts) if target is Target.VARIANCE else math.nan,
        skill=1 - rmse / naive_rmse if naive_rmse > 0 else 0.0,
        dm_vs_naive=diebold_mariano(err, naive_err),
        dm_vs_best_baseline=dm_best,
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    asset: str
    target: Target
    scores: tuple[ModelScore, ...]
    runs: tuple[DeepRun, ...]
    n_forecasts: int
    best_baseline: str

    def deep_beats_baselines(self) -> bool:
        """True only if some deep model has lower RMSE than every baseline *and* the DM test
        against the best baseline favours it at the 5% level."""

        base = min(s.rmse for s in self.scores if s.kind != "deep")
        for s in self.scores:
            if s.kind == "deep" and s.rmse < base and s.dm_vs_best_baseline is not None:
                stat, p = s.dm_vs_best_baseline
                if stat < 0 and p < 0.05:
                    return True
        return False

    def table(self) -> str:
        unit = "returns" if self.target is Target.RETURN else "squared returns (x1e4)"
        k = 1.0 if self.target is Target.RETURN else 1e4
        lines = [
            f"{self.asset} | target: next-day {self.target.value} | {self.n_forecasts} "
            f"out-of-sample days | errors in {unit}",
            f"{'model':<12}{'MAE':>10}{'RMSE':>10}{'dir%':>7}{'QLIKE':>8}{'skill':>8}"
            f"{'DM-naive p':>12}{'DM-best p':>11}",
        ]
        for s in self.scores:
            direction = (
                "  n/a"
                if math.isnan(s.directional_accuracy)
                else (f"{s.directional_accuracy * 100:5.1f}")
            )
            q = "    n/a" if math.isnan(s.qlike) else f"{s.qlike:7.3f}"
            dm_best = "-" if s.dm_vs_best_baseline is None else f"{s.dm_vs_best_baseline[1]:.3f}"
            lines.append(
                f"{s.name:<12}{s.mae * k:>10.5f}{s.rmse * k:>10.5f}{direction:>7} {q}"
                f"{s.skill:>+8.2%}{s.dm_vs_naive[1]:>12.3f}{dm_best:>11}"
            )
        verdict = (
            "a deep model beats every baseline (DM p < 0.05)"
            if self.deep_beats_baselines()
            else f"no deep model significantly beats the best baseline ({self.best_baseline})"
        )
        lines.append(f"verdict: {verdict}")
        return "\n".join(lines)


def compare_forecasters(
    returns: npt.ArrayLike,
    target: Target,
    *,
    asset: str = "series",
    specs: Sequence[ModelSpec] | None = None,
    config: TrainConfig | None = None,
    folds: int = 4,
    min_train: int = 750,
) -> Comparison:
    """Walk-forward comparison of deep models with naive and classical baselines."""

    r = _returns(returns)
    cfg = config or TrainConfig()
    plan = plan_folds(r.size, min_train=min_train, folds=folds)
    index = np.arange(plan[0].test_start, plan[-1].test_end)
    actual = target_values(r, target)[index]
    baselines = baseline_forecasts(r, target, plan)
    naive_name = "naive-zero" if target is Target.RETURN else "naive-last"
    naive = baselines[naive_name]
    # direction: sign of the return, or whether variance rises above yesterday's squared return
    reference = np.zeros(index.size) if target is Target.RETURN else naive
    best_name = min(baselines, key=lambda n: float(np.mean((actual - baselines[n]) ** 2)))
    best = baselines[best_name]
    scores = [
        score_forecasts(
            name,
            "naive" if name == naive_name else "baseline",
            pred,
            actual,
            naive,
            reference,
            target,
            None if name == best_name else best,
        )
        for name, pred in baselines.items()
    ]
    runs = []
    for spec in specs if specs is not None else default_specs():
        run = walk_forward_deep(r, spec, target, plan, cfg)
        runs.append(run)
        scores.append(
            score_forecasts(run.name, "deep", run.forecasts, actual, naive, reference, target, best)
        )
    return Comparison(asset, target, tuple(scores), tuple(runs), int(index.size), best_name)


def complexity_from_runs(runs: Sequence[DeepRun]) -> ComplexitySweep:
    """1-SE rule across model sizes, using each run's per-fold out-of-sample MSE."""

    return one_standard_error_rule(
        [Candidate(r.name, r.parameters, tuple(f.test_mse for f in r.folds)) for r in runs]
    )
