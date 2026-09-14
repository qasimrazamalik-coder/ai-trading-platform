"""Explicit PyTorch training with validation-based early stopping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class TorchTrainingResult:
    epochs_run: int
    best_validation_loss: float
    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


def train_torch_regressor(
    model: object,
    x_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
    x_validation: npt.ArrayLike,
    y_validation: npt.ArrayLike,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 10,
    seed: int = 7,
) -> TorchTrainingResult:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required; install quant-os[dl]") from exc
    if epochs < 1 or patience < 1 or learning_rate <= 0.0:
        raise ValueError("epochs/patience must be positive and learning_rate > 0")
    xtr = torch.as_tensor(np.asarray(x_train, dtype=np.float32))
    ytr = torch.as_tensor(np.asarray(y_train, dtype=np.float32))
    xva = torch.as_tensor(np.asarray(x_validation, dtype=np.float32))
    yva = torch.as_tensor(np.asarray(y_validation, dtype=np.float32))
    if xtr.ndim < 2 or xva.ndim != xtr.ndim or ytr.ndim != 1 or yva.ndim != 1:
        raise ValueError("invalid training/validation tensor dimensions")
    if xtr.shape[0] != ytr.shape[0] or xva.shape[0] != yva.shape[0]:
        raise ValueError("feature and target row counts must match")
    torch.manual_seed(seed)
    module = model
    parameters = getattr(module, "parameters", None)
    if parameters is None:
        raise TypeError("model must be a PyTorch module")
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    best_state = None
    best_loss = float("inf")
    stale = 0
    train_history: list[float] = []
    val_history: list[float] = []
    for _ in range(epochs):
        module.train()
        optimizer.zero_grad()
        train_prediction = module(xtr)
        train_loss = loss_fn(train_prediction, ytr)
        train_loss.backward()
        optimizer.step()
        module.eval()
        with torch.no_grad():
            validation_loss = loss_fn(module(xva), yva)
        train_value = float(train_loss.detach().cpu())
        validation_value = float(validation_loss.detach().cpu())
        train_history.append(train_value)
        val_history.append(validation_value)
        if validation_value < best_loss - 1e-12:
            best_loss = validation_value
            best_state = {key: value.detach().clone() for key, value in module.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    return TorchTrainingResult(
        epochs_run=len(train_history),
        best_validation_loss=best_loss,
        training_loss=tuple(train_history),
        validation_loss=tuple(val_history),
    )
