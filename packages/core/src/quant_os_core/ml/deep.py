"""Compact PyTorch sequence models for research; training orchestration remains explicit."""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "PyTorch is required for quant_os_core.ml.deep; install quant-os[dl]"
    ) from exc


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1]).squeeze(-1)


class GRUForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1]).squeeze(-1)


class TemporalConvForecaster(nn.Module):
    def __init__(self, input_size: int, channels: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.net = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size, padding=kernel_size - 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(hidden).squeeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        model_dim: int = 32,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        self.input_projection = nn.Linear(input_size, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.input_projection(x))
        return self.head(hidden[:, -1]).squeeze(-1)


class PatchTSTForecaster(nn.Module):
    """Compact patch-based Transformer forecaster for fixed-length sequences."""

    def __init__(
        self,
        input_size: int,
        *,
        patch_size: int = 4,
        model_dim: int = 32,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if patch_size < 1 or model_dim % heads != 0:
            raise ValueError("patch_size must be positive and model_dim divisible by heads")
        self.patch_size = patch_size
        self.input_size = input_size
        self.patch_projection = nn.Linear(input_size * patch_size, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        usable = (x.shape[1] // self.patch_size) * self.patch_size
        if usable < self.patch_size:
            raise ValueError("sequence is shorter than one patch")
        trimmed = x[:, -usable:]
        patches = trimmed.unfold(1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 1, 3, 2).reshape(
            x.shape[0], -1, self.patch_size * self.input_size
        )
        hidden = self.encoder(self.patch_projection(patches))
        return self.head(hidden[:, -1]).squeeze(-1)


class ITransformerForecaster(nn.Module):
    """Inverted Transformer: variables are tokens and history is embedded per variable."""

    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        *,
        model_dim: int = 32,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if sequence_length < 2 or model_dim % heads != 0:
            raise ValueError("invalid sequence_length or attention dimensions")
        self.sequence_length = sequence_length
        self.history_projection = nn.Linear(sequence_length, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(model_dim * input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.sequence_length:
            raise ValueError("input sequence length does not match configured sequence_length")
        tokens = self.history_projection(x.transpose(1, 2))
        hidden = self.encoder(tokens).flatten(1)
        return self.head(hidden).squeeze(-1)
