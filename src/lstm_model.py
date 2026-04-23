"""
Baseline LSTM architecture and sliding-window dataset helpers.

The model consumes a fixed-length window of past features and predicts the
log of the 21-day forward realized volatility at the last timestep of the
window. Log-target training is handled by the caller — this module is
target-agnostic.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class LSTMRegressor(nn.Module):
    """
    Stacked LSTM → linear head producing a scalar prediction per sequence.

    Parameters
    ----------
    input_size  : number of input features per timestep
    hidden_size : LSTM hidden state dimension
    n_layers    : number of stacked LSTM layers
    dropout     : dropout between LSTM layers (ignored if n_layers == 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class VolatilityWindowDataset(Dataset):
    """
    Sliding-window view over aligned feature/target arrays.

    Window i contains features[i : i + seq_len] and the target at the window's
    last row, targets[i + seq_len - 1]. Rows must already be cleaned of NaNs
    and chronologically sorted by the caller.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int,
    ) -> None:
        if len(features) != len(targets):
            raise ValueError(
                f"features ({len(features)}) and targets ({len(targets)}) length mismatch"
            )
        if len(features) < seq_len:
            raise ValueError(
                f"Not enough rows ({len(features)}) for seq_len={seq_len}"
            )
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return x, y
