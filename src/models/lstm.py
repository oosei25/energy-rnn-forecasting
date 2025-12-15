from __future__ import annotations
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self, n_features: int, hidden: int, layers: int, dropout: float, horizon: int
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # (B, L, H)
        last = out[:, -1, :]  # (B, H)
        return self.head(last)  # (B, horizon)
