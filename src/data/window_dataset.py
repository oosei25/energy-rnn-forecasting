from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """
    Produces:
      x: (lookback, n_features)
      y: (horizon,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int, horizon: int):
        if y.ndim != 1:
            raise ValueError("y must be 1D after preprocessing/scaling.")
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        self.n = len(self.X) - self.lookback - self.horizon + 1
        if self.n <= 0:
            raise ValueError("Not enough rows for lookback+horizon.")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        i = idx
        xb = self.X[i : i + self.lookback]
        yb = self.y[i + self.lookback : i + self.lookback + self.horizon]
        return torch.from_numpy(xb), torch.from_numpy(yb)
