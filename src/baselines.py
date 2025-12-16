from __future__ import annotations
import numpy as np

def seasonal_naive_windows(
    y_full: np.ndarray,
    start: int,
    end: int,
    lookback: int,
    horizon: int,
    season: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (y_true, y_pred) windows on ORIGINAL SCALE for a split [start:end).

    Window definition matches WindowDataset on the split slice:
      split_len = end-start
      n = split_len - lookback - horizon + 1
      for i in [0..n-1]:
        t = (start + i) + lookback
        y_true[i] = y_full[t : t+horizon]
        y_pred[i,k] = y_full[(t+k)-season]
                  -> y_full[t-season : t-season+horizon]
    """
    y_full = np.asarray(y_full, dtype=np.float64)
    split_len = end - start
    n = split_len - lookback - horizon + 1
    if n <= 0:
        raise ValueError("Split too small for lookback+horizon.")

    y_true = np.empty((n, horizon), dtype=np.float64)
    y_pred = np.empty((n, horizon), dtype=np.float64)

    for i in range(n):
        t = (start + i) + lookback
        y_true[i, :] = y_full[t : t + horizon]
        y_pred[i, :] = y_full[t - season : t - season + horizon]

    return y_true, y_pred