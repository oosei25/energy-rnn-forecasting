# evaluate.py
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING
from metrics import mae, rmse, smape

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler
else:
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception:  # pragma: no cover
        StandardScaler = None


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    yscaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_pred) shaped (N, horizon).
    If yscaler is provided, outputs are inverse-transformed to original scale.
    """
    model.eval()
    ys, preds = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)              # (B, H)

        y_np = yb.cpu().numpy()      # (B, H)
        p_np = out.cpu().numpy()     # (B, H)

        if yscaler is not None:
            # StandardScaler expects 2D, so reshape (BxH,1) -> inverse -> (B,H)
            y_np = yscaler.inverse_transform(y_np.reshape(-1, 1)).reshape(y_np.shape)
            p_np = yscaler.inverse_transform(p_np.reshape(-1, 1)).reshape(p_np.shape)

        ys.append(y_np)
        preds.append(p_np)

    return np.concatenate(ys, axis=0), np.concatenate(preds, axis=0)


def evaluate(model, loader, device: str, yscaler=None) -> dict[str, float]:
    y_true, y_pred = predict(model, loader, device, yscaler=yscaler)
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }