from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def enforce_hourly_index(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(ts_col)
    df = df.set_index(ts_col)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df.index.name = ts_col
    return df.reset_index()


def impute_median_ffill_bfill(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = out[c]
        med = np.nanmedian(s.to_numpy(dtype=float))
        out[c] = s.fillna(med).ffill().bfill()
    return out


def time_split_indices(T: int, train: float, val: float):
    t1 = int(T * train)
    t2 = int(T * (train + val))
    return slice(0, t1), slice(t1, t2), slice(t2, T)


def make_arrays_and_scalers(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_slice: slice,
):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    xscaler = StandardScaler()
    yscaler = StandardScaler()

    X_train = xscaler.fit_transform(X[train_slice])
    y_train = yscaler.fit_transform(y[train_slice].reshape(-1, 1))[:, 0]

    # transform full arrays (train/val/test later via slices)
    X_all = xscaler.transform(X)
    y_all = yscaler.transform(y.reshape(-1, 1))[:, 0]

    return X_all, y_all, xscaler, yscaler
