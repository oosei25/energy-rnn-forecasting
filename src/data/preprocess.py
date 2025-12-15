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

def add_calendar_features(df: pd.DataFrame, ts_col: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Adds cyclical hour-of-day and day-of-week features.
    Returns (df, new_feature_cols).
    """
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])

    hour = ts.dt.hour.astype(np.float32)
    dow = ts.dt.dayofweek.astype(np.float32)  # Mon=0..Sun=6

    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"]  = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"]  = np.cos(2.0 * np.pi * dow / 7.0)

    return out, ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """
    Adds lag features of the TARGET (past-only), e.g. lag_24, lag_168.
    Returns (df, new_feature_cols).
    """
    out = df.copy()
    new_cols = []
    for lag in lags:
        col = f"lag_{lag}"
        out[col] = out[target_col].shift(lag)
        new_cols.append(col)
    return out, new_cols


def drop_initial_rows_for_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    Drops the first max(lags) rows so lag features are not NaN.
    """
    if not lags:
        return df
    return df.iloc[max(lags):].reset_index(drop=True)