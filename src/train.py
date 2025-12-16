from __future__ import annotations
import os
from datetime import datetime
from typing import Any
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.sqlite_opsd import load_opsd_sqlite
from data.preprocess import (
    enforce_hourly_index,
    impute_median_ffill_bfill,
    time_split_indices,
    make_arrays_and_scalers,
    add_calendar_features,
    add_lag_features,
    drop_initial_rows_for_lags,
)
from data.window_dataset import WindowDataset
from models.lstm import LSTMForecaster
from models.gru import GRUForecaster
from baselines import seasonal_naive_windows
from metrics import mae, rmse, smape
from evaluate import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    opsd = cfg["opsd"]
    target = opsd["target"]
    features = opsd["features"]
    ts_col = opsd["timestamp_col"]

    df = load_opsd_sqlite(
        sqlite_path=opsd["sqlite_path"],
        table=opsd["table"],
        timestamp_col=ts_col,
        columns=list(dict.fromkeys(features + [target])),
        start_utc=opsd.get("start_utc"),
        end_utc=opsd.get("end_utc"),
    )

    if cfg["preprocess"]["enforce_hourly_grid"]:
        df = enforce_hourly_index(df, ts_col)

    all_cols = list(dict.fromkeys(features + [target]))
    if cfg["preprocess"]["impute"] == "median_ffill_bfill":
        df = impute_median_ffill_bfill(df, all_cols)
        y_raw = df[target].to_numpy(dtype=np.float64)  # ORIGINAL SCALE for baselines

        # calendar features
        extra_feats = []
        if cfg["preprocess"].get("calendar_features", False):
            df, cal_cols = add_calendar_features(df, ts_col)
            extra_feats.extend(cal_cols)

        # lag features of target
        lags = cfg["preprocess"].get("lags", [])
        if lags:
            df, lag_cols = add_lag_features(df, target, lags)
            extra_feats.extend(lag_cols)
            df = drop_initial_rows_for_lags(df, lags)

        # keep feature list in sync
        features = features + extra_feats

        # update y_raw too (since we dropped initial rows)
        y_raw = df[target].to_numpy(dtype=np.float64)

        # Residual target
        target_used = target
        mode = cfg["preprocess"].get("target_mode", "absolute")
        residual_lag = int(cfg["preprocess"].get("residual_lag", 168))

        if mode == "residual_over_lag":
            lag_col = f"lag_{residual_lag}"
            if lag_col not in df.columns:
                raise ValueError(
                    f"Residual mode requires feature '{lag_col}' to exist. Ensure preprocess.lags includes {residual_lag}."
                )
            df["_target_residual"] = df[target] - df[lag_col]
            target_used = "_target_residual"
    else:
        raise ValueError("Unknown impute method in config.")

    T = len(df)
    tr, va, te = time_split_indices(T, cfg["split"]["train"], cfg["split"]["val"])

    X_all, y_all, xscaler, yscaler = make_arrays_and_scalers(
        df, features, target_used, tr
    )

    lookback = cfg["window"]["lookback"]
    horizon = cfg["window"]["horizon"]

    val_base = None
    test_base = None

    if mode == "residual_over_lag":
        # baseline for reconstruction is lag-168 seasonal naive prediction windows
        _, val_base = seasonal_naive_windows(
            y_raw, va.start, va.stop, lookback, horizon, season=residual_lag
        )
        _, test_base = seasonal_naive_windows(
            y_raw, te.start, te.stop, lookback, horizon, season=residual_lag
        )

    # build per-split datasets *after* scaling, but slices must still be time-safe
    X_tr, y_tr = X_all[tr], y_all[tr]
    X_va, y_va = X_all[va], y_all[va]
    X_te, y_te = X_all[te], y_all[te]

    train_ds = WindowDataset(X_tr, y_tr, lookback, horizon)
    val_ds = WindowDataset(X_va, y_va, lookback, horizon)
    test_ds = WindowDataset(X_te, y_te, lookback, horizon)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    n_features = X_all.shape[1]
    mcfg = cfg["model"]
    if mcfg["name"] == "lstm":
        model = LSTMForecaster(
            n_features, mcfg["hidden"], mcfg["layers"], mcfg["dropout"], horizon
        ).to(device)
    elif mcfg["name"] == "gru":
        model = GRUForecaster(
            n_features, mcfg["hidden"], mcfg["layers"], mcfg["dropout"], horizon
        ).to(device)
    else:
        raise ValueError("model.name must be 'lstm' or 'gru'")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    loss_fn = torch.nn.MSELoss()

    out_dir = cfg["output"]["checkpoints_dir"]
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, cfg["output"]["best_name"])

    sel = cfg.get("selection", {"metric": "RMSE", "mode": "min"})
    metric = sel.get("metric", "RMSE")
    mode = sel.get("mode", "min").lower()

    best_val = float("inf") if mode == "min" else -float("inf")
    best_epoch = -1
    best_metrics: dict[str, float] | None = None

    patience = int(cfg["train"].get("patience", 0))
    min_delta = float(cfg["train"].get("min_delta", 0.0))
    bad_epochs = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total = 0.0
        count = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)
            count += xb.size(0)

        train_mse = total / max(count, 1)
        val_metrics = evaluate(
            model, val_dl, device, yscaler=yscaler, baseline_windows=val_base
        )

        if metric not in val_metrics:
            raise KeyError(
                f"selection.metric='{metric}' not in val_metrics={list(val_metrics.keys())}"
            )

        val_score = val_metrics[metric]
        if mode == "min":
            better = val_score < (best_val - min_delta)
        else:
            better = val_score > (best_val + min_delta)

        extras = " | ".join(
            f"val_{k}={val_metrics[k]:.2f}"
            for k in ("RMSE", "MAE", "sMAPE")
            if k != metric
        )

        print(
            f"epoch {epoch:02d} | train_mse={train_mse:.4f} | "
            f"val_{metric}={val_score:.2f}" + (f" | {extras}" if extras else "")
        )

        if better:
            best_val = val_score
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(model.state_dict(), best_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(
                    f"EARLY STOPPING at epoch {epoch} (no improvement in {patience} epochs; best at {best_epoch})"
                )
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    assert best_metrics is not None

    extras = " | ".join(
        f"val_{k}={best_metrics[k]:.2f}"
        for k in ("RMSE", "MAE", "sMAPE")
        if k != metric
    )

    print(
        f"BEST @ epoch {best_epoch} | val_{metric}={best_val:.2f}"
        + (f" | {extras}" if extras else "")
    )

    # Baselines (original scale)
    lookback = cfg["window"]["lookback"]
    horizon = cfg["window"]["horizon"]

    # Seasonal naive: yesterday (24h) and last week (168h)
    val_y24_true, val_y24_pred = seasonal_naive_windows(
        y_raw, va.start, va.stop, lookback, horizon, season=24
    )
    val_y168_true, val_y168_pred = seasonal_naive_windows(
        y_raw, va.start, va.stop, lookback, horizon, season=168
    )

    val_24 = {
        "RMSE": rmse(val_y24_true, val_y24_pred),
        "MAE": mae(val_y24_true, val_y24_pred),
        "sMAPE": smape(val_y24_true, val_y24_pred),
    }
    val_168 = {
        "RMSE": rmse(val_y168_true, val_y168_pred),
        "MAE": mae(val_y168_true, val_y168_pred),
        "sMAPE": smape(val_y168_true, val_y168_pred),
    }

    print(
        f"BASELINE (VAL) | lag24  RMSE={val_24['RMSE']:.2f} | "
        f"MAE={val_24['MAE']:.2f} | sMAPE={val_24['sMAPE']:.2f}"
    )
    print(
        f"BASELINE (VAL) | lag168 RMSE={val_168['RMSE']:.2f} | "
        f"MAE={val_168['MAE']:.2f} | sMAPE={val_168['sMAPE']:.2f}"
    )

    # Test baselines
    test_y24_true, test_y24_pred = seasonal_naive_windows(
        y_raw, te.start, te.stop, lookback, horizon, season=24
    )
    test_y168_true, test_y168_pred = seasonal_naive_windows(
        y_raw, te.start, te.stop, lookback, horizon, season=168
    )

    test_24 = {
        "RMSE": rmse(test_y24_true, test_y24_pred),
        "MAE": mae(test_y24_true, test_y24_pred),
        "sMAPE": smape(test_y24_true, test_y24_pred),
    }
    test_168 = {
        "RMSE": rmse(test_y168_true, test_y168_pred),
        "MAE": mae(test_y168_true, test_y168_pred),
        "sMAPE": smape(test_y168_true, test_y168_pred),
    }

    print(
        f"BASELINE (TEST)| lag24  RMSE={test_24['RMSE']:.2f} | "
        f"MAE={test_24['MAE']:.2f} | sMAPE={test_24['sMAPE']:.2f}"
    )
    print(
        f"BASELINE (TEST)| lag168 RMSE={test_168['RMSE']:.2f} | "
        f"MAE={test_168['MAE']:.2f} | sMAPE={test_168['sMAPE']:.2f}"
    )

    test_metrics = evaluate(
        model, test_dl, device, yscaler=yscaler, baseline_windows=test_base
    )
    print(
        f"TEST | RMSE={test_metrics['RMSE']:.2f} | "
        f"MAE={test_metrics['MAE']:.2f} | sMAPE={test_metrics['sMAPE']:.2f}"
    )

    return {
        "config_path": config_path,
        "dataset": cfg.get("dataset"),
        "selection": cfg.get("selection", {}),
        "best": {"epoch": best_epoch, "val": best_metrics},
        "baselines": {
            "val": {"lag24": val_24, "lag168": val_168},
            "test": {"lag24": test_24, "lag168": test_168},
        },
        "metrics": {"test": test_metrics},
        "checkpoint_path": best_path,
    }


def eval_only(config_path: str, ckpt_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    opsd = cfg["opsd"]
    target = opsd["target"]
    features = opsd["features"]
    ts_col = opsd["timestamp_col"]

    df = load_opsd_sqlite(
        sqlite_path=opsd["sqlite_path"],
        table=opsd["table"],
        timestamp_col=ts_col,
        columns=list(dict.fromkeys(features + [target])),
        start_utc=opsd.get("start_utc"),
        end_utc=opsd.get("end_utc"),
    )

    if cfg["preprocess"]["enforce_hourly_grid"]:
        df = enforce_hourly_index(df, ts_col)

    all_cols = list(dict.fromkeys(features + [target]))
    if cfg["preprocess"]["impute"] == "median_ffill_bfill":
        df = impute_median_ffill_bfill(df, all_cols)
    else:
        raise ValueError("Unknown impute method in config.")

    # raw target (original scale) for baselines/reconstruction
    y_raw = df[target].to_numpy(dtype=np.float64)

    # calendar + lags (same as training)
    extra_feats = []
    if cfg["preprocess"].get("calendar_features", False):
        df, cal_cols = add_calendar_features(df, ts_col)
        extra_feats.extend(cal_cols)

    lags = cfg["preprocess"].get("lags", [])
    if lags:
        df, lag_cols = add_lag_features(df, target, lags)
        extra_feats.extend(lag_cols)
        df = drop_initial_rows_for_lags(df, lags)

    features = features + extra_feats
    y_raw = df[target].to_numpy(dtype=np.float64)

    # residual mode
    target_used = target
    mode = cfg["preprocess"].get("target_mode", "absolute")
    residual_lag = int(cfg["preprocess"].get("residual_lag", 168))
    if mode == "residual_over_lag":
        lag_col = f"lag_{residual_lag}"
        if lag_col not in df.columns:
            raise ValueError(f"Residual mode requires feature '{lag_col}' to exist.")
        df["_target_residual"] = df[target] - df[lag_col]
        target_used = "_target_residual"

    T = len(df)
    tr, va, te = time_split_indices(T, cfg["split"]["train"], cfg["split"]["val"])

    X_all, y_all, xscaler, yscaler = make_arrays_and_scalers(
        df, features, target_used, tr
    )

    lookback = cfg["window"]["lookback"]
    horizon = cfg["window"]["horizon"]

    X_tr, y_tr = X_all[tr], y_all[tr]
    X_va, y_va = X_all[va], y_all[va]
    X_te, y_te = X_all[te], y_all[te]

    val_ds = WindowDataset(X_va, y_va, lookback, horizon)
    test_ds = WindowDataset(X_te, y_te, lookback, horizon)

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    n_features = X_all.shape[1]
    mcfg = cfg["model"]
    if mcfg["name"] == "lstm":
        model = LSTMForecaster(
            n_features, mcfg["hidden"], mcfg["layers"], mcfg["dropout"], horizon
        ).to(device)
    elif mcfg["name"] == "gru":
        model = GRUForecaster(
            n_features, mcfg["hidden"], mcfg["layers"], mcfg["dropout"], horizon
        ).to(device)
    else:
        raise ValueError("model.name must be 'lstm' or 'gru'")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # baseline windows for reconstruction in residual mode
    val_base = None
    test_base = None
    if mode == "residual_over_lag":
        _, val_base = seasonal_naive_windows(
            y_raw, va.start, va.stop, lookback, horizon, season=residual_lag
        )
        _, test_base = seasonal_naive_windows(
            y_raw, te.start, te.stop, lookback, horizon, season=residual_lag
        )

    val_metrics = evaluate(
        model, val_dl, device, yscaler=yscaler, baseline_windows=val_base
    )
    test_metrics = evaluate(
        model, test_dl, device, yscaler=yscaler, baseline_windows=test_base
    )

    # baselines
    val_y24_true, val_y24_pred = seasonal_naive_windows(
        y_raw, va.start, va.stop, lookback, horizon, season=24
    )
    val_y168_true, val_y168_pred = seasonal_naive_windows(
        y_raw, va.start, va.stop, lookback, horizon, season=168
    )
    test_y24_true, test_y24_pred = seasonal_naive_windows(
        y_raw, te.start, te.stop, lookback, horizon, season=24
    )
    test_y168_true, test_y168_pred = seasonal_naive_windows(
        y_raw, te.start, te.stop, lookback, horizon, season=168
    )

    baselines = {
        "val": {
            "lag24": {
                "RMSE": rmse(val_y24_true, val_y24_pred),
                "MAE": mae(val_y24_true, val_y24_pred),
                "sMAPE": smape(val_y24_true, val_y24_pred),
            },
            "lag168": {
                "RMSE": rmse(val_y168_true, val_y168_pred),
                "MAE": mae(val_y168_true, val_y168_pred),
                "sMAPE": smape(val_y168_true, val_y168_pred),
            },
        },
        "test": {
            "lag24": {
                "RMSE": rmse(test_y24_true, test_y24_pred),
                "MAE": mae(test_y24_true, test_y24_pred),
                "sMAPE": smape(test_y24_true, test_y24_pred),
            },
            "lag168": {
                "RMSE": rmse(test_y168_true, test_y168_pred),
                "MAE": mae(test_y168_true, test_y168_pred),
                "sMAPE": smape(test_y168_true, test_y168_pred),
            },
        },
    }

    return {
        "config_path": config_path,
        "checkpoint_path": ckpt_path,
        "dataset": cfg.get("dataset"),
        "selection": cfg.get("selection", {}),
        "metrics": {"val": val_metrics, "test": test_metrics},
        "baselines": baselines,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/opsd_lstm.yaml")
    args = ap.parse_args()
    main(args.config)
