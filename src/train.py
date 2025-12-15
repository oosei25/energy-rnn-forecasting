from __future__ import annotations
import os
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

        # --- calendar features ---
        extra_feats = []
        if cfg["preprocess"].get("calendar_features", False):
            df, cal_cols = add_calendar_features(df, ts_col)
            extra_feats.extend(cal_cols)

        # --- lag features of target ---
        lags = cfg["preprocess"].get("lags", [])
        if lags:
            df, lag_cols = add_lag_features(df, target, lags)
            extra_feats.extend(lag_cols)
            df = drop_initial_rows_for_lags(df, lags)

        # keep feature list in sync
        features = features + extra_feats

        # update y_raw too (since we dropped initial rows)
        y_raw = df[target].to_numpy(dtype=np.float64)
    else:
        raise ValueError("Unknown impute method in config.")

    T = len(df)
    tr, va, te = time_split_indices(T, cfg["split"]["train"], cfg["split"]["val"])

    X_all, y_all, xscaler, yscaler = make_arrays_and_scalers(df, features, target, tr)

    lookback = cfg["window"]["lookback"]
    horizon = cfg["window"]["horizon"]

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
        val_metrics = evaluate(model, val_dl, device, yscaler=yscaler)

        if metric not in val_metrics:
            raise KeyError(
                f"selection.metric='{metric}' not in val_metrics={list(val_metrics.keys())}"
            )

        val_score = val_metrics[metric]
        better = (val_score < best_val) if mode == "min" else (val_score > best_val)

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

    # ---- Baselines (original scale) ----
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

    test_metrics = evaluate(model, test_dl, device, yscaler=yscaler)
    print(
        f"TEST | RMSE={test_metrics['RMSE']:.2f} | "
        f"MAE={test_metrics['MAE']:.2f} | sMAPE={test_metrics['sMAPE']:.2f}"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/opsd_lstm.yaml")
    args = ap.parse_args()
    main(args.config)
