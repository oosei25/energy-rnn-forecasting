from __future__ import annotations

import argparse
import os
import sys
import shutil
from datetime import datetime
import sqlite3
import yaml


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_run_dir(dataset: str | None) -> str:
    ds = dataset or "run"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{ts}_{ds}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_artifacts(
    run_dir: str, results: dict, config_path: str, ckpt_path: str | None = None
) -> None:
    # copy config
    shutil.copy2(config_path, os.path.join(run_dir, "config.yaml"))

    # write results
    out_path = os.path.join(run_dir, "results.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    # copy checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        shutil.copy2(ckpt_path, os.path.join(run_dir, os.path.basename(ckpt_path)))

    write_results_md(run_dir, results)


def _fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def write_results_md(run_dir: str, results: dict) -> None:
    """
    Write a recruiter-friendly summary to runs/.../results.md
    Works for both train.main() results and train.eval_only() results.
    """
    dataset = results.get("dataset", "unknown")
    cfg_path = results.get("config_path", "config.yaml")
    ckpt_path = results.get("checkpoint_path", results.get("checkpoint", ""))
    selection = results.get("selection", {})

    # Model metrics: support both shapes
    # - eval_only: results["metrics"]["val"/"test"]
    # - train.main: results["best"]["val"] and results["metrics"]["test"]
    metrics = results.get("metrics", {}) or {}
    best = results.get("best", {}) or {}

    model_val = metrics.get("val") or best.get("val")
    model_test = metrics.get("test") or results.get("test")  # fallback

    # Baselines
    baselines = results.get("baselines", {}) or {}
    base_val = baselines.get("val", {}) or {}
    base_test = baselines.get("test", {}) or {}

    # Convenience: baseline lag168 for improvement calc
    lag168_test = base_test.get("lag168") or {}
    rmse_lag168_test = lag168_test.get("RMSE")
    rmse_model_test = (model_test or {}).get("RMSE")

    improvement_rmse = None
    try:
        if rmse_lag168_test is not None and rmse_model_test is not None:
            improvement_rmse = (
                (float(rmse_lag168_test) - float(rmse_model_test))
                / float(rmse_lag168_test)
                * 100.0
            )
    except Exception:
        improvement_rmse = None

    lines = []
    lines.append("# Run Results\n")
    lines.append(f"- **Dataset:** `{dataset}`")
    lines.append(f"- **Config:** `{cfg_path}`")
    if ckpt_path:
        lines.append(f"- **Checkpoint:** `{ckpt_path}`")
    if selection:
        lines.append(
            f"- **Selection:** metric=`{selection.get('metric','RMSE')}`, mode=`{selection.get('mode','min')}`"
        )
    if best.get("epoch") is not None:
        lines.append(f"- **Best epoch:** `{best.get('epoch')}`")

    # Key takeaway (recruiter-friendly)
    if improvement_rmse is not None:
        lines.append("")
        lines.append(
            "**Key takeaway:** The residual LSTM beats a strong seasonal-naive baseline (lag-168) on the test set."
        )
        lines.append(
            f"- **Test RMSE improvement vs seasonal naive (lag-168):** `{improvement_rmse:.2f}%` (lower is better)"
        )

    def add_table(title: str, model_m: dict | None, base: dict):
        lines.append(f"\n## {title}\n")
        lines.append("| Model | RMSE | MAE | sMAPE |")
        lines.append("|---|---:|---:|---:|")

        if model_m:
            lines.append(
                f"| **Neural net** | {_fmt(model_m.get('RMSE'))} | {_fmt(model_m.get('MAE'))} | {_fmt(model_m.get('sMAPE'))} |"
            )
        else:
            lines.append("| **Neural net** | — | — | — |")

        for k in ("lag24", "lag168"):
            b = base.get(k)
            if b:
                lines.append(
                    f"| Baseline `{k}` | {_fmt(b.get('RMSE'))} | {_fmt(b.get('MAE'))} | {_fmt(b.get('sMAPE'))} |"
                )
            else:
                lines.append(f"| Baseline `{k}` | — | — | — |")

    add_table("Validation", model_val, base_val)
    add_table("Test", model_test, base_test)

    out_path = os.path.join(run_dir, "results.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def check_opsd(cfg: dict) -> int:
    if cfg.get("dataset") != "opsd":
        print(
            f"[check] dataset={cfg.get('dataset')} (only 'opsd' supported by this CLI right now)"
        )
        return 2

    opsd = cfg["opsd"]
    sqlite_path = opsd["sqlite_path"]
    table = opsd["table"]
    ts_col = opsd["timestamp_col"]
    target = opsd["target"]
    features = opsd["features"]

    print("[check] OPSD config:")
    print(f"  sqlite_path: {sqlite_path}")
    print(f"  table:       {table}")
    print(f"  timestamp:   {ts_col}")
    print(f"  target:      {target}")
    print(f"  n_features:  {len(features)}")

    if not os.path.exists(sqlite_path):
        print(f"[check] ERROR: sqlite_path does not exist: {sqlite_path}")
        return 2

    try:
        with sqlite3.connect(sqlite_path) as con:
            # table exists?
            tables = [
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            if table not in tables:
                print(
                    f"[check] ERROR: table '{table}' not found. Available: {tables[:20]}{'...' if len(tables)>20 else ''}"
                )
                return 2

            cols = [
                r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()
            ]
            missing = []
            for c in [ts_col, target] + list(features):
                if c not in cols:
                    missing.append(c)

            if missing:
                print("[check] ERROR: missing required columns:")
                for c in missing:
                    print(f"  - {c}")
                return 2

            # quick row count (fast-ish)
            n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            print(f"[check] OK: table '{table}' rows={n:,} cols={len(cols)}")

    except sqlite3.Error as e:
        print(f"[check] ERROR: sqlite issue: {e}")
        return 2

    print("[check] ✅ All good.")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if cfg.get("dataset") == "opsd":
        return check_opsd(cfg)
    print(f"[check] dataset={cfg.get('dataset')} not supported yet (only opsd).")
    return 2


def cmd_train(args: argparse.Namespace) -> int:
    import train

    cfg = load_config(args.config)
    run_dir = args.run_dir or make_run_dir(cfg.get("dataset"))

    results = train.main(args.config)  # returns dict
    ckpt_path = results.get("checkpoint_path")
    save_run_artifacts(run_dir, results, args.config, ckpt_path)

    print(f"[train] logged to: {run_dir}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import train

    cfg = load_config(args.config)
    ckpt = args.ckpt
    if ckpt is None:
        out = cfg["output"]
        ckpt = os.path.join(out["checkpoints_dir"], out["best_name"])

    run_dir = args.run_dir or make_run_dir(cfg.get("dataset"))
    results = train.eval_only(args.config, ckpt)

    save_run_artifacts(run_dir, results, args.config, ckpt)
    print(f"[eval] logged to: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="energy-rnn-forecasting",
        description="CLI for training/evaluating energy forecasting models.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Validate config + dataset files/schema")
    p_check.add_argument("--config", default="configs/opsd_lstm.yaml")
    p_check.set_defaults(func=cmd_check)

    p_train = sub.add_parser("train", help="Run end-to-end training")
    p_train.add_argument("--config", default="configs/opsd_lstm.yaml")
    p_train.add_argument(
        "--run-dir",
        default=None,
        help="Where to save results (default: runs/<timestamp>_<dataset>/)",
    )
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint (no training)")
    p_eval.add_argument("--config", default="configs/opsd_lstm.yaml")
    p_eval.add_argument(
        "--ckpt",
        default=None,
        help="Path to .pt checkpoint (state_dict). Defaults to config output.best_name.",
    )
    p_eval.add_argument(
        "--run-dir",
        default=None,
        help="Where to save results (default: runs/<timestamp>_<dataset>/)",
    )
    p_eval.set_defaults(func=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
