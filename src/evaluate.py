from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import DEFAULT_PATHS, DEFAULT_TRAIN_CONFIG
from utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and compare model metrics")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_PATHS.outputs_dir)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TRAIN_CONFIG.top_k)
    parser.add_argument("--save-name", type=str, default="metrics_comparison.csv")
    return parser.parse_args()


def find_metrics(outputs_dir: Path) -> List[Dict]:
    rows = []
    for sub in sorted(outputs_dir.glob("*")):
        metrics_path = sub / "metrics.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            metrics["model_dir"] = sub.name
            rows.append(metrics)
    return rows


def main() -> None:
    args = parse_args()
    rows = find_metrics(args.outputs_dir)
    if not rows:
        raise FileNotFoundError("No metrics.json found under outputs/. Run training scripts first.")

    p_col = f"precision@{args.top_k}"
    r_col = f"recall@{args.top_k}"
    f_col = f"f1@{args.top_k}"

    keep_cols = ["model", "model_dir", "val_rmse", "test_rmse", p_col, r_col, f_col, "num_users_eval"]
    df = pd.DataFrame(rows)
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    df = df[keep_cols].sort_values("test_rmse")

    baseline = None
    if "svd" in set(df["model"]):
        baseline = df[df["model"] == "svd"].iloc[0]

    if baseline is not None:
        df["delta_rmse_vs_svd"] = df["test_rmse"] - baseline["test_rmse"]
        df[f"delta_{f_col}_vs_svd"] = df[f_col] - baseline[f_col]

    out_path = args.outputs_dir / args.save_name
    df.to_csv(out_path, index=False)

    print("Model comparison:")
    print(df.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
