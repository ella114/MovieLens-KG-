from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from config import DEFAULT_PATHS, DEFAULT_TRAIN_CONFIG
from models.cf import ItemCF, ItemCFConfig
from train_utils import build_relevant_dict, build_seen_dict
from utils import ensure_dir, precision_recall_f1_at_k, rmse, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ItemCF baseline recommender")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PATHS.processed_dir)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PATHS.outputs_dir / "cf")
    parser.add_argument("--topn-sim", type=int, default=50)
    parser.add_argument("--shrinkage", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TRAIN_CONFIG.top_k)
    parser.add_argument("--positive-threshold", type=float, default=DEFAULT_TRAIN_CONFIG.positive_threshold)
    return parser.parse_args()


def eval_rmse(cf: ItemCF, df: pd.DataFrame) -> float:
    pred = []
    truth = df["rating"].to_numpy(dtype=np.float32)
    for row in df[["user_idx", "item_idx"]].itertuples(index=False):
        user_scores = cf.predict_user(int(row.user_idx))
        pred.append(user_scores[int(row.item_idx)])
    pred_arr = np.array(pred, dtype=np.float32)
    return rmse(pred_arr, truth)


def eval_topk(cf: ItemCF, train_df: pd.DataFrame, test_df: pd.DataFrame, top_k: int, threshold: float) -> dict:
    seen_dict = build_seen_dict(train_df)
    relevant = build_relevant_dict(test_df, threshold=threshold)

    users = sorted(relevant.keys())
    metrics = []
    rec_rows = []

    for uid in users:
        scores = cf.predict_user(uid)
        seen = seen_dict.get(uid, set())
        if seen:
            scores[list(seen)] = -np.inf
        top_items = np.argpartition(scores, -top_k)[-top_k:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]
        top_items = top_items.tolist()
        p, r, f1 = precision_recall_f1_at_k(top_items, list(relevant[uid]), k=top_k)
        metrics.append((p, r, f1))
        for rank, item_idx in enumerate(top_items, start=1):
            rec_rows.append(
                {
                    "user_idx": uid,
                    "rank": rank,
                    "item_idx": int(item_idx),
                    "score": float(scores[item_idx]),
                }
            )

    arr = np.array(metrics) if metrics else np.zeros((1, 3))
    return {
        "precision": float(arr[:, 0].mean()),
        "recall": float(arr[:, 1].mean()),
        "f1": float(arr[:, 2].mean()),
        "num_users_eval": len(users),
        "recommendations": pd.DataFrame(rec_rows),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    train_df = pd.read_csv(args.processed_dir / "train.csv")
    val_df = pd.read_csv(args.processed_dir / "val.csv")
    test_df = pd.read_csv(args.processed_dir / "test.csv")

    train_mat = sparse.load_npz(args.processed_dir / "rating_matrix_train.npz").tocsr()

    cf = ItemCF(ItemCFConfig(topn_sim=args.topn_sim, shrinkage=args.shrinkage))
    cf.fit(train_mat)

    val_rmse = eval_rmse(cf, val_df)
    test_rmse = eval_rmse(cf, test_df)
    topk = eval_topk(cf, train_df, test_df, top_k=args.top_k, threshold=args.positive_threshold)

    metrics = {
        "model": "itemcf",
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        f"precision@{args.top_k}": topk["precision"],
        f"recall@{args.top_k}": topk["recall"],
        f"f1@{args.top_k}": topk["f1"],
        "num_users_eval": topk["num_users_eval"],
        "config": {
            "topn_sim": args.topn_sim,
            "shrinkage": args.shrinkage,
        },
    }
    save_json(metrics, args.out_dir / "metrics.json")
    topk["recommendations"].to_csv(args.out_dir / "topk_recommendations.csv", index=False)

    print("[ItemCF] done")
    print(metrics)


if __name__ == "__main__":
    main()
