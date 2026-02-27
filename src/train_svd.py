from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import DEFAULT_PATHS, DEFAULT_TRAIN_CONFIG
from models.svd import SVDBaseline
from train_utils import InteractionDataset, build_relevant_dict, build_seen_dict, evaluate_rmse, evaluate_topk
from utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVD baseline recommender")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PATHS.processed_dir)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PATHS.outputs_dir / "svd")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TRAIN_CONFIG.top_k)
    parser.add_argument("--positive-threshold", type=float, default=DEFAULT_TRAIN_CONFIG.positive_threshold)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_CONFIG.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    train_df = pd.read_csv(args.processed_dir / "train.csv")
    val_df = pd.read_csv(args.processed_dir / "val.csv")
    test_df = pd.read_csv(args.processed_dir / "test.csv")

    num_users = int(train_df["user_idx"].max()) + 1
    num_items = int(train_df["item_idx"].max()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVDBaseline(num_users=num_users, num_items=num_items, embedding_dim=args.embedding_dim).to(device)

    train_loader = DataLoader(InteractionDataset(train_df), batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    patience = 0
    best_path = args.out_dir / "best.pt"
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for user_idx, item_idx, rating in train_loader:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating = rating.to(device)

            pred = model(user_idx, item_idx)
            loss = criterion(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(rating)

        epoch_loss /= len(train_df)
        model.eval()

        val_rmse = evaluate_rmse(model.forward, val_df, device=device)
        train_curve.append({"epoch": epoch, "train_loss": epoch_loss, "val_rmse": val_rmse})
        print(f"[SVD] epoch={epoch:03d} train_loss={epoch_loss:.4f} val_rmse={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    val_rmse = evaluate_rmse(model.forward, val_df, device=device)
    test_rmse = evaluate_rmse(model.forward, test_df, device=device)

    seen_dict = build_seen_dict(train_df)
    relevant_dict = build_relevant_dict(test_df, threshold=args.positive_threshold)
    topk_metrics = evaluate_topk(
        score_fn=lambda uid: model.all_item_scores(uid),
        seen_dict=seen_dict,
        relevant_dict=relevant_dict,
        k=args.top_k,
        device=device,
    )

    metrics = {
        "model": "svd",
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        f"precision@{args.top_k}": topk_metrics["precision"],
        f"recall@{args.top_k}": topk_metrics["recall"],
        f"f1@{args.top_k}": topk_metrics["f1"],
        "num_users_eval": len(relevant_dict),
        "config": {
            "embedding_dim": args.embedding_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stop": args.early_stop,
            "positive_threshold": args.positive_threshold,
        },
    }

    pd.DataFrame(train_curve).to_csv(args.out_dir / "train_curve.csv", index=False)
    save_json(metrics, args.out_dir / "metrics.json")

    # Save top-10 predictions for sample users.
    sample_users = sorted(relevant_dict.keys())[:50]
    rec_rows = []
    with torch.no_grad():
        for uid in sample_users:
            scores = model.all_item_scores(torch.tensor([uid], dtype=torch.long, device=device)).squeeze(0)
            seen = seen_dict.get(uid, set())
            if seen:
                idx = torch.tensor(list(seen), dtype=torch.long, device=device)
                scores[idx] = -1e9
            top_items = torch.topk(scores, k=args.top_k).indices.detach().cpu().numpy().tolist()
            for rank, item_idx in enumerate(top_items, start=1):
                rec_rows.append(
                    {
                        "user_idx": uid,
                        "rank": rank,
                        "item_idx": item_idx,
                        "score": float(scores[item_idx].detach().cpu().item()),
                    }
                )
    pd.DataFrame(rec_rows).to_csv(args.out_dir / "topk_recommendations.csv", index=False)

    print("[SVD] done")
    print(metrics)


if __name__ == "__main__":
    main()
