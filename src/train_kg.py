from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader

from config import DEFAULT_PATHS, DEFAULT_TRAIN_CONFIG
from models.kg_gcn import KGGCNRecommender
from train_utils import (
    InteractionDataset,
    build_relevant_dict,
    build_seen_dict,
    evaluate_rmse,
    evaluate_topk,
)
from utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KG-enhanced GCN recommender")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PATHS.processed_dir)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PATHS.outputs_dir / "kg_gcn")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.2, help="Weight for BPR loss")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TRAIN_CONFIG.top_k)
    parser.add_argument("--positive-threshold", type=float, default=DEFAULT_TRAIN_CONFIG.positive_threshold)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_CONFIG.seed)
    return parser.parse_args()


def build_norm_adj(
    num_items: int,
    edge_df: pd.DataFrame,
    device: torch.device,
) -> torch.Tensor:
    if edge_df.empty:
        indices = torch.arange(num_items, device=device)
        idx = torch.stack([indices, indices], dim=0)
        vals = torch.ones(num_items, device=device)
        return torch.sparse_coo_tensor(idx, vals, (num_items, num_items), device=device).coalesce()

    src = edge_df["src"].to_numpy(dtype=np.int64)
    dst = edge_df["dst"].to_numpy(dtype=np.int64)
    weight = edge_df["weight"].to_numpy(dtype=np.float32)

    self_idx = np.arange(num_items, dtype=np.int64)
    src = np.concatenate([src, self_idx])
    dst = np.concatenate([dst, self_idx])
    weight = np.concatenate([weight, np.ones(num_items, dtype=np.float32)])

    src_t = torch.from_numpy(src).to(device)
    dst_t = torch.from_numpy(dst).to(device)
    val_t = torch.from_numpy(weight).to(device)

    deg = torch.zeros(num_items, device=device, dtype=torch.float32)
    deg.index_add_(0, src_t, val_t)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    norm_val = val_t * deg_inv_sqrt[src_t] * deg_inv_sqrt[dst_t]

    idx = torch.stack([src_t, dst_t], dim=0)
    adj = torch.sparse_coo_tensor(idx, norm_val, (num_items, num_items), device=device)
    return adj.coalesce()


def sample_neg_items(users: np.ndarray, seen_dict: Dict[int, Set[int]], num_items: int, rng: np.random.Generator) -> np.ndarray:
    neg = np.empty(len(users), dtype=np.int64)
    for i, uid in enumerate(users):
        seen = seen_dict.get(int(uid), set())
        while True:
            cand = int(rng.integers(0, num_items))
            if cand not in seen:
                neg[i] = cand
                break
    return neg


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    train_df = pd.read_csv(args.processed_dir / "train.csv")
    val_df = pd.read_csv(args.processed_dir / "val.csv")
    test_df = pd.read_csv(args.processed_dir / "test.csv")

    node_features = sparse.load_npz(args.processed_dir / "node_features.npz").astype(np.float32)
    edge_df = pd.read_csv(args.processed_dir / "kg_edges.csv")

    num_users = int(train_df["user_idx"].max()) + 1
    num_items = int(train_df["item_idx"].max()) + 1
    feature_dim = int(node_features.shape[1])

    if node_features.shape[0] != num_items:
        raise ValueError(
            f"node_features rows({node_features.shape[0]}) != num_items({num_items})."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(node_features.toarray(), dtype=torch.float32, device=device)
    adj = build_norm_adj(num_items=num_items, edge_df=edge_df, device=device)

    model = KGGCNRecommender(
        num_users=num_users,
        num_items=num_items,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss()
    train_loader = DataLoader(InteractionDataset(train_df), batch_size=args.batch_size, shuffle=True)

    seen_dict = build_seen_dict(train_df)
    pos_df = train_df[train_df["rating"] >= args.positive_threshold]
    pos_pairs = pos_df[["user_idx", "item_idx"]].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(args.seed)

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

            item_emb = model.compute_item_embeddings(x, adj)
            pred = model.score(user_idx, item_idx, item_emb)
            loss = mse_loss(pred, rating)

            if args.alpha > 0 and len(pos_pairs) > 0:
                sampled = pos_pairs[rng.integers(0, len(pos_pairs), size=len(user_idx))]
                bpr_users = sampled[:, 0]
                bpr_pos_items = sampled[:, 1]
                bpr_neg_items = sample_neg_items(bpr_users, seen_dict, num_items, rng)

                bpr_loss = model.bpr_loss(
                    user_idx=torch.tensor(bpr_users, dtype=torch.long, device=device),
                    pos_item_idx=torch.tensor(bpr_pos_items, dtype=torch.long, device=device),
                    neg_item_idx=torch.tensor(bpr_neg_items, dtype=torch.long, device=device),
                    item_emb=item_emb,
                )
                loss = loss + args.alpha * bpr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(rating)

        epoch_loss /= len(train_df)

        model.eval()
        with torch.no_grad():
            item_emb_eval = model.compute_item_embeddings(x, adj)

        val_rmse = evaluate_rmse(
            predict_fn=lambda u, i: model.score(u, i, item_emb_eval),
            df=val_df,
            device=device,
        )
        train_curve.append({"epoch": epoch, "train_loss": epoch_loss, "val_rmse": val_rmse})
        print(f"[KG-GCN] epoch={epoch:03d} train_loss={epoch_loss:.4f} val_rmse={val_rmse:.4f}")

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
    with torch.no_grad():
        item_emb = model.compute_item_embeddings(x, adj)

    val_rmse = evaluate_rmse(
        predict_fn=lambda u, i: model.score(u, i, item_emb),
        df=val_df,
        device=device,
    )
    test_rmse = evaluate_rmse(
        predict_fn=lambda u, i: model.score(u, i, item_emb),
        df=test_df,
        device=device,
    )

    relevant_dict = build_relevant_dict(test_df, threshold=args.positive_threshold)
    topk_metrics = evaluate_topk(
        score_fn=lambda uid: model.all_item_scores(uid, item_emb),
        seen_dict=seen_dict,
        relevant_dict=relevant_dict,
        k=args.top_k,
        device=device,
    )

    metrics = {
        "model": "kg_gcn",
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        f"precision@{args.top_k}": topk_metrics["precision"],
        f"recall@{args.top_k}": topk_metrics["recall"],
        f"f1@{args.top_k}": topk_metrics["f1"],
        "num_users_eval": len(relevant_dict),
        "config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stop": args.early_stop,
            "alpha": args.alpha,
            "positive_threshold": args.positive_threshold,
        },
    }

    pd.DataFrame(train_curve).to_csv(args.out_dir / "train_curve.csv", index=False)
    save_json(metrics, args.out_dir / "metrics.json")

    sample_users = sorted(relevant_dict.keys())[:50]
    rec_rows = []
    with torch.no_grad():
        for uid in sample_users:
            scores = model.all_item_scores(torch.tensor([uid], dtype=torch.long, device=device), item_emb).squeeze(0)
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

    print("[KG-GCN] done")
    print(metrics)


if __name__ == "__main__":
    main()
