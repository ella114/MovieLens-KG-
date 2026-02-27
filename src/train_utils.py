from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:  # type: ignore[override]
        pass

from utils import aggregate_topk_metrics, precision_recall_f1_at_k, rmse


class InteractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        if torch is None:
            raise ImportError("torch is required for InteractionDataset")
        self.user_idx = torch.tensor(df["user_idx"].to_numpy(), dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].to_numpy(), dtype=torch.long)
        self.rating = torch.tensor(df["rating"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        return self.user_idx[idx], self.item_idx[idx], self.rating[idx]


def build_seen_dict(df: pd.DataFrame) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for row in df[["user_idx", "item_idx"]].itertuples(index=False):
        out.setdefault(int(row.user_idx), set()).add(int(row.item_idx))
    return out


def build_relevant_dict(df: pd.DataFrame, threshold: float) -> Dict[int, Set[int]]:
    filt = df[df["rating"] >= threshold]
    out: Dict[int, Set[int]] = {}
    for row in filt[["user_idx", "item_idx"]].itertuples(index=False):
        out.setdefault(int(row.user_idx), set()).add(int(row.item_idx))
    return out


def evaluate_rmse(
    predict_fn: Callable[[Any, Any], Any],
    df: pd.DataFrame,
    device: Any,
    batch_size: int = 8192,
) -> float:
    if torch is None:
        raise ImportError("torch is required for evaluate_rmse")

    user = torch.tensor(df["user_idx"].to_numpy(), dtype=torch.long, device=device)
    item = torch.tensor(df["item_idx"].to_numpy(), dtype=torch.long, device=device)
    target = df["rating"].to_numpy().astype(np.float32)

    preds: List[np.ndarray] = []
    n = len(df)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            p = predict_fn(user[start:end], item[start:end])
            preds.append(p.detach().cpu().numpy())
    pred = np.concatenate(preds)
    return rmse(pred, target)


def evaluate_topk(
    score_fn: Callable[[Any], Any],
    seen_dict: Dict[int, Set[int]],
    relevant_dict: Dict[int, Set[int]],
    k: int,
    device: Any,
) -> Dict[str, float]:
    if torch is None:
        raise ImportError("torch is required for evaluate_topk")

    per_user = []
    users = sorted(relevant_dict.keys())
    if not users:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    with torch.no_grad():
        for uid in users:
            scores = score_fn(torch.tensor([uid], dtype=torch.long, device=device)).squeeze(0)
            seen_items = seen_dict.get(uid, set())
            if seen_items:
                seen_idx = torch.tensor(list(seen_items), dtype=torch.long, device=device)
                scores[seen_idx] = -1e9
            topk_items = torch.topk(scores, k=k).indices.detach().cpu().numpy().tolist()
            rel_items = list(relevant_dict[uid])
            per_user.append(precision_recall_f1_at_k(topk_items, rel_items, k=k))
    return aggregate_topk_metrics(per_user)


def sample_bpr_triplets(
    train_df: pd.DataFrame,
    num_items: int,
    seen_dict: Dict[int, Set[int]],
    threshold: float,
    num_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_df = train_df[train_df["rating"] >= threshold]
    if len(pos_df) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    sample_idx = rng.integers(0, len(pos_df), size=num_samples)
    sampled = pos_df.iloc[sample_idx]

    users = sampled["user_idx"].to_numpy(dtype=np.int64)
    pos_items = sampled["item_idx"].to_numpy(dtype=np.int64)
    neg_items = np.empty_like(pos_items)

    for i, uid in enumerate(users):
        seen = seen_dict.get(int(uid), set())
        while True:
            ni = int(rng.integers(0, num_items))
            if ni not in seen:
                neg_items[i] = ni
                break

    return users, pos_items, neg_items
