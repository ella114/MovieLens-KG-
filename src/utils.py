from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def precision_recall_f1_at_k(
    ranked_items: Sequence[int], relevant_items: Sequence[int], k: int
) -> Tuple[float, float, float]:
    if k <= 0:
        return 0.0, 0.0, 0.0
    topk = list(ranked_items[:k])
    rel_set = set(relevant_items)
    if not rel_set:
        return 0.0, 0.0, 0.0
    hit = sum(1 for item in topk if item in rel_set)
    precision = hit / k
    recall = hit / len(rel_set)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def aggregate_topk_metrics(metrics: Iterable[Tuple[float, float, float]]) -> Dict[str, float]:
    metrics = list(metrics)
    if not metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    arr = np.array(metrics)
    return {
        "precision": float(arr[:, 0].mean()),
        "recall": float(arr[:, 1].mean()),
        "f1": float(arr[:, 2].mean()),
    }
