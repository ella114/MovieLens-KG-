from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


@dataclass
class ItemCFConfig:
    topn_sim: int = 50
    shrinkage: float = 10.0


class ItemCF:
    def __init__(self, config: ItemCFConfig | None = None) -> None:
        self.config = config or ItemCFConfig()
        self.sim_matrix: sparse.csr_matrix | None = None
        self.user_item: sparse.csr_matrix | None = None

    def fit(self, user_item_matrix: sparse.csr_matrix) -> None:
        self.user_item = user_item_matrix.tocsr()
        item_user = self.user_item.T.tocsr()
        item_user = normalize(item_user, norm="l2", axis=1)
        sim = item_user @ item_user.T
        sim = sim.tolil()

        topn = self.config.topn_sim
        for i in range(sim.shape[0]):
            row = sim.data[i]
            cols = sim.rows[i]
            if len(row) > topn:
                idx = np.argpartition(np.array(row), -topn)[-topn:]
                sim.data[i] = [row[j] for j in idx]
                sim.rows[i] = [cols[j] for j in idx]

        self.sim_matrix = sim.tocsr()

    def predict_user(self, user_idx: int) -> np.ndarray:
        if self.user_item is None or self.sim_matrix is None:
            raise RuntimeError("Call fit() first")
        u_vec = self.user_item[user_idx]
        scores = u_vec @ self.sim_matrix
        denom = np.asarray(np.abs(self.sim_matrix).sum(axis=0)).ravel() + self.config.shrinkage
        score_arr = np.asarray(scores.toarray()).ravel()
        return score_arr / denom

    def recommend(self, user_idx: int, seen_items: Iterable[int], k: int = 10) -> List[int]:
        scores = self.predict_user(user_idx)
        scores[list(seen_items)] = -np.inf
        top_items = np.argpartition(scores, -k)[-k:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]
        return top_items.tolist()
