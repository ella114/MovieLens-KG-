from __future__ import annotations

import torch
from torch import nn


class SVDBaseline(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        pu = self.user_emb(user_idx)
        qi = self.item_emb(item_idx)
        bu = self.user_bias(user_idx).squeeze(-1)
        bi = self.item_bias(item_idx).squeeze(-1)
        interaction = (pu * qi).sum(dim=-1)
        return self.global_bias + bu + bi + interaction

    @torch.no_grad()
    def all_item_scores(self, user_idx: torch.Tensor) -> torch.Tensor:
        pu = self.user_emb(user_idx)
        scores = pu @ self.item_emb.weight.T
        scores = scores + self.user_bias(user_idx)
        scores = scores + self.item_bias.weight.squeeze(-1)
        scores = scores + self.global_bias
        return scores
