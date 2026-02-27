from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = torch.sparse.mm(adj, x)
        return self.linear(x)


class KGGCNRecommender(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.item_id_emb = nn.Embedding(num_items, hidden_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList(
            GCNLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_id_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def compute_item_embeddings(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.feature_proj(node_features)
        residual = h
        for layer in self.gcn_layers:
            h = layer(h, adj)
            h = F.relu(h)
            h = self.layer_norm(h)
        h = self.dropout(h)
        return h + residual + self.item_id_emb.weight

    def score(self, user_idx: torch.Tensor, item_idx: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        i = item_emb[item_idx]
        bu = self.user_bias(user_idx).squeeze(-1)
        bi = self.item_bias(item_idx).squeeze(-1)
        return self.global_bias + bu + bi + (u * i).sum(dim=-1)

    def bpr_loss(
        self,
        user_idx: torch.Tensor,
        pos_item_idx: torch.Tensor,
        neg_item_idx: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        pos_score = self.score(user_idx, pos_item_idx, item_emb)
        neg_score = self.score(user_idx, neg_item_idx, item_emb)
        return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

    @torch.no_grad()
    def all_item_scores(self, user_idx: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        scores = u @ item_emb.T
        scores = scores + self.user_bias(user_idx)
        scores = scores + self.item_bias.weight.squeeze(-1)
        scores = scores + self.global_bias
        return scores
