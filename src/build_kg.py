from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from config import DEFAULT_PATHS
from utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build movie knowledge graph")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PATHS.processed_dir)
    parser.add_argument("--topk-director", type=int, default=1000)
    parser.add_argument("--topk-actor", type=int, default=3000)
    parser.add_argument("--min-actor-overlap", type=int, default=2)
    parser.add_argument("--genre-jaccard", type=float, default=0.5)
    parser.add_argument("--max-group-size", type=int, default=200)
    parser.add_argument("--w-director", type=float, default=1.0)
    parser.add_argument("--w-actor", type=float, default=0.7)
    parser.add_argument("--w-genre", type=float, default=0.5)
    return parser.parse_args()


def tokenize_pipe(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    parts = [p.strip() for p in value.split("|") if p.strip()]
    return parts


def topk_tokens(series: pd.Series, topk: int) -> List[str]:
    counter = Counter()
    for val in series:
        counter.update(tokenize_pipe(str(val)))
    return [k for k, _ in counter.most_common(topk)]


def build_multi_hot(indexed_tokens: List[List[str]], vocab: List[str]) -> sparse.csr_matrix:
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    rows, cols = [], []
    for r, tokens in enumerate(indexed_tokens):
        seen = set()
        for tok in tokens:
            if tok in tok2idx and tok not in seen:
                rows.append(r)
                cols.append(tok2idx[tok])
                seen.add(tok)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(indexed_tokens), len(vocab)), dtype=np.float32)


def add_bidirectional_edge(
    edges: List[Tuple[int, int, float, str]],
    src: int,
    dst: int,
    weight: float,
    relation: str,
) -> None:
    if src == dst:
        return
    edges.append((src, dst, weight, relation))
    edges.append((dst, src, weight, relation))


def build_relation_edges(
    df: pd.DataFrame,
    min_actor_overlap: int,
    genre_jaccard: float,
    max_group_size: int,
    w_director: float,
    w_actor: float,
    w_genre: float,
) -> List[Tuple[int, int, float, str]]:
    edges: List[Tuple[int, int, float, str]] = []

    item_indices = df["item_idx"].to_numpy()

    director_map: Dict[str, List[int]] = defaultdict(list)
    for item_idx, director_str in zip(item_indices, df["director"].astype(str)):
        for director in tokenize_pipe(director_str):
            if director != "Unknown":
                director_map[director].append(int(item_idx))

    for movies in director_map.values():
        uniq = sorted(set(movies))
        if 2 <= len(uniq) <= max_group_size:
            for i, j in combinations(uniq, 2):
                add_bidirectional_edge(edges, i, j, w_director, "same_director")

    actor_to_movies: Dict[str, List[int]] = defaultdict(list)
    for item_idx, actor_str in zip(item_indices, df["actors"].astype(str)):
        for actor in tokenize_pipe(actor_str):
            if actor != "Unknown":
                actor_to_movies[actor].append(int(item_idx))

    pair_count: Dict[Tuple[int, int], int] = defaultdict(int)
    for movies in actor_to_movies.values():
        uniq = sorted(set(movies))
        if len(uniq) < 2 or len(uniq) > max_group_size:
            continue
        for i, j in combinations(uniq, 2):
            pair_count[(i, j)] += 1

    for (i, j), cnt in pair_count.items():
        if cnt >= min_actor_overlap:
            weight = w_actor * min(1.0, cnt / 5.0)
            add_bidirectional_edge(edges, i, j, float(weight), "same_actor")

    genre_sets: Dict[int, Set[str]] = {
        int(row.item_idx): set(tokenize_pipe(str(row.genres)))
        for row in df.itertuples(index=False)
    }

    genre_to_movies: Dict[str, List[int]] = defaultdict(list)
    for item_idx, genres in genre_sets.items():
        for genre in genres:
            genre_to_movies[genre].append(item_idx)

    shared_genre_count: Dict[Tuple[int, int], int] = defaultdict(int)
    for movies in genre_to_movies.values():
        uniq = sorted(set(movies))
        if len(uniq) < 2 or len(uniq) > max_group_size:
            continue
        for i, j in combinations(uniq, 2):
            shared_genre_count[(i, j)] += 1

    for (i, j), inter in shared_genre_count.items():
        gi = genre_sets.get(i, set())
        gj = genre_sets.get(j, set())
        union = len(gi | gj)
        if union == 0:
            continue
        jac = inter / union
        if jac >= genre_jaccard:
            weight = w_genre * jac
            add_bidirectional_edge(edges, i, j, float(weight), "same_genre")

    return edges


def main() -> None:
    args = parse_args()
    ensure_dir(args.processed_dir)

    movie_side_path = args.processed_dir / "movie_side_info.csv"
    if not movie_side_path.exists():
        raise FileNotFoundError(
            f"Missing {movie_side_path}. Run preprocess.py first."
        )

    movie_side = pd.read_csv(movie_side_path)
    movie_side = movie_side.sort_values("item_idx").reset_index(drop=True)

    genre_tokens = [tokenize_pipe(v) for v in movie_side["genres"].astype(str)]
    director_tokens = [tokenize_pipe(v) for v in movie_side["director"].astype(str)]
    actor_tokens = [tokenize_pipe(v) for v in movie_side["actors"].astype(str)]

    director_vocab = [tok for tok in topk_tokens(movie_side["director"], args.topk_director) if tok != "Unknown"]
    actor_vocab = [tok for tok in topk_tokens(movie_side["actors"], args.topk_actor) if tok != "Unknown"]
    genre_vocab = sorted({t for arr in genre_tokens for t in arr})

    x_genre = build_multi_hot(genre_tokens, genre_vocab)
    x_director = build_multi_hot(director_tokens, director_vocab)
    x_actor = build_multi_hot(actor_tokens, actor_vocab)
    x = sparse.hstack([x_genre, x_director, x_actor], format="csr", dtype=np.float32)

    sparse.save_npz(args.processed_dir / "node_features.npz", x)

    feature_meta = {
        "genre_dim": x_genre.shape[1],
        "director_dim": x_director.shape[1],
        "actor_dim": x_actor.shape[1],
        "feature_dim": x.shape[1],
        "genre_vocab": genre_vocab,
        "director_vocab": director_vocab,
        "actor_vocab": actor_vocab,
    }
    save_json(feature_meta, args.processed_dir / "feature_meta.json")

    edges = build_relation_edges(
        movie_side,
        min_actor_overlap=args.min_actor_overlap,
        genre_jaccard=args.genre_jaccard,
        max_group_size=args.max_group_size,
        w_director=args.w_director,
        w_actor=args.w_actor,
        w_genre=args.w_genre,
    )

    edge_df = pd.DataFrame(edges, columns=["src", "dst", "weight", "relation"])
    edge_df.to_csv(args.processed_dir / "kg_edges.csv", index=False)

    relation_count = edge_df["relation"].value_counts().to_dict() if len(edge_df) else {}
    print("KG build done:")
    print(f"  movies={len(movie_side)}, feature_dim={x.shape[1]}")
    print(f"  edges={len(edge_df)}")
    print(f"  relations={relation_count}")


if __name__ == "__main__":
    main()
