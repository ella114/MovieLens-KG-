from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from config import DEFAULT_PATHS, DEFAULT_TRAIN_CONFIG
from utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MovieLens data")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_PATHS.raw_dir)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PATHS.processed_dir)
    parser.add_argument("--min-user-inter", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_CONFIG.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_TRAIN_CONFIG.val_ratio)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_CONFIG.seed)
    parser.add_argument(
        "--external-metadata",
        type=Path,
        default=None,
        help="Optional CSV with columns like movieId/imdbId/tmdbId + director + actors",
    )
    return parser.parse_args()


def _find_required_file(raw_dir: Path, filename: str) -> Path:
    file_path = raw_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")
    return file_path


def _find_optional_file(raw_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    return None


def _normalize_id_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _select_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lowered:
            return lowered[key]
    return None


def _norm_people(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    txt = str(value).strip()
    if not txt:
        return "Unknown"
    txt = txt.replace(",", "|").replace(";", "|")
    parts = [p.strip() for p in txt.split("|") if p.strip()]
    if not parts:
        return "Unknown"
    return "|".join(parts)


def load_external_metadata(
    metadata_path: Optional[Path],
    raw_dir: Path,
    links_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if metadata_path is None:
        metadata_path = _find_optional_file(
            raw_dir,
            [
                "movie_metadata.csv",
                "tmdb_metadata.csv",
                "imdb_metadata.csv",
                "external_metadata.csv",
            ],
        )
    if metadata_path is None:
        return pd.DataFrame(columns=["movieId", "director", "actors"])

    ext = pd.read_csv(metadata_path)
    if ext.empty:
        return pd.DataFrame(columns=["movieId", "director", "actors"])

    movie_id_col = _normalize_id_col(ext, ["movieId", "movie_id"])
    imdb_col = _normalize_id_col(ext, ["imdbId", "imdb_id"])
    tmdb_col = _normalize_id_col(ext, ["tmdbId", "tmdb_id"])

    director_col = _select_col(ext, ["director", "directors", "crew_director"])
    actors_col = _select_col(ext, ["actors", "cast", "top_cast", "actor"])

    if director_col is None:
        ext["director"] = "Unknown"
    else:
        ext["director"] = ext[director_col].apply(_norm_people)

    if actors_col is None:
        ext["actors"] = "Unknown"
    else:
        ext["actors"] = ext[actors_col].apply(_norm_people)

    if movie_id_col is not None:
        out = ext[[movie_id_col, "director", "actors"]].copy()
        out = out.rename(columns={movie_id_col: "movieId"})
        out["movieId"] = pd.to_numeric(out["movieId"], errors="coerce")
        out = out.dropna(subset=["movieId"]).astype({"movieId": "int64"})
        return out.drop_duplicates("movieId", keep="first")

    if links_df is None:
        return pd.DataFrame(columns=["movieId", "director", "actors"])

    join_df = ext.copy()
    if imdb_col is not None:
        join_df["imdbId"] = pd.to_numeric(join_df[imdb_col], errors="coerce")
    if tmdb_col is not None:
        join_df["tmdbId"] = pd.to_numeric(join_df[tmdb_col], errors="coerce")

    merged = None
    if "imdbId" in join_df.columns:
        merged = links_df.merge(join_df[["imdbId", "director", "actors"]], on="imdbId", how="inner")
    if merged is None and "tmdbId" in join_df.columns:
        merged = links_df.merge(join_df[["tmdbId", "director", "actors"]], on="tmdbId", how="inner")
    if merged is None:
        return pd.DataFrame(columns=["movieId", "director", "actors"])

    out = merged[["movieId", "director", "actors"]].drop_duplicates("movieId", keep="first")
    return out


def split_by_time(
    ratings: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio, 0.9):
        test_ratio = 1.0 - train_ratio - val_ratio
    else:
        test_ratio = 0.1
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train/val/test ratio must be > 0")

    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    n = len(ratings)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = ratings.iloc[:n_train].copy()
    val = ratings.iloc[n_train : n_train + n_val].copy()
    test = ratings.iloc[n_train + n_val :].copy()
    return train, val, test


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    ensure_dir(args.out_dir)

    ratings_path = _find_required_file(args.raw_dir, "ratings.csv")
    movies_path = _find_required_file(args.raw_dir, "movies.csv")
    links_path = _find_optional_file(args.raw_dir, ["links.csv"])

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    required_rating_cols = {"userId", "movieId", "rating"}
    missing = required_rating_cols - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings.csv missing columns: {missing}")

    if "timestamp" not in ratings.columns:
        ratings["timestamp"] = np.arange(len(ratings), dtype=np.int64)

    ratings["movieId"] = ratings["movieId"].astype("int64")
    ratings["userId"] = ratings["userId"].astype("int64")

    user_counts = ratings.groupby("userId")["movieId"].count()
    keep_users = user_counts[user_counts >= args.min_user_inter].index
    ratings = ratings[ratings["userId"].isin(keep_users)].copy()

    movies["movieId"] = movies["movieId"].astype("int64")
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])].copy()

    users = sorted(ratings["userId"].unique().tolist())
    items = sorted(ratings["movieId"].unique().tolist())

    user2idx: Dict[int, int] = {uid: idx for idx, uid in enumerate(users)}
    item2idx: Dict[int, int] = {iid: idx for idx, iid in enumerate(items)}

    ratings["user_idx"] = ratings["userId"].map(user2idx)
    ratings["item_idx"] = ratings["movieId"].map(item2idx)

    train_df, val_df, test_df = split_by_time(ratings, args.train_ratio, args.val_ratio)

    # Save rating matrix (all interactions) for CF/SVD style methods.
    num_users = len(user2idx)
    num_items = len(item2idx)
    mat_all = sparse.coo_matrix(
        (ratings["rating"].to_numpy(), (ratings["user_idx"].to_numpy(), ratings["item_idx"].to_numpy())),
        shape=(num_users, num_items),
    )
    sparse.save_npz(args.out_dir / "rating_matrix_all.npz", mat_all.tocsr())

    mat_train = sparse.coo_matrix(
        (train_df["rating"].to_numpy(), (train_df["user_idx"].to_numpy(), train_df["item_idx"].to_numpy())),
        shape=(num_users, num_items),
    )
    sparse.save_npz(args.out_dir / "rating_matrix_train.npz", mat_train.tocsr())

    train_df.to_csv(args.out_dir / "train.csv", index=False)
    val_df.to_csv(args.out_dir / "val.csv", index=False)
    test_df.to_csv(args.out_dir / "test.csv", index=False)

    links_df = None
    if links_path is not None:
        links_df = pd.read_csv(links_path)
        if "movieId" in links_df.columns:
            links_df["movieId"] = pd.to_numeric(links_df["movieId"], errors="coerce")
            links_df = links_df.dropna(subset=["movieId"]).astype({"movieId": "int64"})
        if "imdbId" in links_df.columns:
            links_df["imdbId"] = pd.to_numeric(links_df["imdbId"], errors="coerce")
        if "tmdbId" in links_df.columns:
            links_df["tmdbId"] = pd.to_numeric(links_df["tmdbId"], errors="coerce")

    ext_meta = load_external_metadata(args.external_metadata, args.raw_dir, links_df)

    movie_side = movies[movies["movieId"].isin(items)][["movieId", "title", "genres"]].copy()
    movie_side["genres"] = movie_side["genres"].fillna("(no genres listed)").astype(str)
    movie_side = movie_side.merge(ext_meta, on="movieId", how="left")
    movie_side["director"] = movie_side["director"].fillna("Unknown").astype(str)
    movie_side["actors"] = movie_side["actors"].fillna("Unknown").astype(str)
    movie_side["item_idx"] = movie_side["movieId"].map(item2idx)
    movie_side = movie_side.sort_values("item_idx").reset_index(drop=True)
    movie_side.to_csv(args.out_dir / "movie_side_info.csv", index=False)

    mapping = {
        "num_users": num_users,
        "num_items": num_items,
        "user2idx": {str(k): v for k, v in user2idx.items()},
        "item2idx": {str(k): v for k, v in item2idx.items()},
    }
    save_json(mapping, args.out_dir / "mappings.json")

    print("Preprocess done:")
    print(f"  users={num_users}, items={num_items}, interactions={len(ratings)}")
    print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"  side_info rows={len(movie_side)}")


if __name__ == "__main__":
    main()
