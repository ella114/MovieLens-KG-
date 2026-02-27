from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import pandas as pd

from config import DEFAULT_PATHS, DEFAULT_TRAIN_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize recommendation metrics and top-K cases")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PATHS.processed_dir)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_PATHS.outputs_dir)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TRAIN_CONFIG.top_k)
    parser.add_argument("--model-dir", type=str, default="kg_gcn")
    parser.add_argument("--user-idx", type=int, default=None)
    return parser.parse_args()


def tokenize_pipe(value: str) -> Set[str]:
    if not isinstance(value, str):
        return set()
    return {x.strip() for x in value.split("|") if x.strip()}


def build_user_profile(train_df: pd.DataFrame, movie_info: pd.DataFrame, user_idx: int, positive_th: float) -> Dict[str, Set[str]]:
    pos = train_df[(train_df["user_idx"] == user_idx) & (train_df["rating"] >= positive_th)]
    liked_items = set(pos["item_idx"].tolist())

    subset = movie_info[movie_info["item_idx"].isin(liked_items)]
    genres = set()
    directors = set()
    actors = set()
    for row in subset.itertuples(index=False):
        genres |= tokenize_pipe(str(row.genres))
        directors |= tokenize_pipe(str(row.director))
        actors |= tokenize_pipe(str(row.actors))
    return {"genres": genres, "directors": directors, "actors": actors}


def explain_reason(profile: Dict[str, Set[str]], row: pd.Series) -> str:
    row_directors = tokenize_pipe(str(row.get("director", "")))
    row_actors = tokenize_pipe(str(row.get("actors", "")))
    row_genres = tokenize_pipe(str(row.get("genres", "")))

    if profile["directors"] & row_directors:
        return "同导演（基于知识图谱关系）"
    if profile["actors"] & row_actors:
        return "同演员（基于知识图谱关系）"
    if profile["genres"] & row_genres:
        return "同类型（基于知识图谱关系）"
    return "图嵌入语义相近"


def plot_metrics(comp_df: pd.DataFrame, top_k: int, out_path: Path) -> None:
    model_names = comp_df["model"].tolist()
    x = range(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(x, comp_df["test_rmse"].to_numpy(), color="#1f77b4")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(model_names, rotation=20)
    axes[0].set_title("Test RMSE")
    axes[0].set_ylabel("RMSE")

    p_col = f"precision@{top_k}"
    r_col = f"recall@{top_k}"
    f_col = f"f1@{top_k}"

    width = 0.25
    axes[1].bar([i - width for i in x], comp_df[p_col].to_numpy(), width=width, label=p_col)
    axes[1].bar(x, comp_df[r_col].to_numpy(), width=width, label=r_col)
    axes[1].bar([i + width for i in x], comp_df[f_col].to_numpy(), width=width, label=f_col)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(model_names, rotation=20)
    axes[1].set_title(f"Top-{top_k} Ranking Metrics")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    comp_path = args.outputs_dir / "metrics_comparison.csv"
    if not comp_path.exists():
        raise FileNotFoundError(f"Missing {comp_path}, run evaluate.py first.")

    comp_df = pd.read_csv(comp_path)
    plot_metrics(comp_df, args.top_k, args.outputs_dir / "figures" / "metrics_comparison.png")

    rec_path = args.outputs_dir / args.model_dir / "topk_recommendations.csv"
    if not rec_path.exists():
        raise FileNotFoundError(f"Missing {rec_path}, run training first.")

    rec_df = pd.read_csv(rec_path)
    movie_info = pd.read_csv(args.processed_dir / "movie_side_info.csv")
    train_df = pd.read_csv(args.processed_dir / "train.csv")

    if args.user_idx is None:
        args.user_idx = int(rec_df["user_idx"].iloc[0])

    user_rec = rec_df[rec_df["user_idx"] == args.user_idx].sort_values("rank").head(args.top_k)
    if user_rec.empty:
        raise ValueError(f"No recommendations found for user_idx={args.user_idx}")

    merged = user_rec.merge(movie_info, on="item_idx", how="left")
    profile = build_user_profile(train_df, movie_info, args.user_idx, DEFAULT_TRAIN_CONFIG.positive_threshold)
    merged["reason"] = merged.apply(lambda row: explain_reason(profile, row), axis=1)

    out_cols = ["user_idx", "rank", "title", "score", "genres", "director", "actors", "reason"]
    out_df = merged[out_cols]
    out_df.to_csv(args.outputs_dir / args.model_dir / f"top{args.top_k}_with_reasons_user{args.user_idx}.csv", index=False)

    print("Visualization done:")
    print(f"  metric figure: {args.outputs_dir / 'figures' / 'metrics_comparison.png'}")
    print(
        f"  top-{args.top_k} table: {args.outputs_dir / args.model_dir / f'top{args.top_k}_with_reasons_user{args.user_idx}.csv'}"
    )


if __name__ == "__main__":
    main()
