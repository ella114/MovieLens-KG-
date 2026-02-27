from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from config import DEFAULT_PATHS


TMDB_BASE = "https://api.themoviedb.org/3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch movie director/cast metadata from TMDB")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_PATHS.raw_dir)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_PATHS.raw_dir / "tmdb_metadata.csv")
    parser.add_argument("--api-key", type=str, default=os.getenv("TMDB_API_KEY", ""))
    parser.add_argument("--max-cast", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def get_json(path: str, api_key: str) -> Optional[Dict]:
    url = f"{TMDB_BASE}{path}"
    params = {"api_key": api_key, "language": "en-US"}
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        return None
    return resp.json()


def extract_director(crew: List[Dict]) -> str:
    directors = [c.get("name", "").strip() for c in crew if c.get("job") == "Director"]
    directors = [d for d in directors if d]
    return "|".join(directors) if directors else "Unknown"


def extract_cast(cast_list: List[Dict], max_cast: int) -> str:
    actors = [c.get("name", "").strip() for c in cast_list[:max_cast]]
    actors = [a for a in actors if a]
    return "|".join(actors) if actors else "Unknown"


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("TMDB API key is required. Pass --api-key or set TMDB_API_KEY.")

    links_path = args.raw_dir / "links.csv"
    if not links_path.exists():
        raise FileNotFoundError(f"Missing {links_path}")

    links_df = pd.read_csv(links_path)
    if "movieId" not in links_df.columns or "tmdbId" not in links_df.columns:
        raise ValueError("links.csv must include movieId and tmdbId columns")

    links_df = links_df.dropna(subset=["tmdbId"]).copy()
    links_df["tmdbId"] = pd.to_numeric(links_df["tmdbId"], errors="coerce")
    links_df = links_df.dropna(subset=["tmdbId"]).astype({"tmdbId": "int64", "movieId": "int64"})

    if args.limit > 0:
        links_df = links_df.head(args.limit)

    rows = []
    for row in tqdm(list(links_df.itertuples(index=False)), desc="Fetching TMDB"):
        tmdb_id = int(row.tmdbId)
        movie_id = int(row.movieId)

        detail = get_json(f"/movie/{tmdb_id}", args.api_key)
        credits = get_json(f"/movie/{tmdb_id}/credits", args.api_key)
        if detail is None and credits is None:
            continue

        genres = "Unknown"
        if detail is not None and "genres" in detail:
            genre_names = [g.get("name", "").strip() for g in detail.get("genres", [])]
            genre_names = [g for g in genre_names if g]
            if genre_names:
                genres = "|".join(genre_names)

        director = "Unknown"
        actors = "Unknown"
        if credits is not None:
            director = extract_director(credits.get("crew", []))
            actors = extract_cast(credits.get("cast", []), max_cast=args.max_cast)

        rows.append(
            {
                "movieId": movie_id,
                "tmdbId": tmdb_id,
                "director": director,
                "actors": actors,
                "genres_ext": genres,
            }
        )
        time.sleep(args.sleep)

    out_df = pd.DataFrame(rows)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_file, index=False)

    print(f"Saved metadata: {args.out_file}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
