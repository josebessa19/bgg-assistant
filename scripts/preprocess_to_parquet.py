"""Batch preprocess BGG raw CSVs to Parquet."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from bgg.config import settings
from bgg.preprocessing.export import write_ratings_parquet_chunked, write_spark_df_parquet
from bgg.preprocessing.games import build_games_features, load_games_csv
from bgg.preprocessing.ratings import REVIEWS_FILENAME
from bgg.preprocessing.spark import build_spark

LOOKUP_LIMIT = 5_000


def validate_exports(output_dir: Path) -> None:
    games_path = output_dir / "games_features.parquet"
    ratings_path = output_dir / "ratings_implicit.parquet"

    games_pdf = pq.read_table(games_path).to_pandas()
    ratings_pdf = pq.read_table(ratings_path).to_pandas()

    games_count = len(games_pdf)
    two_player_count = len(
        games_pdf[(games_pdf["min_players"] <= 2) & (games_pdf["max_players"] >= 2)]
    )
    ratings_count = len(ratings_pdf)
    distinct_games = ratings_pdf["game_id"].nunique()

    print(f"games_features rows: {games_count}")
    print(f"2-player capable games: {two_player_count}")
    print(f"ratings_implicit rows: {ratings_count}")
    print(f"distinct rated games: {distinct_games}")

    assert games_count >= 20_000, f"Expected >= 20k games, got {games_count}"
    assert two_player_count > 0, "Expected 2-player capable games"
    assert ratings_count > 0, "Expected ratings rows"
    assert distinct_games > 1_000, f"Expected > 1k rated games, got {distinct_games}"

    print("Validation passed.")
    print("Sample games:")
    print(games_pdf[["game_id", "name", "min_players", "max_players", "bayes_average"]].head())
    print("Sample ratings:")
    print(ratings_pdf.head())


def main(raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spark = build_spark()

    try:
        games_raw = load_games_csv(spark, raw_dir)
        games_features = build_games_features(games_raw)

        games_path = output_dir / "games_features.parquet"
        ratings_path = output_dir / "ratings_implicit.parquet"

        print("Writing games_features.parquet ...")
        write_spark_df_parquet(games_features, games_path)

        valid_ids = set(games_features.select("game_id").toPandas()["game_id"].tolist())
        print(f"Writing ratings_implicit.parquet (filtering to {len(valid_ids):,} game ids) ...")
        filtered_count = write_ratings_parquet_chunked(
            raw_dir / REVIEWS_FILENAME,
            valid_ids,
            ratings_path,
        )
        print(f"Filtered reviews (in catalog): {filtered_count:,}")

        validate_exports(output_dir)

        games_pdf = pq.read_table(games_path).to_pandas()
        lookup_rows = (
            games_pdf.sort_values("bayes_average", ascending=False, na_position="last")
            .head(LOOKUP_LIMIT)
        )
        lookup = {str(row.game_id): row.name for row in lookup_rows.itertuples()}
        lookup_path = output_dir / "game_lookup.json"
        lookup_path.write_text(
            json.dumps(lookup, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote {len(lookup)} entries to {lookup_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BGG raw CSVs to Parquet.")
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir)
    parser.add_argument("--output-dir", type=Path, default=settings.processed_dir)
    args = parser.parse_args()
    main(args.raw_dir, args.output_dir)
