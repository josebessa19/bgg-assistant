"""Parquet export helpers that avoid Hadoop NativeIO issues on Windows."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame

REVIEWS_CHUNK_SIZE = 500_000


def write_spark_df_parquet(df: DataFrame, output_path: Path) -> None:
    pdf = df.toPandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_dir():
            import shutil

            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    pq.write_table(pa.Table.from_pandas(pdf), output_path, compression="snappy")


def write_ratings_parquet_chunked(
    reviews_path: Path,
    valid_game_ids: set[int],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_dir():
            import shutil

            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for chunk in pd.read_csv(
        reviews_path,
        usecols=["user", "rating", "ID"],
        chunksize=REVIEWS_CHUNK_SIZE,
    ):
        chunk = chunk.dropna(subset=["user", "rating", "ID"])
        chunk["game_id"] = pd.to_numeric(chunk["ID"], errors="coerce").astype("Int64")
        chunk = chunk[chunk["game_id"].isin(valid_game_ids)]
        if chunk.empty:
            continue

        chunk["game_id"] = chunk["game_id"].astype("int32")
        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
        chunk = chunk.dropna(subset=["rating"])
        chunk["user_id"] = pd.util.hash_pandas_object(chunk["user"], index=False).astype("int64")
        chunk["implicit_strength"] = chunk["rating"] / 10.0
        chunk = chunk[["user_id", "game_id", "rating", "implicit_strength"]].drop_duplicates(
            subset=["user_id", "game_id"]
        )

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)
        total_rows += len(chunk)

    if writer is not None:
        writer.close()
    return total_rows
