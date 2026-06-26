# Step 02 — Data & EDA

**Week:** 1 | **Est. time:** 3–6h | **Depends on:** [01-project-setup](01-project-setup.md) | **Blocks:** [03-recommender](03-recommender.md)

Explore the BGG dataset with PySpark, define the canonical feature schema, and export Parquet for downstream training and API lookup.

---

## Objectives

- `notebooks/01_eda.ipynb` runs end-to-end on laptop
- `scripts/preprocess_to_parquet.py` reproduces notebook exports
- `games_features.parquet` and `ratings_implicit.parquet` in `data/processed/`
- Documented schema with ≥ 20k games

---

## Tasks

### Task 1: SparkSession setup

**Subtasks:**

1. Create `notebooks/01_eda.ipynb`
2. First cell — use shared Spark builder (1g driver, Windows-safe):

```python
from bgg.preprocessing.spark import build_spark

spark = build_spark("bgg-eda")
spark.sparkContext.setLogLevel("WARN")
```

On **32-bit Java** (common on Windows), do not set `spark.driver.memory` above `1g` — larger values fail with `Invalid maximum heap size`.

3. Load raw CSVs from `data/raw/` — primary games file: `games_detailed_info2025.csv` (columns: `id`, `name`, `minplayers`, `maxplayers`, `playingtime`, `averageweight`, `average`, `bayesaverage`, `usersrated`, `boardgamemechanic`, `boardgamecategory`, …). Ratings file: `bgg-26m-reviews.csv` (`user`, `rating`, `ID`; drop `comment` and `name` when loading).
4. Print schema and row counts for each table

**Checkpoint:**

- [ ] SparkSession starts without OOM on laptop (`build_spark()` uses `1g` by default)

---

### Task 2: Exploratory analysis

**Subtasks:**

1. **Games overview**
   - Count distinct games
   - Distribution of `minplayers`, `maxplayers`, `playingtime`
   - Top 20 games by `bayesaverage` or `average` rating
   - Missing value counts per column

2. **Mechanics / categories**
   - If separate tables: join on `game_id`
   - If pipe-delimited in games table: `split` and `explode`
   - Count top 30 mechanics

3. **Ratings sparsity** (justifies ALS + content hybrid)
   - Number of unique users and games in ratings
   - Sparsity: `1 - (num_ratings / (num_users * num_games))`
   - Ratings per game distribution (histogram data)

4. **2-player filter analysis** (primary use case)
   - Count games where `minplayers <= 2 <= maxplayers`
   - Average playing time for 2-player-capable games

5. **Cooperative flag**
   - Identify mechanic/category `Cooperative Game` for session filter

**Checkpoint:**

- [ ] Notebook contains at least 4 visualizations or summary tables
- [ ] Sparsity metric computed and noted (expect > 99%)

---

### Task 3: Define canonical schema

**Subtasks:**

Document and implement cleaning for `games_features`:

| Column | Type | Source / transform |
|--------|------|-------------------|
| `game_id` | int | BGG `id` |
| `name` | string | `name` |
| `min_players` | int | `minplayers` |
| `max_players` | int | `maxplayers` |
| `playing_time` | int | `playingtime` (minutes) |
| `min_play_time` | int | `minplaytime` if available, else `playingtime` |
| `max_play_time` | int | `maxplaytime` if available, else `playingtime` |
| `avg_weight` | float | `averageweight` |
| `avg_rating` | float | `average` |
| `bayes_average` | float | `bayesaverage` |
| `num_ratings` | int | `usersrated` |
| `year_published` | int | `yearpublished` |
| `mechanics` | array&lt;string&gt; | joined mechanics |
| `categories` | array&lt;string&gt; | joined categories |
| `is_cooperative` | boolean | `array_contains(mechanics, "Cooperative Game")` |
| `designers` | array&lt;string&gt; | optional |
| `publishers` | array&lt;string&gt; | optional |

For `ratings_implicit`:

| Column | Type | Transform |
|--------|------|-----------|
| `user_id` | int | raw user id |
| `game_id` | int | raw game id |
| `rating` | float | original 1–10 scale |
| `implicit_strength` | float | `rating / 10.0` or binary `(rating >= 7)` |

**Checkpoint:**

- [ ] Schema table added to bottom of this step file or notebook markdown cell

---

### Task 4: Preprocessing pipeline

**Subtasks:**

1. Implement `scripts/preprocess_to_parquet.py`:

```python
"""Batch preprocess BGG raw CSVs to Parquet."""
import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from bgg.config import settings

def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("bgg-preprocess")
        .master(settings.spark_master)
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

def main(raw_dir: str, output_dir: str) -> None:
    spark = build_spark()
    # TODO: load CSVs, join, clean, write parquet
    # games_features.write.mode("overwrite").parquet(f"{output_dir}/games_features.parquet")
    # ratings_implicit.write.mode("overwrite").parquet(f"{output_dir}/ratings_implicit.parquet")
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(settings.raw_dir))
    parser.add_argument("--output-dir", default=str(settings.processed_dir))
    args = parser.parse_args()
    main(args.raw_dir, args.output_dir)
```

2. Key transforms:
   - Drop games with null `name` or `game_id`
   - Cast numerics; clamp `min_players` / `max_players` to sensible range (1–20)
   - Filter ratings to games present in `games_features`
   - Deduplicate ratings on `(user_id, game_id)`

3. Write Parquet:
   - `data/processed/games_features.parquet`
   - `data/processed/ratings_implicit.parquet`

4. Run: `python scripts/preprocess_to_parquet.py`

**Checkpoint:**

- [ ] `games_features` row count ≥ 20,000
- [ ] `ratings_implicit` row count > 0
- [ ] Re-running script is idempotent (overwrite mode)

---

### Task 5: Validate exports

**Subtasks:**

1. Notebook or script validation cell:

```python
games = spark.read.parquet("data/processed/games_features.parquet")
ratings = spark.read.parquet("data/processed/ratings_implicit.parquet")

assert games.count() >= 20_000
assert games.filter("min_players <= 2 AND max_players >= 2").count() > 0
assert ratings.select("game_id").distinct().count() > 1000
```

2. Export a small `data/processed/game_lookup.json` (id → name) for Streamlit search (top 5k by bayes_average or full set if small enough)

**Checkpoint:**

- [ ] All assertions pass
- [ ] Sample rows printed for manual sanity check

---

## Notebook Outline (`01_eda.ipynb`)

| Section | Cells |
|---------|-------|
| 1. Setup | SparkSession, paths |
| 2. Load raw | CSV load, schema inspect |
| 3. Games EDA | counts, distributions, nulls |
| 4. Mechanics | top mechanics, coop flag |
| 5. Ratings | sparsity, histogram |
| 6. 2-player slice | filter analysis |
| 7. Export | write parquet, validate |
| 8. Conclusions | markdown: key findings for recommender design |

---

## Definition of Done

- [ ] Notebook runs end-to-end on laptop
- [ ] Parquet row count ≥ 20k games
- [ ] Schema documented
- [ ] `preprocess_to_parquet.py` matches notebook logic

Mark Step 02 complete in [docs/README.md](../README.md).

**Next:** [03-recommender](03-recommender.md)

---

## Canonical schema (implemented)

Source files: `games_detailed_info2025.csv`, `bgg-26m-reviews.csv`

### `games_features`

| Column | Type | Source / transform |
|--------|------|-------------------|
| `game_id` | int | `id` |
| `name` | string | `name` |
| `min_players` | int | `minplayers`; clamp 1–20 |
| `max_players` | int | `maxplayers`; clamp 1–20 |
| `playing_time` | int | `playingtime` |
| `min_play_time` | int | `minplaytime` or `playingtime` |
| `max_play_time` | int | `maxplaytime` or `playingtime` |
| `avg_weight` | float | `averageweight` |
| `avg_rating` | float | `average` |
| `bayes_average` | float | `bayesaverage` |
| `num_ratings` | int | `usersrated` |
| `year_published` | int | `yearpublished` (1900–2026) |
| `mechanics` | array&lt;string&gt; | Spark `regexp_replace` + `split` on `boardgamemechanic` |
| `categories` | array&lt;string&gt; | Spark `regexp_replace` + `split` on `boardgamecategory` |
| `is_cooperative` | boolean | `array_contains(mechanics, "Cooperative Game")` |
| `designers` | array&lt;string&gt; | `boardgamedesigner` (parsed list) |
| `publishers` | array&lt;string&gt; | `boardgamepublisher` (parsed list) |

### `ratings_implicit`

| Column | Type | Transform |
|--------|------|-----------|
| `user_id` | int | `hash_pandas_object(user)` (chunked export) |
| `game_id` | int | `ID` (join to `games_features.game_id`) |
| `rating` | float | 1–10 scale |
| `implicit_strength` | float | `rating / 10.0` |

Implementation: `src/bgg/preprocessing/games.py`, `src/bgg/preprocessing/ratings.py`, `src/bgg/preprocessing/export.py`, `scripts/preprocess_to_parquet.py`

Note: Spark transforms games in-memory; Parquet export uses PyArrow (avoids Windows Hadoop NativeIO issues). Reviews are written in pandas chunks.

---

## Reference

- [Pitfalls — ALS cold start](../architecture/pitfalls-and-mitigations.md#3-als-cold-start-few-user-ratings)
- [Folder structure — data/processed](../architecture/folder-structure.md)
