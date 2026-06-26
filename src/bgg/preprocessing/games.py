"""Transform games_detailed_info2025.csv into games_features schema."""
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

GAMES_FILENAME = "games_detailed_info2025.csv"
CURRENT_YEAR = 2026


def _parse_list_column(column: F.Column) -> F.Column:
    """Parse Python-list strings like \"['A', 'B']\" into array<string>."""
    stripped = F.trim(column)
    inner = F.regexp_replace(F.regexp_replace(stripped, r"^\[|\]$", ""), r"'", "")
    return F.when(
        stripped.isNull() | (stripped == "") | (stripped == "[]"),
        F.array().cast("array<string>"),
    ).otherwise(F.split(inner, r",\s*"))


def load_games_csv(spark: SparkSession, raw_dir: Path) -> DataFrame:
    path = str(raw_dir / GAMES_FILENAME)
    return spark.read.option("header", True).option("inferSchema", False).csv(path)


def build_games_features(games_raw: DataFrame) -> DataFrame:
    mechanics = _parse_list_column(F.col("boardgamemechanic"))
    categories = _parse_list_column(F.col("boardgamecategory"))
    designers = _parse_list_column(F.col("boardgamedesigner"))
    publishers = _parse_list_column(F.col("boardgamepublisher"))

    playing_time = F.col("playingtime").cast("int")
    min_play_time = F.coalesce(F.col("minplaytime").cast("int"), playing_time)
    max_play_time = F.coalesce(F.col("maxplaytime").cast("int"), playing_time)

    return (
        games_raw.withColumn("mechanics", mechanics)
        .withColumn("categories", categories)
        .withColumn("game_id", F.col("id").cast("int"))
        .withColumn("name", F.trim(F.col("name")))
        .withColumn(
            "min_players",
            F.least(F.greatest(F.col("minplayers").cast("int"), F.lit(1)), F.lit(20)),
        )
        .withColumn(
            "max_players",
            F.least(F.greatest(F.col("maxplayers").cast("int"), F.lit(1)), F.lit(20)),
        )
        .withColumn("playing_time", playing_time)
        .withColumn("min_play_time", min_play_time)
        .withColumn("max_play_time", max_play_time)
        .withColumn("avg_weight", F.col("averageweight").cast("double"))
        .withColumn("avg_rating", F.col("average").cast("double"))
        .withColumn("bayes_average", F.col("bayesaverage").cast("double"))
        .withColumn("num_ratings", F.col("usersrated").cast("int"))
        .withColumn("year_published", F.col("yearpublished").cast("int"))
        .withColumn(
            "is_cooperative",
            F.array_contains(F.col("mechanics"), "Cooperative Game"),
        )
        .withColumn("designers", designers)
        .withColumn("publishers", publishers)
        .filter(F.col("game_id").isNotNull())
        .filter(F.col("name").isNotNull() & (F.length(F.col("name")) > 0))
        .filter(
            (F.col("year_published").isNull())
            | ((F.col("year_published") >= 1900) & (F.col("year_published") <= CURRENT_YEAR))
        )
        .select(
            "game_id",
            "name",
            "min_players",
            "max_players",
            "playing_time",
            "min_play_time",
            "max_play_time",
            "avg_weight",
            "avg_rating",
            "bayes_average",
            "num_ratings",
            "year_published",
            "mechanics",
            "categories",
            "is_cooperative",
            "designers",
            "publishers",
        )
    )
