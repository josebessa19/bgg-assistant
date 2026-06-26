"""Transform bgg-26m-reviews.csv into ratings_implicit schema."""
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

REVIEWS_FILENAME = "bgg-26m-reviews.csv"


def load_reviews_csv(spark: SparkSession, raw_dir: Path) -> DataFrame:
    path = str(raw_dir / REVIEWS_FILENAME)
    return (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .csv(path)
        .select("user", "rating", "ID")
    )


def build_ratings_implicit(reviews_raw: DataFrame, games_features: DataFrame) -> DataFrame:
    reviews_clean = (
        reviews_raw.withColumnRenamed("ID", "game_id")
        .withColumn("game_id", F.col("game_id").cast("int"))
        .withColumn("rating", F.col("rating").cast("double"))
        .withColumn("user_id", F.xxhash64("user").cast("int"))
        .withColumn("implicit_strength", F.col("rating") / 10.0)
        .filter(F.col("game_id").isNotNull())
        .filter(F.col("rating").isNotNull())
        .select("user_id", "game_id", "rating", "implicit_strength")
    )

    valid_ids = games_features.select("game_id")
    return reviews_clean.join(valid_ids, on="game_id", how="inner").dropDuplicates(
        ["user_id", "game_id"]
    )
