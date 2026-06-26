import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from bgg.config import settings

_HADOOP_HOME = Path(__file__).resolve().parents[3] / "tools" / "hadoop"
if _HADOOP_HOME.exists():
    os.environ.setdefault("HADOOP_HOME", str(_HADOOP_HOME))
    os.environ.setdefault("hadoop.home.dir", str(_HADOOP_HOME))

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def build_spark(app_name: str = "bgg-preprocess") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master(settings.spark_master)
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .getOrCreate()
    )
