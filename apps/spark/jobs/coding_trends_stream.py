from __future__ import annotations

"""Spark Structured Streaming job to capture trending skills from job postings.

This job listens to a Kafka topic (or any streaming source) that emits job
postings, extracts domain-specific skill tokens, aggregates their frequency
within a sliding window, and persists the results as a Delta table that the
application can query.

Run (example):
    spark-submit coding_trends_stream.py \
        --kafka-bootstrap localhost:9092 \
        --topic gradgear.jobs
"""

import argparse
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, from_json, window
from pyspark.sql.types import StructType, StructField, StringType, TimestampType


def build_job_schema() -> StructType:
    return StructType(
        [
            StructField("id", StringType(), False),
            StructField("domain", StringType(), True),
            StructField("title", StringType(), True),
            StructField("description", StringType(), True),
            StructField("location", StringType(), True),
            StructField("timestamp", TimestampType(), True),
        ]
    )


def extract_skills(df, nlp_model):
    # Placeholder for custom NLP pipeline.
    # Replace with Spark NLP or custom UDF that yields (domain, skill, timestamp).
    return df.select(col("domain"), col("timestamp"), col("description").alias("skill"))


def main(args: argparse.Namespace) -> None:
    spark = (
        SparkSession.builder.appName("gradgear-coding-trends")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    schema = build_job_schema()

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.kafka_bootstrap)
        .option("subscribe", args.topic)
        .load()
    )

    parsed = raw.select(from_json(col("value").cast("string"), schema).alias("job")).select("job.*")
    skill_events = extract_skills(parsed, nlp_model=None)

    trends = (
        skill_events.groupBy(
            window(col("timestamp"), args.window), col("domain"), col("skill")
        ).agg(count("*").alias("freq"))
    )

    query = (
        trends.writeStream.outputMode("complete")
        .format("delta")
        .option("checkpointLocation", str(Path(args.checkpoint).absolute()))
        .start(str(Path(args.output).absolute()))
    )

    query.awaitTermination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GradGear coding trend stream job")
    parser.add_argument("--kafka-bootstrap", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output", default="delta/trending_skills")
    parser.add_argument("--checkpoint", default="delta/_checkpoints/coding_trends")
    parser.add_argument("--window", default="7 days")
    main(parser.parse_args())
