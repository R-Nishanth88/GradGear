from __future__ import annotations

"""Daily batch job to build domain/difficulty question playlists.

This job joins the trending skill Delta table with the question bank stored in
MongoDB (or exported as Parquet/Delta), ranks questions by a composite score,
then materialises a playlist per domain/level that downstream services can read.
"""

from datetime import date
from typing import Dict

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def compute_playlist_scores(trending_df, questions_df):
    base = trending_df.join(questions_df, ["domain", "skill"], "inner")
    return (
        base.withColumn("trend_score", F.col("freq") * F.col("weight"))
        .withColumn("composite", F.col("trend_score") + F.col("base_score"))
        .groupBy("domain", "level")
        .agg(
            F.collect_list(
                F.struct(
                    "question_id",
                    "composite",
                    "skill",
                    "difficulty",
                )
            ).alias("questions")
        )
    )


def main() -> None:
    spark = SparkSession.builder.appName("gradgear-coding-playlists").getOrCreate()

    trending = spark.read.format("delta").load("delta/trending_skills")
    questions = spark.read.format("mongo").load()

    playlists = compute_playlist_scores(trending, questions)

    today = date.today().isoformat()
    playlists = playlists.withColumn("date", F.lit(today))

    (playlists.write.format("delta").mode("overwrite").save("delta/playlists_daily"))


if __name__ == "__main__":
    main()
