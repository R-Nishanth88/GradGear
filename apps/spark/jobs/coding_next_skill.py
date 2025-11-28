from __future__ import annotations

"""Adaptive next-skill recommender using Spark MLlib.

This batch job reads learner telemetry (submissions, progress), trending skills,
resume gaps, and question performance to produce the next skill (and difficulty)
for each learner. The output feeds the `coding_next_skills` Mongo collection
consumed by the API.
"""

from typing import Dict

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


FEATURE_COLUMNS = [
    "trend_score",
    "gap_match",
    "mastery",
    "recent_score",
    "streak",
    "time_to_solve",
]


def load_frames(spark: SparkSession):
    progress = spark.read.format("delta").load("delta/user_progress")
    trends = spark.read.format("delta").load("delta/trending_skills")
    gaps = spark.read.format("delta").load("delta/resume_gaps")
    submissions = spark.read.format("delta").load("delta/coding_submissions")
    return progress, trends, gaps, submissions


def build_training_frame(progress, trends, gaps, submissions):
    joined = (
        progress.join(trends, ["user_id", "domain"], "left")
        .join(gaps, ["user_id", "domain"], "left")
        .join(submissions, ["user_id", "skill"], "left")
    )
    return joined.fillna(0)


def train_model(training_df):
    indexer = StringIndexer(inputCol="skill", outputCol="skill_idx")
    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features")
    classifier = GBTClassifier(labelCol="skill_idx", featuresCol="features", maxIter=20)

    pipeline = Pipeline(stages=[indexer, assembler, classifier])
    model = pipeline.fit(training_df)
    return model


def main() -> None:
    spark = SparkSession.builder.appName("gradgear-next-skill").getOrCreate()

    progress, trends, gaps, submissions = load_frames(spark)
    training_df = build_training_frame(progress, trends, gaps, submissions)

    model = train_model(training_df)

    recommendations = model.transform(training_df)
    ranked = (
        recommendations.groupBy("user_id", "domain")
        .agg(F.collect_list("prediction").alias("predicted_skills"))
        .withColumn("generated_at", F.current_timestamp())
    )

    (ranked.write.format("mongo").mode("overwrite").option("collection", "coding_next_skills").save())


if __name__ == "__main__":
    main()
