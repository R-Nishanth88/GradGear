from __future__ import annotations

"""Batch job to generate trending skills and user skill gap profiles.

This script can be run locally with `spark-submit` once PySpark and delta-spark
are installed. For demo purposes we use the bundled sample CSVs and push
aggregated results into MongoDB so the FastAPI application can surface the
latest intelligence.
"""

from pathlib import Path
from typing import Dict, List
import os

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pymongo import MongoClient

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
JOB_POSTINGS = DATA_DIR / "job_postings_sample.csv"
RESUME_SKILLS = DATA_DIR / "user_resume_skills.csv"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGODB_DB_NAME", os.getenv("MONGO_DB", "gradgear"))


def bootstrap_spark() -> SparkSession:
    builder = (
        SparkSession.builder.appName("gradgear-intelligence-batch")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.mongodb.write.connection.uri", MONGO_URI)
        .config("spark.mongodb.read.connection.uri", MONGO_URI)
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def extract_trending_skills(spark: SparkSession):
    postings_df = spark.read.option("header", True).csv(str(JOB_POSTINGS))

    tokenised = postings_df.select(
        F.regexp_replace(F.lower(F.col("domain")), r"[^a-z0-9]+", "_").alias("domain"),
        F.lower(F.col("description")).alias("description"),
    )

    skills = tokenised.select(
        "domain",
        F.explode(F.split(F.col("description"), r"[\s,;/]+")).alias("token"),
    ).where(F.length("token") > 2)

    keywords = (
        skills.groupBy("domain", "token")
        .agg(F.count("*").alias("freq"))
        .withColumn("rank", F.row_number().over(Window.partitionBy("domain").orderBy(F.desc("freq"))))
        .where(F.col("rank") <= 15)
        .groupBy("domain")
        .agg(
            F.collect_list(F.struct("token", "freq")).alias("skills"),
            F.first(F.current_timestamp()).alias("generated_at"),
        )
    )

    docs: List[Dict] = []
    for row in keywords.collect():
        record = row.asDict(recursive=True)
        skills = [dict(skill) for skill in record.get("skills", [])]
        docs.append({
            "domain": record["domain"],
            "skills": skills,
            "generated_at": record.get("generated_at"),
        })
    write_documents("trending_skills", docs)


def build_skill_gaps(spark: SparkSession):
    if not RESUME_SKILLS.exists():
        return

    resumes = spark.read.option("header", True).csv(str(RESUME_SKILLS))
    postings = spark.read.option("header", True).csv(str(JOB_POSTINGS))

    postings_skills = postings.select(
        F.regexp_replace(F.lower(F.col("domain")), r"[^a-z0-9]+", "_").alias("domain"),
        F.lower(F.col("description")).alias("description"),
    )

    domain_skill_counts = (
        postings_skills.select("domain", F.explode(F.split("description", r"[\s,;/]+")).alias("token"))
        .groupBy("domain", "token")
        .agg(F.count("*").alias("freq"))
        .filter(F.col("freq") > 1)
    )

    resume_tokens = resumes.select(
        F.col("user_id"),
        F.regexp_replace(F.lower(F.col("domain")), r"[^a-z0-9]+", "_").alias("domain"),
        F.lower(F.col("skill")).alias("skill"),
    )

    gaps = (
        domain_skill_counts.alias("d")
        .join(
            resume_tokens.alias("r"),
            (F.col("d.domain") == F.col("r.domain")) & (F.col("d.token") == F.col("r.skill")),
            how="left",
        )
        .where(F.col("r.skill").isNull())
        .groupBy("d.domain")
        .agg(F.collect_list("d.token").alias("missing"))
    )

    user_gaps = (
        resume_tokens.groupBy("user_id", "domain").agg(F.collect_set("skill").alias("skills"))
        .alias("resume")
        .join(gaps.alias("gap"), F.col("resume.domain") == F.col("gap.domain"), how="left")
        .select(
            F.col("user_id"),
            F.col("resume.domain").alias("domain"),
            F.coalesce(F.col("gap.missing"), F.array()).alias("missing_skills"),
            F.current_timestamp().alias("generated_at"),
        )
    )

    docs: List[Dict] = []
    for row in user_gaps.collect():
        record = row.asDict(recursive=True)
        docs.append({
            "user_id": int(record["user_id"]),
            "domain": record["domain"],
            "missing_skills": record.get("missing_skills", []),
            "generated_at": record.get("generated_at"),
        })
    write_documents("skill_gap_profiles", docs)


def write_documents(collection: str, documents: List[Dict]):
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    coll = db[collection]
    coll.delete_many({})
    if documents:
        coll.insert_many(documents)
    client.close()


def main() -> None:
    spark = bootstrap_spark()
    extract_trending_skills(spark)
    build_skill_gaps(spark)
    spark.stop()


if __name__ == "__main__":
    main()
