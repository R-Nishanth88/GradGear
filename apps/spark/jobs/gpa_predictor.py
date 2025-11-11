import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg


def main(input_path: str, output_path: str) -> None:
    spark = (
        SparkSession.builder.appName("gradgear-gpa-predictor").getOrCreate()
    )
    df = spark.read.option("header", True).csv(input_path)
    # Stub: simple average per student as baseline
    pred = df.groupBy("student_id").agg(avg(col("grade")).alias("predicted_gpa"))
    pred.write.mode("overwrite").parquet(output_path)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)


