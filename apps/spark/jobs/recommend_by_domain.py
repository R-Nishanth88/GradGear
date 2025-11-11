import argparse
from pyspark.sql import SparkSession


def main(input_projects: str, user_domain: str, output_path: str) -> None:
    spark = SparkSession.builder.appName("gradgear-recommend-by-domain").getOrCreate()
    df = spark.read.option("header", True).csv(input_projects)
    filtered = df.filter(df["domain"] == user_domain)
    # TODO: add ranking / scoring with MLlib; for demo we just write filtered list
    filtered.write.mode("overwrite").parquet(output_path)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", required=True, help="CSV of projects with a 'domain' column")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.projects, args.domain, args.output)


