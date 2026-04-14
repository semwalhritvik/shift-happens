import argparse
import os
import pandas as pd
from google.cloud import bigquery


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load drift demo CSV data into BigQuery."
    )
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument(
        "--dataset",
        default="ml_observability",
        help="BigQuery dataset name",
    )
    parser.add_argument(
        "--table",
        default="prediction_logs",
        help="BigQuery table name",
    )
    parser.add_argument(
        "--source_csv",
        default=os.path.join("demo-environment", "uploads", "test_data_drifted.csv"),
        help="Path to the drift CSV file.",
    )
    parser.add_argument(
        "--write_disposition",
        default="WRITE_TRUNCATE",
        choices=["WRITE_TRUNCATE", "WRITE_APPEND"],
        help="How to write the BigQuery table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.source_csv):
        raise FileNotFoundError(
            f"Source CSV not found: {args.source_csv}\n"
            "Run generate_drift.py first or point --source_csv to a valid file."
        )

    print(f"Loading drift data from {args.source_csv}...")
    df = pd.read_csv(args.source_csv)

    if df.empty:
        raise ValueError("Source CSV is empty.")

    client = bigquery.Client(project=args.project_id)
    table_id = f"{args.project_id}.{args.dataset}.{args.table}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=getattr(bigquery.WriteDisposition, args.write_disposition),
        autodetect=True,
    )

    print(f"Writing {len(df)} rows to BigQuery table {table_id}...")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    print(f"✅ Drift CSV loaded to BigQuery: {table_id} ({len(df)} rows)")


if __name__ == "__main__":
    main()
