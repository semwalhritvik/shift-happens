import argparse
import os
import subprocess
import sys

import pandas as pd
from google.cloud import bigquery


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load drift CSV into BigQuery and run drift summary calculation."
    )
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument(
        "--dataset",
        default="ml_observability",
        help="BigQuery dataset name",
    )
    parser.add_argument(
        "--log_table",
        default="prediction_logs",
        help="BigQuery table name for incoming drift/log data",
    )
    parser.add_argument(
        "--drift_table",
        default="drift_summary_daily",
        help="BigQuery table name for drift summary output",
    )
    parser.add_argument(
        "--baseline_table",
        default="training_baseline",
        help="BigQuery training baseline table name",
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
        help="How to write the BigQuery log table.",
    )
    parser.add_argument(
        "--location",
        default="US",
        help="BigQuery dataset location.",
    )
    parser.add_argument(
        "--time_field",
        default="",
        help="Optional time field name on the log table. Leave blank to skip time filtering.",
    )
    return parser.parse_args()


def ensure_dataset(client: bigquery.Client, dataset: str, location: str):
    dataset_ref = client.dataset(dataset)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset already exists: {client.project}.{dataset}")
    except Exception:
        print(f"Creating dataset: {client.project}.{dataset} ({location})")
        dataset_obj = bigquery.Dataset(dataset_ref)
        dataset_obj.location = location
        client.create_dataset(dataset_obj, timeout=30)
        print(f"Created dataset: {client.project}.{dataset}")


def load_csv_to_bigquery(client: bigquery.Client, source_csv: str, table: str, write_disposition: str):
    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    print(f"Loading CSV from {source_csv} into {table}...")
    df = pd.read_csv(source_csv)
    if df.empty:
        raise ValueError("Source CSV is empty.")

    table_id = f"{client.project}.{table}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=getattr(bigquery.WriteDisposition, write_disposition),
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Loaded {len(df)} rows into {table_id}")


def run_drift_job(args):
    env = os.environ.copy()
    env["BQ_PROJECT"] = args.project_id
    env["BQ_DATASET"] = args.dataset
    env["BQ_LOG_TABLE"] = args.log_table
    env["BQ_DRIFT_TABLE"] = args.drift_table
    env["BQ_BASELINE_TABLE"] = args.baseline_table
    if args.time_field:
        env["BQ_TIME_FIELD"] = args.time_field

    script_path = os.path.join("cloud_function", "drift_detection_job.py")
    print(f"Running drift detection job: {script_path}")
    subprocess.run([sys.executable, script_path], check=True, env=env)


def main():
    args = parse_args()
    client = bigquery.Client(project=args.project_id)

    ensure_dataset(client, args.dataset, args.location)

    table_ref = f"{args.dataset}.{args.log_table}"
    load_csv_to_bigquery(client, args.source_csv, table_ref, args.write_disposition)

    run_drift_job(args)

    print("✅ Drift data uploaded and drift summary computed.")


if __name__ == "__main__":
    main()
