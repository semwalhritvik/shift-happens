import argparse
import os
import pandas as pd
from google.cloud import bigquery


def parse_args():
    parser = argparse.ArgumentParser(description="Load the training baseline into BigQuery for drift comparison.")
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument("--dataset", default="ml_observability", help="BigQuery dataset name")
    parser.add_argument("--table", default="training_baseline", help="BigQuery table name")
    parser.add_argument(
        "--source_pickle",
        default=os.path.join("Data-Pipeline", "data", "processed", "application_train_merged.pkl"),
        help="Path to the training baseline pickle file.",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Optional fraction of baseline rows to load when the dataset is large.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.source_pickle):
        raise FileNotFoundError(
            f"Source pickle not found: {args.source_pickle}.\nRun your pipeline first or point --source_pickle to a valid file."
        )

    print(f"Loading training baseline from {args.source_pickle}...")
    df = pd.read_pickle(args.source_pickle)

    if "TARGET" not in df.columns:
        print("Warning: TARGET column not found in baseline data.")

    if args.sample_fraction < 1.0:
        df = df.sample(frac=args.sample_fraction, random_state=42)
        print(f"Sampling baseline to {len(df)} rows.")

    if "TARGET" in df.columns:
        df = df[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED", "TARGET"]]
    else:
        df = df[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"]]

    client = bigquery.Client(project=args.project_id)
    table_id = f"{args.project_id}.{args.dataset}.{args.table}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    print(f"✅ Training baseline loaded to BigQuery: {table_id} ({len(df)} rows)")


if __name__ == "__main__":
    main()
