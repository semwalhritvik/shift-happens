& "C:\Northeastern Univesity\Academic\MLOps\Project\shift-happens-main\SDK\observability-sdk\.venv\Scripts\python.exe" .\cloud_function\drift_detection_job.py& "C:\Northeastern Univesity\Academic\MLOps\Project\shift-happens-main\SDK\observability-sdk\.venv\Scripts\python.exe" .\cloud_function\drift_detection_job.pyimport logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]


def env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def full_table_name(project: str, dataset: str, table: str) -> str:
    return f"{project}.{dataset}.{table}"


def build_feature_union_query(project: str, dataset: str, table: str, features: List[str], time_filter: str = None) -> str:
    source = full_table_name(project, dataset, table)
    parts = []
    for feature in features:
        condition = f"{feature} IS NOT NULL"
        if time_filter:
            condition = f"{condition} AND {time_filter}"
        parts.append(
            f"SELECT '{feature}' AS feature_name, {feature} AS value FROM `{source}` WHERE {condition}"
        )
    return "\nUNION ALL\n".join(parts)


def get_existing_time_field(
    client: bigquery.Client,
    project: str,
    dataset: str,
    table: str,
    requested_time_field: str | None = None,
) -> str | None:
    candidate_fields = []
    if requested_time_field:
        candidate_fields.append(requested_time_field)
    for field in ["timestamp", "ingest_timestamp"]:
        if field not in candidate_fields:
            candidate_fields.append(field)

    quoted_fields = ", ".join(f"'{field}'" for field in candidate_fields)
    schema_query = f"""
        SELECT column_name
        FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table}'
        AND column_name IN ({quoted_fields})
    """

    rows = client.query(schema_query).result()
    existing_fields = {row.column_name for row in rows}
    for field in candidate_fields:
        if field in existing_fields:
            return field
    return None


def load_feature_values(client: bigquery.Client, query: str) -> pd.DataFrame:
    job = client.query(query)
    return job.result().to_dataframe()


def compute_bin_edges(values: pd.Series, bins: int) -> np.ndarray:
    if values.dropna().empty:
        raise ValueError("No baseline values available for this feature.")
    edges = np.quantile(values.dropna().values, np.linspace(0, 1, bins + 1))
    edges[0] = edges[0] - 1e-9
    edges[-1] = edges[-1] + 1e-9
    return edges


def compute_psi_counts(training_values: pd.Series, live_values: pd.Series, bins: int) -> Tuple[float, List[int], List[int]]:
    edges = compute_bin_edges(training_values, bins)
    training_counts, _ = np.histogram(training_values.dropna().values, bins=edges)
    live_counts, _ = np.histogram(live_values.dropna().values, bins=edges)

    training_total = int(training_counts.sum())
    live_total = int(live_counts.sum())
    if training_total == 0 or live_total == 0:
        return 0.0, training_counts.tolist(), live_counts.tolist()

    training_pct = (training_counts + 1) / (training_total + bins)
    live_pct = (live_counts + 1) / (live_total + bins)
    psi = float(np.sum((live_pct - training_pct) * np.log(live_pct / training_pct)))
    return psi, training_counts.tolist(), live_counts.tolist()


def compute_ks(training_values: pd.Series, live_values: pd.Series) -> float:
    t = np.sort(training_values.dropna().values)
    l = np.sort(live_values.dropna().values)
    if len(t) == 0 or len(l) == 0:
        return 0.0

    all_values = np.unique(np.concatenate([t, l]))
    cdf_t = np.searchsorted(t, all_values, side="right") / len(t)
    cdf_l = np.searchsorted(l, all_values, side="right") / len(l)
    return float(np.max(np.abs(cdf_t - cdf_l)))


def derive_severity(psi: float, ks: float) -> Tuple[str, float]:
    scaled_psi = min(max(psi, 0.0), 1.0)
    scaled_ks = min(max(ks, 0.0), 1.0)
    average_score = (scaled_psi + scaled_ks) / 2.0
    if average_score < 0.1:
        return "low", average_score
    if average_score < 0.2:
        return "medium", average_score
    return "high", average_score


def create_drift_table_if_missing(client: bigquery.Client, dataset_id: str, table_id: str):
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = env("BQ_LOCATION", "US")
        client.create_dataset(dataset, timeout=30)

    full_table_id = full_table_name(client.project, dataset_id, table_id)
    try:
        client.get_table(full_table_id)
        return
    except Exception:
        schema = [
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("feature_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("psi", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("ks", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("avg_score", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("severity", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("training_total", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("live_total", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("training_buckets", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("live_buckets", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        table = bigquery.Table(full_table_id, schema=schema)
        client.create_table(table, exists_ok=True)


def clear_existing_day(client: bigquery.Client, dataset: str, table: str):
    full_name = full_table_name(client.project, dataset, table)
    client.query(f"DELETE FROM `{full_name}` WHERE date = CURRENT_DATE()").result()


def build_records(feature: str, training_values: pd.Series, live_values: pd.Series, bins: int) -> Dict:
    psi, training_counts, live_counts = compute_psi_counts(training_values, live_values, bins)
    ks = compute_ks(training_values, live_values)
    severity, avg_score = derive_severity(psi, ks)
    return {
        "date": datetime.utcnow().date(),
        "feature_name": feature,
        "psi": psi,
        "ks": ks,
        "avg_score": avg_score,
        "severity": severity,
        "training_total": int(training_values.dropna().shape[0]),
        "live_total": int(live_values.dropna().shape[0]),
        "training_buckets": ",".join(str(x) for x in training_counts),
        "live_buckets": ",".join(str(x) for x in live_counts),
        "created_at": datetime.utcnow(),
    }


def main():
    project_id = env("BQ_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    dataset = env("BQ_DATASET", "ml_observability")
    baseline_table = env("BQ_BASELINE_TABLE", "training_baseline")
    logs_table = env("BQ_LOG_TABLE", "prediction_logs")
    drift_table = env("BQ_DRIFT_TABLE", "drift_summary_daily")
    time_field = env("BQ_TIME_FIELD", "timestamp")
    bins = int(env("DRIFT_BINS", "10"))
    window_days = int(env("DRIFT_WINDOW_DAYS", "1"))

    if not project_id:
        raise ValueError("BQ_PROJECT or GOOGLE_CLOUD_PROJECT must be set.")

    client = bigquery.Client(project=project_id)
    training_query = build_feature_union_query(project_id, dataset, baseline_table, FEATURES)
    resolved_time_field = get_existing_time_field(
        client,
        project_id,
        dataset,
        logs_table,
        requested_time_field=time_field or None,
    )

    if resolved_time_field:
        logging.info(f"Using time field '{resolved_time_field}' for live data filtering.")
        time_filter = f"{resolved_time_field} >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {window_days} DAY)"
    else:
        logging.warning(
            "No valid time field found on live table; querying all rows without time filtering."
        )
        time_filter = None

    live_query = build_feature_union_query(
        project_id,
        dataset,
        logs_table,
        FEATURES,
        time_filter=time_filter,
    )

    training_df = load_feature_values(client, training_query)
    live_df = load_feature_values(client, live_query)

    records = []
    for feature in FEATURES:
        training_values = training_df.loc[training_df["feature_name"] == feature, "value"]
        live_values = live_df.loc[live_df["feature_name"] == feature, "value"]
        records.append(build_records(feature, training_values, live_values, bins))

    result_df = pd.DataFrame(records)
    create_drift_table_if_missing(client, dataset, drift_table)
    clear_existing_day(client, dataset, drift_table)

    table_id = full_table_name(project_id, dataset, drift_table)
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
    job = client.load_table_from_dataframe(result_df, table_id, job_config=job_config)
    job.result()

    print(f"✅ Drift summary written to {table_id} for date {datetime.utcnow().date()}")


if __name__ == "__main__":
    main()
