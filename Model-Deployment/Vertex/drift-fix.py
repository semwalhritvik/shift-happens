"""
drift_detector_fix.py — Post-retraining drift resolution script.

After Vertex AI retrains the model, this script:
  1. Loads the training baseline from BigQuery
  2. Loads the latest prediction logs from BigQuery
  3. Recalculates PSI and KS drift scores
  4. Writes updated (low) scores to drift_summary_daily
  5. Dashboard reads new scores → turns GREEN

Run:
    python drift_detector_fix.py
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from google.cloud import bigquery
from datetime import datetime, date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

GCP_PROJECT = "shifthappens0123"
BQ_DATASET = "ml_observability"

BASELINE_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.training_baseline"
PREDICTIONS_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.prediction_logs"
DRIFT_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.drift_summary_daily"

MONITORED_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]


def load_baseline(client):
    query = f"SELECT * FROM `{BASELINE_TABLE}` LIMIT 50000"
    logging.info(f"Loading baseline from: {BASELINE_TABLE}")
    df = client.query(query).to_dataframe()
    logging.info(f"Baseline loaded: {len(df)} rows")
    return df


def load_predictions(client):
    query = f"SELECT * FROM `{PREDICTIONS_TABLE}` LIMIT 50000"
    logging.info(f"Loading predictions from: {PREDICTIONS_TABLE}")
    df = client.query(query).to_dataframe()
    logging.info(f"Predictions loaded: {len(df)} rows")
    return df


def calculate_psi(baseline, current, bins=10):
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    if len(baseline) == 0 or len(current) == 0:
        return 0.0
    breakpoints = np.linspace(
        min(baseline.min(), current.min()),
        max(baseline.max(), current.max()),
        bins + 1
    )
    baseline_pct = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
    current_pct = np.histogram(current, bins=breakpoints)[0] / len(current)
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return round(float(psi), 4)


def calculate_ks(baseline, current):
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    if len(baseline) == 0 or len(current) == 0:
        return 0.0
    ks_stat, _ = stats.ks_2samp(baseline, current)
    return round(float(ks_stat), 4)


def get_severity(avg_score):
    if avg_score < 0.10:
        return "low"
    elif avg_score < 0.20:
        return "medium"
    else:
        return "high"


def detect_drift(df_baseline, df_current):
    results = []
    today = date.today()
    now = datetime.utcnow()

    for feature in MONITORED_FEATURES:
        if feature not in df_baseline.columns or feature not in df_current.columns:
            logging.warning(f"Feature '{feature}' not found. Skipping.")
            continue

        baseline_vals = df_baseline[feature].values.astype(float)
        current_vals = df_current[feature].values.astype(float)

        psi = calculate_psi(baseline_vals, current_vals)
        ks = calculate_ks(baseline_vals, current_vals)

        bins = 10
        baseline_clean = baseline_vals[~np.isnan(baseline_vals)]
        current_clean = current_vals[~np.isnan(current_vals)]

        if len(baseline_clean) > 0 and len(current_clean) > 0:
            breakpoints = np.linspace(
                min(baseline_clean.min(), current_clean.min()),
                max(baseline_clean.max(), current_clean.max()),
                bins + 1
            )
            training_buckets = str(np.histogram(baseline_clean, bins=breakpoints)[0].tolist())
            live_buckets = str(np.histogram(current_clean, bins=breakpoints)[0].tolist())
        else:
            training_buckets = "[]"
            live_buckets = "[]"

        avg_score = round((psi + ks) / 2, 4)
        severity = get_severity(avg_score)

        results.append({
            "date": today,
            "feature_name": feature,
            "psi": psi,
            "ks": ks,
            "avg_score": avg_score,
            "severity": severity,
            "training_total": len(baseline_clean),
            "live_total": len(current_clean),
            "training_buckets": training_buckets,
            "live_buckets": live_buckets,
            "created_at": now,
        })

        logging.info(
            f"{feature}: PSI={psi:.4f} | KS={ks:.4f} | "
            f"Avg={avg_score:.4f} | Severity={severity}"
        )

    return pd.DataFrame(results)


def write_drift_scores(client, df_results):
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    job = client.load_table_from_dataframe(
        df_results, DRIFT_TABLE, job_config=job_config
    )
    job.result()
    logging.info(f"Drift scores written to: {DRIFT_TABLE}")


if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("ShiftHappens — Drift Resolution (Post-Retraining)")
    logging.info("=" * 60)

    client = bigquery.Client(project=GCP_PROJECT)

    df_baseline = load_baseline(client)
    df_current = load_predictions(client)

    df_results = detect_drift(df_baseline, df_current)

    logging.info("-" * 40)
    logging.info("Drift Resolution Summary:")
    for _, row in df_results.iterrows():
        logging.info(f"  {row['feature_name']}: {row['severity']} (PSI={row['psi']:.4f})")

    high_drift = df_results[df_results["severity"] == "high"]
    if len(high_drift) > 0:
        logging.warning(f"Still HIGH drift in {len(high_drift)} features.")
    else:
        logging.info("All features healthy. Dashboard will show GREEN.")

    write_drift_scores(client, df_results)

    logging.info("=" * 60)
    logging.info("Drift resolution complete.")
    logging.info("=" * 60)
