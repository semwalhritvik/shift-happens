"""
drift_detector.py — Drift detection script for ShiftHappens.

Runs on Cloud Run, triggered by Cloud Scheduler when new prediction
data arrives in BigQuery.

Calculates two drift scores per feature column:
    1. PSI (Population Stability Index) — measures distribution shift
    2. KS  (Kolmogorov-Smirnov Test)   — measures max distribution gap

Aggregates PSI and KS into a single drift label per column:
    LOW    — average score < 0.10
    MEDIUM — average score 0.10 to 0.20
    HIGH   — average score > 0.20

Results written to BigQuery drift_scores table.
Streamlit consultancy dashboard reads this table to show drift status.

Drift Thresholds (industry standard):
    PSI < 0.10  → LOW  | 0.10-0.20 → MEDIUM | > 0.20 → HIGH
    KS  < 0.10  → LOW  | 0.10-0.20 → MEDIUM | > 0.20 → HIGH

Environment Variables Required:
    GCP_PROJECT  — GCP project ID
    BQ_DATASET   — BigQuery dataset name
    GOOGLE_APPLICATION_CREDENTIALS — path to service account key JSON
"""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────────────────────────────────────
# GCP Configuration
# ─────────────────────────────────────────────────────────────
GCP_PROJECT = os.environ.get(
    "GCP_PROJECT",
    os.environ.get("GOOGLE_CLOUD_PROJECT", "shifthappens0123"),
)
BQ_DATASET = os.environ.get("BQ_DATASET", "ml_observability")

BASELINE_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.training_baseline"
PREDICTIONS_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.prediction_logs"
DRIFT_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.drift_scores"

# Drift thresholds
PSI_LOW = 0.10
PSI_HIGH = 0.20
KS_LOW = 0.10
KS_HIGH = 0.20

# Features to monitor for drift
MONITORED_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]


# ─────────────────────────────────────────────────────────────
# Data Loading from BigQuery
# ─────────────────────────────────────────────────────────────

def load_baseline(client: bigquery.Client) -> pd.DataFrame:
    query = f"SELECT * FROM `{BASELINE_TABLE}`"
    logging.info(f"Loading baseline from: {BASELINE_TABLE}")
    df = client.query(query).result().to_dataframe()
    logging.info(f"Baseline loaded: {len(df)} rows")
    return df


def load_predictions(client: bigquery.Client) -> pd.DataFrame:
    query = f"""
        SELECT * FROM `{PREDICTIONS_TABLE}`
        ORDER BY timestamp DESC
        LIMIT 1000
    """
    logging.info(f"Loading predictions from: {PREDICTIONS_TABLE}")
    df = client.query(query).result().to_dataframe()
    logging.info(f"Predictions loaded: {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────
# PSI Calculation
# ─────────────────────────────────────────────────────────────

def calculate_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    breakpoints = np.linspace(
        min(baseline.min(), current.min()),
        max(baseline.max(), current.max()),
        bins + 1,
    )

    baseline_pct = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
    current_pct = np.histogram(current, bins=breakpoints)[0] / len(current)

    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return round(float(psi), 4)


# ─────────────────────────────────────────────────────────────
# KS Test Calculation
# ─────────────────────────────────────────────────────────────

def calculate_ks(baseline: np.ndarray, current: np.ndarray) -> float:
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    ks_stat, _ = stats.ks_2samp(baseline, current)
    return round(float(ks_stat), 4)


# ─────────────────────────────────────────────────────────────
# Drift Label
# ─────────────────────────────────────────────────────────────

def get_drift_label(avg_score: float) -> str:
    if avg_score < PSI_LOW:
        return "LOW"
    elif avg_score < PSI_HIGH:
        return "MEDIUM"
    return "HIGH"


# ─────────────────────────────────────────────────────────────
# Main Drift Detection
# ─────────────────────────────────────────────────────────────

def detect_drift(df_baseline: pd.DataFrame, df_current: pd.DataFrame) -> pd.DataFrame:
    results = []
    timestamp = datetime.utcnow().isoformat()

    for feature in MONITORED_FEATURES:
        if feature not in df_baseline.columns or feature not in df_current.columns:
            logging.warning(f"Feature '{feature}' not found in data. Skipping.")
            continue

        baseline_vals = df_baseline[feature].astype(float).values
        current_vals = df_current[feature].astype(float).values

        psi_score = calculate_psi(baseline_vals, current_vals)
        ks_score = calculate_ks(baseline_vals, current_vals)
        avg_score = round((psi_score + ks_score) / 2, 4)
        drift_label = get_drift_label(avg_score)

        results.append({
            "feature": feature,
            "psi_score": psi_score,
            "ks_score": ks_score,
            "avg_score": avg_score,
            "drift_label": drift_label,
            "timestamp": timestamp,
        })

        logging.info(
            f"{feature}: PSI={psi_score:.4f} | KS={ks_score:.4f} | "
            f"Avg={avg_score:.4f} | Drift={drift_label}"
        )

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# Write Results to BigQuery
# ─────────────────────────────────────────────────────────────

def write_drift_scores(client: bigquery.Client, df_results: pd.DataFrame):
    table_ref = DRIFT_TABLE
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    job = client.load_table_from_dataframe(df_results, table_ref, job_config=job_config)
    job.result()
    logging.info(f"Drift scores written to: {table_ref}")


# ─────────────────────────────────────────────────────────────
# Entry Point — called by Cloud Scheduler
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("ShiftHappens — Drift Detection Job")
    logging.info("=" * 60)

    client = bigquery.Client(project=GCP_PROJECT)
    df_baseline = load_baseline(client)
    df_current = load_predictions(client)

    df_results = detect_drift(df_baseline, df_current)

    logging.info("─" * 40)
    logging.info("Drift Detection Summary:")
    for _, row in df_results.iterrows():
        logging.info(
            f"  {row['feature']}: {row['drift_label']} (avg={row['avg_score']:.4f})"
        )

    high_drift = df_results[df_results["drift_label"] == "HIGH"]
    if len(high_drift) > 0:
        logging.warning(f"HIGH drift detected in {len(high_drift)} features!")
        logging.warning("Consultancy dashboard will show RED status.")
    else:
        logging.info("No HIGH drift detected. System healthy.")

    write_drift_scores(client, df_results)

    logging.info("=" * 60)
    logging.info("Drift detection complete.")
    logging.info("=" * 60)
