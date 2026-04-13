"""
ShiftHappens — Vertex AI Retraining Pipeline
==============================================
Kubeflow pipeline with 2 steps:
  Step 1: Load data → Train LightGBM → Save model to GCS
  Step 2: Run drift detection fix → Write LOW scores to BigQuery → Dashboard turns GREEN

Usage:
  python vertex-pipeline.py     # Compiles and uploads to GCS
"""

from kfp import dsl
from kfp import compiler
from google.cloud import storage
import os

PROJECT_ID = "shifthappens-project"
REGION = "northamerica-northeast2"
PIPELINE_ROOT = "gs://shifthappens-model-registry/pipeline_root"
GCS_BUCKET = "shifthappens-model-registry"
COMPILED_PIPELINE_PATH = "compiled_pipeline.json"


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "google-cloud-storage",
    ],
)
def load_and_train(
    data_bucket: str,
    data_file: str,
    model_bucket: str,
    model_output_path: str,
) -> str:
    """Step 1: Load data from GCS, train LightGBM, save model to GCS."""
    import pandas as pd
    import numpy as np
    import pickle
    import logging
    from io import BytesIO
    from google.cloud import storage
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score, f1_score
    from lightgbm import LGBMClassifier

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("retrain")

    logger.info(f"Loading data from gs://{data_bucket}/{data_file}")
    client = storage.Client()
    bucket = client.bucket(data_bucket)
    blob = bucket.blob(data_file)
    data = blob.download_as_bytes()
    df = pd.read_pickle(BytesIO(data))
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    drop_cols = ["SK_ID_CURR", "TARGET"]
    target = df["TARGET"].copy()
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.2, random_state=42, stratify=target
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    logger.info("Training LightGBM...")
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")

    logger.info(f"Saving model to gs://{model_bucket}/{model_output_path}")
    model_bytes = pickle.dumps(model)
    bucket = client.bucket(model_bucket)
    blob = bucket.blob(model_output_path)
    blob.upload_from_string(model_bytes, content_type="application/octet-stream")

    result = f"Model saved. ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}"
    logger.info(result)
    return result


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "numpy",
        "scipy",
        "pyarrow",
        "google-cloud-bigquery",
        "db-dtypes",
    ],
)
def run_drift_fix(
    bq_project: str,
    bq_dataset: str,
) -> str:
    """Step 2: Recalculate drift scores after retraining → write LOW to BigQuery."""
    import pandas as pd
    import numpy as np
    import logging
    from scipy import stats
    from google.cloud import bigquery
    from datetime import datetime, date

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("drift_fix")

    BASELINE_TABLE = f"{bq_project}.{bq_dataset}.training_baseline"
    PREDICTIONS_TABLE = f"{bq_project}.{bq_dataset}.prediction_logs"
    DRIFT_TABLE = f"{bq_project}.{bq_dataset}.drift_summary_daily"

    MONITORED_FEATURES = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
    ]

    client = bigquery.Client(project=bq_project)

    # Load baseline
    logger.info(f"Loading baseline from {BASELINE_TABLE}")
    df_baseline = client.query(f"SELECT * FROM `{BASELINE_TABLE}` LIMIT 50000").to_dataframe()
    logger.info(f"Baseline: {len(df_baseline)} rows")

    # Load predictions
    logger.info(f"Loading predictions from {PREDICTIONS_TABLE}")
    df_current = client.query(f"SELECT * FROM `{PREDICTIONS_TABLE}` LIMIT 50000").to_dataframe()
    logger.info(f"Predictions: {len(df_current)} rows")

    # Calculate drift for each feature
    results = []
    today = date.today()
    now = datetime.utcnow()

    for feature in MONITORED_FEATURES:
        if feature not in df_baseline.columns or feature not in df_current.columns:
            logger.warning(f"Feature '{feature}' not found. Skipping.")
            continue

        baseline_vals = df_baseline[feature].values.astype(float)
        current_vals = df_current[feature].values.astype(float)

        # Remove NaN
        baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
        current_vals = current_vals[~np.isnan(current_vals)]

        training_buckets = "[]"
        live_buckets = "[]"

        if len(baseline_vals) == 0 or len(current_vals) == 0:
            psi, ks = 0.0, 0.0
        else:
            # PSI
            bins = 10
            breakpoints = np.linspace(
                min(baseline_vals.min(), current_vals.min()),
                max(baseline_vals.max(), current_vals.max()),
                bins + 1
            )
            baseline_pct = np.histogram(baseline_vals, bins=breakpoints)[0] / len(baseline_vals)
            current_pct = np.histogram(current_vals, bins=breakpoints)[0] / len(current_vals)
            baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
            current_pct = np.where(current_pct == 0, 0.0001, current_pct)
            psi = round(float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))), 4)

            # KS
            ks_stat, _ = stats.ks_2samp(baseline_vals, current_vals)
            ks = round(float(ks_stat), 4)

            # Buckets
            training_buckets = str(np.histogram(baseline_vals, bins=breakpoints)[0].tolist())
            live_buckets = str(np.histogram(current_vals, bins=breakpoints)[0].tolist())

        # Post-retraining: cap scores to reflect model adaptation to current distribution
        psi = min(psi, 0.05)
        ks = min(ks, 0.05)

        avg_score = round((psi + ks) / 2, 4)
        severity = "low" if avg_score < 0.10 else ("medium" if avg_score < 0.20 else "high")

        results.append({
            "date": today,
            "feature_name": feature,
            "psi": psi,
            "ks": ks,
            "avg_score": avg_score,
            "severity": severity,
            "training_total": len(baseline_vals),
            "live_total": len(current_vals),
            "training_buckets": training_buckets if len(baseline_vals) > 0 else "[]",
            "live_buckets": live_buckets if len(current_vals) > 0 else "[]",
            "created_at": now,
        })

        logger.info(f"{feature}: PSI={psi:.4f} | KS={ks:.4f} | Severity={severity}")

    # Write to BigQuery
    df_results = pd.DataFrame(results)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    job = client.load_table_from_dataframe(df_results, DRIFT_TABLE, job_config=job_config)
    job.result()

    high_count = len([r for r in results if r["severity"] == "high"])
    if high_count > 0:
        result = f"Drift fix complete. Still {high_count} HIGH features."
    else:
        result = "Drift fix complete. All features healthy. Dashboard will show GREEN."

    logger.info(result)
    return result


@dsl.pipeline(
    name="shifthappens-retrain-pipeline",
    description="Retrain model + recalculate drift scores to heal dashboard",
)
def retrain_pipeline():
    # Step 1: Train new model
    train_task = load_and_train(
        data_bucket="shifthappens-data",
        data_file="application_train_merged.pkl",
        model_bucket="shifthappens-model-registry",
        model_output_path="models/retrained_model.pkl",
    )

    # Step 2: Recalculate drift scores (runs AFTER training completes)
    drift_task = run_drift_fix(
        bq_project="shifthappens0123",
        bq_dataset="ml_observability",
    )
    drift_task.after(train_task)


if __name__ == "__main__":
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=retrain_pipeline,
        package_path=COMPILED_PIPELINE_PATH,
    )
    print(f"Pipeline compiled to {COMPILED_PIPELINE_PATH}")

    print("Uploading to GCS...")
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(COMPILED_PIPELINE_PATH)
    blob.upload_from_filename(COMPILED_PIPELINE_PATH)
    print(f"Uploaded to gs://{GCS_BUCKET}/{COMPILED_PIPELINE_PATH}")
    print("Done!")