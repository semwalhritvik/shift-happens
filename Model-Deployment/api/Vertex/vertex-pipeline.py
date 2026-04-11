"""
ShiftHappens — Vertex AI Retraining Pipeline
==============================================
Kubeflow pipeline that:
  1. Loads data from GCS
  2. Trains a LightGBM model
  3. Saves the new model back to GCS

Usage:
  python pipeline/retrain_pipeline.py     # Compiles and uploads to GCS
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
        "fairlearn",
        "google-cloud-storage",
    ],
)
def load_and_train(
    data_bucket: str,
    data_file: str,
    model_bucket: str,
    model_output_path: str,
) -> str:
    """Load data from GCS, train LightGBM, save model to GCS."""
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

    # ── Step 1: Load data from GCS ────────────────────────
    logger.info(f"Loading data from gs://{data_bucket}/{data_file}")
    client = storage.Client()
    bucket = client.bucket(data_bucket)
    blob = bucket.blob(data_file)
    data = blob.download_as_bytes()
    df = pd.read_pickle(BytesIO(data))
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Step 2: Preprocess ────────────────────────────────
    drop_cols = ["SK_ID_CURR", "TARGET"]
    target = df["TARGET"].copy()
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.2, random_state=42, stratify=target
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ── Step 3: Train LightGBM ────────────────────────────
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

    # ── Step 4: Evaluate ──────────────────────────────────
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # ── Step 5: Save model to GCS ─────────────────────────
    logger.info(f"Saving model to gs://{model_bucket}/{model_output_path}")
    model_bytes = pickle.dumps(model)
    bucket = client.bucket(model_bucket)
    blob = bucket.blob(model_output_path)
    blob.upload_from_string(model_bytes, content_type="application/octet-stream")

    result = f"Model saved. ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}"
    logger.info(result)
    return result


@dsl.pipeline(
    name="shifthappens-retrain-pipeline",
    description="Retrain LightGBM model when drift is detected",
)
def retrain_pipeline():
    load_and_train(
        data_bucket="shifthappens-data",
        data_file="application_train_merged.pkl",
        model_bucket="shifthappens-model-registry",
        model_output_path="models/retrained_model.pkl",
    )


if __name__ == "__main__":
    # Compile the pipeline to JSON
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=retrain_pipeline,
        package_path=COMPILED_PIPELINE_PATH,
    )
    print(f"Pipeline compiled to {COMPILED_PIPELINE_PATH}")

    # Upload to GCS
    print("Uploading to GCS...")
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(COMPILED_PIPELINE_PATH)
    blob.upload_from_filename(COMPILED_PIPELINE_PATH)
    print(f"Uploaded to gs://{GCS_BUCKET}/{COMPILED_PIPELINE_PATH}")
    print("Done! You can now trigger this pipeline from Streamlit.")
