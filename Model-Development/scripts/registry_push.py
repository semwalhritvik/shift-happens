"""
registry_push.py — Push validated model to GCP Model Registry (GCS).

Implements a two-gate rollback mechanism:
    Gate 1 — Absolute minimum AUC floor (MIN_AUC_THRESHOLD = 0.70)
    Gate 2 — Relative comparison against current production model AUC

Environment variables required:
    GCS_BUCKET                      — GCS bucket name
    GOOGLE_APPLICATION_CREDENTIALS  — Path to GCP service account JSON key

GCS layout:
    gs://<bucket>/production/model.pkl          — current production model
    gs://<bucket>/production/metadata.json      — current production metadata
    gs://<bucket>/archive/<timestamp>/model.pkl — archived previous versions
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime

from google.cloud import storage
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/registry_push.log"),
        logging.StreamHandler()
    ]
)

MODELS_DIR               = "models"
REPORTS_DIR              = "reports"
GCS_BUCKET               = os.environ.get("GCS_BUCKET", "shifthappens-model-registry")
PRODUCTION_MODEL_BLOB    = "production/model.pkl"
PRODUCTION_METADATA_BLOB = "production/metadata.json"

# Absolute minimum ROC-AUC required for any deployment.
# Enforced regardless of current production model state.
MIN_AUC_THRESHOLD = 0.70


def get_gcs_client():
    """Initialise and return a GCS storage client."""
    return storage.Client()


def get_current_production_auc(bucket) -> float:
    """
    Retrieves ROC-AUC of the current production model from GCS metadata.
    Returns 0.0 if no production model exists (first deployment).
    """
    blob = bucket.blob(PRODUCTION_METADATA_BLOB)
    if not blob.exists():
        logging.info("No existing production model found. First deployment.")
        return 0.0
    metadata = json.loads(blob.download_as_text())
    auc = metadata.get("roc_auc", 0.0)
    logging.info(f"Current production model AUC: {auc:.4f}")
    return auc


def archive_current_production(bucket):
    """
    Copies current production model and metadata to a timestamped
    archive folder before overwriting with the new model.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for blob_name in [PRODUCTION_MODEL_BLOB, PRODUCTION_METADATA_BLOB]:
        blob = bucket.blob(blob_name)
        if blob.exists():
            archive_path = f"archive/{timestamp}/{os.path.basename(blob_name)}"
            bucket.copy_blob(blob, bucket, archive_path)
            logging.info(f"Archived: {blob_name} → {archive_path}")


def upload_model_to_gcs(bucket, model_path: str, metadata: dict):
    """Uploads model binary and metadata JSON to GCS production location."""
    model_blob = bucket.blob(PRODUCTION_MODEL_BLOB)
    model_blob.upload_from_filename(model_path)
    logging.info(f"Model uploaded → gs://{bucket.name}/{PRODUCTION_MODEL_BLOB}")

    metadata_blob = bucket.blob(PRODUCTION_METADATA_BLOB)
    metadata_blob.upload_from_string(
        json.dumps(metadata, indent=2),
        content_type="application/json"
    )
    logging.info(f"Metadata uploaded → gs://{bucket.name}/{PRODUCTION_METADATA_BLOB}")


def push_to_registry(model_path: str, new_auc: float, model_name: str) -> bool:
    """
    Pushes model to GCS with two-gate rollback protection.

    Gate 1 — Absolute minimum AUC floor:
        Rejects any model below MIN_AUC_THRESHOLD (0.70) regardless
        of production state. Prevents a fundamentally weak model
        from being deployed even on first run.

    Gate 2 — Relative comparison against production:
        Rejects model if it does not outperform the current
        production model AUC. Prevents deployment of a degraded
        model that passes the floor but underperforms what is live.

    Args:
        model_path:  Local path to serialised model .pkl file.
        new_auc:     ROC-AUC of the new model on hold-out test set.
        model_name:  Identifier of the model being pushed.

    Returns:
        True if both gates passed and model pushed successfully.
        False if blocked by either gate.
    """
    logging.info("Connecting to GCS...")
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)

    if not bucket.exists():
        logging.error(f"GCS bucket '{GCS_BUCKET}' does not exist.")
        return False

    logging.info(f"New model AUC:          {new_auc:.4f}")
    logging.info(f"Minimum AUC threshold:  {MIN_AUC_THRESHOLD:.4f}")

    # Gate 1 — Absolute minimum AUC floor
    if new_auc < MIN_AUC_THRESHOLD:
        logging.error(
            f"REJECTED — Model AUC ({new_auc:.4f}) is below the minimum "
            f"deployment threshold ({MIN_AUC_THRESHOLD:.4f}). Push blocked."
        )
        return False

    current_auc = get_current_production_auc(bucket)
    logging.info(f"Production model AUC:   {current_auc:.4f}")

    # Gate 2 — Relative comparison against current production model
    if new_auc <= current_auc:
        logging.warning(
            f"ROLLBACK — New model AUC ({new_auc:.4f}) does not exceed "
            f"production AUC ({current_auc:.4f}). Push blocked."
        )
        logging.warning("Existing production model remains deployed.")
        return False

    # Both gates passed — archive current and push new model
    archive_current_production(bucket)

    metadata = {
        "model_name":              model_name,
        "roc_auc":                 round(new_auc, 4),
        "min_auc_threshold":       MIN_AUC_THRESHOLD,
        "previous_production_auc": round(current_auc, 4),
        "pushed_at":               datetime.now().isoformat(),
        "source_path":             model_path,
    }

    upload_model_to_gcs(bucket, model_path, metadata)
    logging.info("Registry push successful. New model is now in production.")
    return True


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess

    logging.info("=" * 60)
    logging.info("ShiftHappens — GCP Registry Push")
    logging.info("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    tuned_path = os.path.join(MODELS_DIR, "best_model_LightGBM_tuned.pkl")
    base_path  = os.path.join(MODELS_DIR, "best_model_LightGBM.pkl")
    model_path = tuned_path if os.path.exists(tuned_path) else base_path
    model_name = "LightGBM_tuned" if os.path.exists(tuned_path) else "LightGBM"

    logging.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_proba = model.predict_proba(X_test)[:, 1]
    new_auc = roc_auc_score(y_test, y_proba)
    logging.info(f"New model ROC-AUC: {new_auc:.4f}")

    success = push_to_registry(model_path, new_auc, model_name)

    if success:
        logging.info("Model deployed to production successfully.")
    else:
        logging.error("Registry push blocked. Review logs for details.")
        sys.exit(1)
