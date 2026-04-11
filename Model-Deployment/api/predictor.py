"""
predictor.py — Loads the final debiased model and runs inference.

The model was trained on 304 features from application_train_merged.pkl.
At inference time, missing features are filled with training data medians
to ensure feature shape consistency with the trained model.
"""

import os
import pickle
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────────────────────────────────────
# Absolute paths — works regardless of where API is run from
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "/Users/kishlayasethi/Desktop/shift/Model-Development/models/final_model_debiased.pkl"
DATA_PATH  = "/Users/kishlayasethi/Desktop/shift/Model-Development/data/processed/application_train_merged.pkl"

SENSITIVE_FEATURE = "CODE_GENDER"
DROP_COLS         = ["SK_ID_CURR", "TARGET"]

MODEL_METRICS = {
    "roc_auc":    0.7779,
    "f1_score":   0.2897,
    "accuracy":   0.7335,
    "model_name": "LightGBM_debiased"
}

def load_model():
    """Loads the final debiased LightGBM model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
    return model


def load_feature_template() -> pd.Series:
    """
    Loads training data and computes column medians as the feature template.
    Used at inference time to fill missing features so the model always
    receives exactly 304 features matching the training shape.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at: {DATA_PATH}")

    df = pd.read_pickle(DATA_PATH)

    # Drop identifier, target and sensitive columns
    for col in DROP_COLS + [SENSITIVE_FEATURE]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encode categorical columns to match training preprocessing
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    medians = df.median(numeric_only=True)
    logging.info(f"Feature template loaded. Total features: {len(medians)}")
    return medians


# Load once at startup
FEATURE_TEMPLATE = load_feature_template()


def preprocess_input(data: dict) -> tuple:
    """
    Preprocesses a single prediction request into a 304-feature vector.
    Starts with all 304 training feature medians as baseline,
    then overrides with values provided in the request.
    """
    row = FEATURE_TEMPLATE.copy().to_dict()

    # Extract sensitive feature
    sensitive_val     = data.get(SENSITIVE_FEATURE, "M")
    sensitive_encoded = 0 if str(sensitive_val).upper() == "M" else 1

    # Override template with provided values
    for key, value in data.items():
        if key not in DROP_COLS and key != SENSITIVE_FEATURE and value is not None:
            row[key] = value

    df = pd.DataFrame([row])

    # Encode any remaining object columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(0)
    logging.info(f"Feature vector shape: {df.shape}")
    return df, sensitive_encoded


def get_risk_level(probability: float) -> str:
    """Converts default probability to LOW / MEDIUM / HIGH."""
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    else:
        return "HIGH"


def predict(model, data: dict) -> dict:
    """
    Runs inference on a single applicant record.
    ThresholdOptimizer requires sensitive_features at predict time.
    Probability extracted from underlying LightGBM base estimator.
    """
    X, sensitive = preprocess_input(data)

    pred = int(model.predict(X, sensitive_features=[sensitive])[0])

    try:
        base_model = model.estimator_
        proba      = float(base_model.predict_proba(X)[:, 1][0])
    except Exception:
        proba = float(pred)

    return {
        "prediction":  pred,
        "probability": round(proba, 4),
        "risk_level":  get_risk_level(proba),
        "model_name":  "LightGBM_debiased"
    }
