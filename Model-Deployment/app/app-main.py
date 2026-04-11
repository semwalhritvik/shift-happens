"""
ShiftHappens — MLOps Monitoring Dashboard
==========================================
Main entry point. Run with: streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import storage
from io import BytesIO
from datetime import datetime

from client_view import render_client_view
from consultancy_view import render_consultancy_view
from company_view import render_company_view

st.set_page_config(
    page_title="ShiftHappens — MLOps Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

GCS_BUCKET = "shifthappens-model-registry"
PREDICTIONS_FILE = "predictions/new_applications_predictions.csv"
TRAINING_DATA_BUCKET = "shifthappens-data"
TRAINING_DATA_FILE = "application_train_merged.pkl"

MONITORED_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]


@st.cache_data(ttl=30)
def load_predictions_from_gcs():
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(PREDICTIONS_FILE)
        data = blob.download_as_bytes()
        return pd.read_csv(BytesIO(data))
    except Exception as e:
        st.error(f"Failed to load predictions: {e}")
        return None


@st.cache_data(ttl=300)
def load_training_baseline():
    try:
        client = storage.Client()
        bucket = client.bucket(TRAINING_DATA_BUCKET)
        blob = bucket.blob(TRAINING_DATA_FILE)
        data = blob.download_as_bytes()
        return pd.read_pickle(BytesIO(data))
    except Exception as e:
        st.error(f"Failed to load training data: {e}")
        return None


def compute_drift(training_df, prediction_df, feature):
    try:
        train_vals = training_df[feature].dropna()
        pred_vals = prediction_df[feature].dropna()
        if len(train_vals) == 0 or len(pred_vals) == 0:
            return 0.0
        bins = np.histogram_bin_edges(train_vals, bins=10)
        train_hist, _ = np.histogram(train_vals, bins=bins)
        pred_hist, _ = np.histogram(pred_vals, bins=bins)
        train_pct = (train_hist + 1) / (train_hist.sum() + len(train_hist))
        pred_pct = (pred_hist + 1) / (pred_hist.sum() + len(pred_hist))
        psi = np.sum((pred_pct - train_pct) * np.log(pred_pct / train_pct))
        return round(psi, 4)
    except Exception:
        return 0.0


def get_model_health(drift_scores):
    max_drift = max(drift_scores.values()) if drift_scores else 0
    if max_drift < 0.1:
        return "HEALTHY", "green"
    elif max_drift < 0.2:
        return "WARNING", "orange"
    else:
        return "DRIFT DETECTED", "red"


def trigger_retraining():
    try:
        from google.cloud import aiplatform
        aiplatform.init(
            project="shifthappens-project",
            location="northamerica-northeast2"
        )
        job = aiplatform.PipelineJob(
            display_name="shifthappens-retrain",
            template_path="gs://shifthappens-model-registry/compiled_pipeline.json",
            pipeline_root="gs://shifthappens-model-registry/pipeline_root/",
        )
        job.submit()
        return True, f"Vertex AI retraining triggered! Job: {job.display_name}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def render_retrain_button(health_status, health_color):
    st.markdown("---")
    if health_color == "red":
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #8c1f1f, #c92727);
                        padding: 16px 24px; border-radius: 12px; margin-bottom: 12px;">
                <span style="color: #f5b8b8; font-size: 14px;">
                    ⚠️ Significant drift detected — retraining recommended
                </span>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("🔄 Trigger Retraining", type="primary", use_container_width=True):
            with st.spinner("Initiating Vertex AI Pipeline..."):
                success, msg = trigger_retraining()
                if success:
                    st.success(msg)
                    st.balloons()
                else:
                    st.error(msg)
    elif health_color == "orange":
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #8c6d1f, #c9a227);
                        padding: 16px 24px; border-radius: 12px; margin-bottom: 12px;">
                <span style="color: #f5e6b8; font-size: 14px;">
                    ⚠️ Moderate drift detected — monitor closely or retrain
                </span>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("🔄 Trigger Retraining", use_container_width=True):
            with st.spinner("Initiating Vertex AI Pipeline..."):
                success, msg = trigger_retraining()
                if success:
                    st.success(msg)
                    st.balloons()
                else:
                    st.error(msg)
    else:
        st.info("✅ Model is healthy — no retraining needed.")
        if st.button("🔄 Trigger Retraining (Manual)", use_container_width=True):
            with st.spinner("Initiating Vertex AI Pipeline..."):
                success, msg = trigger_retraining()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)


# ─── Sidebar ──────────────────────────────────────────────
st.sidebar.title("🔍 ShiftHappens")
st.sidebar.caption("Shift happens. We fix it.")
st.sidebar.divider()

view = st.sidebar.radio(
    "Select View",
    ["📋 Client View", "📊 Consultancy View", "🏢 Company View"],
    index=0
)

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")


# ─── Load Data ────────────────────────────────────────────
predictions_df = load_predictions_from_gcs()
training_df = load_training_baseline()

if predictions_df is None:
    st.error("Cannot load predictions. Check GCS connection.")
    st.stop()

drift_scores = {}
if training_df is not None:
    for feature in MONITORED_FEATURES:
        if feature in predictions_df.columns and feature in training_df.columns:
            drift_scores[feature] = compute_drift(training_df, predictions_df, feature)

health_status, health_color = get_model_health(drift_scores)


# ─── Render Selected View ─────────────────────────────────
if view == "📋 Client View":
    render_client_view(predictions_df, drift_scores, health_status, health_color, MONITORED_FEATURES)
    render_retrain_button(health_status, health_color)

elif view == "📊 Consultancy View":
    render_consultancy_view(predictions_df, training_df, drift_scores, health_status, health_color, MONITORED_FEATURES)
    render_retrain_button(health_status, health_color)

elif view == "🏢 Company View":
    render_company_view(predictions_df, drift_scores, health_status, health_color)
    render_retrain_button(health_status, health_color)
