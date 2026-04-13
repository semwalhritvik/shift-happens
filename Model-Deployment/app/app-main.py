"""
ShiftHappens — MLOps Monitoring Dashboard
==========================================
Run with: streamlit run app-main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
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

BQ_PROJECT = "shifthappens0123"
BQ_DATASET = "ml_observability"
PREDICTIONS_TABLE = "prediction_logs"
TRAINING_TABLE = "training_baseline"

MONITORED_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]


@st.cache_data(ttl=30)
def load_predictions():
    try:
        client = bigquery.Client(project=BQ_PROJECT)
        query = f"""
            SELECT *
            FROM `{BQ_PROJECT}.{BQ_DATASET}.{PREDICTIONS_TABLE}`
            LIMIT 50000
        """
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Failed to load predictions: {e}")
        return None


@st.cache_data(ttl=300)
def load_training_baseline():
    try:
        client = bigquery.Client(project=BQ_PROJECT)
        query = f"""
            SELECT *
            FROM `{BQ_PROJECT}.{BQ_DATASET}.{TRAINING_TABLE}`
        """
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Failed to load training baseline: {e}")
        return None


@st.cache_data(ttl=30)
def load_drift_scores():
    try:
        client = bigquery.Client(project=BQ_PROJECT)
        query = f"""
            SELECT feature_name, psi, severity
            FROM (
                SELECT feature_name, psi, severity,
                       ROW_NUMBER() OVER (PARTITION BY feature_name ORDER BY created_at DESC) AS rn
                FROM `{BQ_PROJECT}.{BQ_DATASET}.drift_summary_daily`
            )
            WHERE rn = 1
        """
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        return None, str(e)


def get_model_health(drift_scores):
    if not drift_scores:
        return "HEALTHY", "green"
    max_drift = max(drift_scores.values())
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
st.sidebar.caption("Detect the shift. Heal the model.")
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
predictions_df = load_predictions()
training_df = load_training_baseline()

if predictions_df is None or len(predictions_df) == 0:
    st.warning("No prediction data in BigQuery yet. Run simulate_live_traffic.py to start streaming data.")
    st.stop()

# ─── Load Drift Scores ───────────────────────────────────
drift_scores = {}
drift_result = load_drift_scores()

if isinstance(drift_result, tuple):
    st.sidebar.error(f"Drift query error: {drift_result[1]}")
    drift_scores = {f: 0.0 for f in MONITORED_FEATURES}
elif drift_result is not None and len(drift_result) > 0:
    st.sidebar.write(f"Drift rows loaded: {len(drift_result)}")
    st.sidebar.dataframe(drift_result, hide_index=True)
    drift_scores = dict(zip(drift_result["feature_name"], drift_result["psi"]))
else:
    st.sidebar.write("No drift data found")
    drift_scores = {f: 0.0 for f in MONITORED_FEATURES}

st.sidebar.write("Final drift scores:", drift_scores)

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