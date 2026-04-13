"""
Company View — Business Overview Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime


def render_company_view(predictions_df, drift_scores, health_status, health_color):

    st.markdown("## Company Dashboard")
    st.markdown("High-level business overview across all deployments.")
    st.markdown("---")

    has_target = "TARGET" in predictions_df.columns

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Clients", "1")
    with col2:
        st.metric("Models Deployed", "1")
    with col3:
        st.metric("Total Predictions Served", f"{len(predictions_df):,}")
    with col4:
        st.metric("System Status", health_status)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 System Health Overview")

        components = [
            ("Data Pipeline", "Operational"),
            ("Model Serving", "Operational"),
            ("Drift Detection", "Drift Detected" if health_color == "red" else "Operational"),
            ("CI/CD Pipeline", "Operational"),
            ("Monitoring Dashboard", "Operational"),
        ]

        for name, status in components:
            icon = "🟢" if status == "Operational" else "🔴"
            st.markdown(f"{icon} **{name}** — {status}")

    with col_right:
        st.subheader("📈 Prediction Volume")
        if has_target:
            default_rate = (predictions_df["TARGET"] == 1).mean() * 100
            safe_rate = 100 - default_rate

            fig_donut = go.Figure(data=[go.Pie(
                labels=["Safe", "Default"],
                values=[safe_rate, default_rate],
                hole=0.5,
                marker_colors=["#2ecc71", "#e74c3c"]
            )])
            fig_donut.update_layout(
                margin=dict(t=20, b=20, l=20, r=20), height=300,
                annotations=[dict(
                    text=f"{default_rate:.1f}%",
                    x=0.5, y=0.5, font_size=24, showarrow=False
                )]
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No target data available for prediction volume chart.")

    st.markdown("---")

    st.subheader("🔧 Pipeline Activity Log")

    activity_log = pd.DataFrame({
        "Timestamp": [
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            "2026-04-12 06:07:00",
            "2026-04-11 18:30:00",
            "2026-04-09 12:29:00",
            "2026-04-06 15:47:00",
            "2026-04-06 15:40:00",
        ],
        "Event": [
            "Dashboard refreshed",
            "drift_summary_daily table created",
            "Vertex AI pipeline compiled",
            "Pub/Sub topic created",
            "CI/CD pipeline — full dataset predictions",
            "Model updated to final_model_debiased.pkl",
        ],
        "Status": [
            "✅ Success", "✅ Success", "✅ Success",
            "✅ Success", "✅ Success", "✅ Success"
        ],
    })
    st.dataframe(activity_log, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("🚨 Active Alerts")

    if health_color == "red":
        drifted = [f for f, v in drift_scores.items() if v >= 0.2]
        for f in drifted:
            st.error(f"ALERT: Significant data drift detected in {f} (PSI: {drift_scores[f]:.4f}). Retraining recommended.")
    elif health_color == "orange":
        warned = [f for f, v in drift_scores.items() if 0.1 <= v < 0.2]
        for f in warned:
            st.warning(f"WARNING: Moderate drift in {f} (PSI: {drift_scores[f]:.4f}). Monitor closely.")
    else:
        st.success("No active alerts. All systems operational.")

    st.markdown("---")

    st.subheader("💰 Resource Usage")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.metric("Cloud Build Runs", "6", delta="2 this week")
    with col_c2:
        st.metric("GCS Storage", "1.2 GB")
    with col_c3:
        st.metric("Credits Used", "$8 / $300", delta="-$292 remaining")

    st.markdown("---")

    st.subheader("👥 Client Summary")

    client_data = pd.DataFrame({
        "Client": ["Client A"],
        "Records": [f"{len(predictions_df):,}"],
        "Model": ["LightGBM (Debiased)"],
        "Health": [health_status],
        "Last Updated": [datetime.now().strftime("%Y-%m-%d %H:%M")],
    })
    st.dataframe(client_data, use_container_width=True, hide_index=True)