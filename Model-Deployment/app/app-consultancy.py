"""
Consultancy View — Model Monitoring Dashboard
===============================================
Your team monitors the model. Detailed drift analysis,
feature distributions, fairness metrics, predictions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_consultancy_view(predictions_df, training_df, drift_scores, health_status, health_color, monitored_features):

    st.markdown("## Consultancy Dashboard")
    st.markdown("Detailed model performance, drift analysis, and monitoring metrics.")
    st.markdown("---")

    # ─── Key Metrics ───────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Predictions", f"{len(predictions_df):,}")
    with col2:
        safe_count = (predictions_df["PREDICTION"] == 0).sum()
        st.metric("Predicted Safe", f"{safe_count:,}")
    with col3:
        default_count = (predictions_df["PREDICTION"] == 1).sum()
        st.metric("Predicted Default", f"{default_count:,}")
    with col4:
        default_rate = (predictions_df["PREDICTION"] == 1).mean() * 100
        st.metric("Default Rate", f"{default_rate:.1f}%")
    with col5:
        st.metric("Health", health_status)

    st.markdown("---")

    # ─── Prediction Distribution + Risk Score ──────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Prediction Distribution")
        pred_counts = predictions_df["PREDICTION"].value_counts().reset_index()
        pred_counts.columns = ["Prediction", "Count"]
        pred_counts["Label"] = pred_counts["Prediction"].map({0: "Safe (0)", 1: "Default (1)"})
        fig_pie = px.pie(
            pred_counts, values="Count", names="Label",
            color="Label",
            color_discrete_map={"Safe (0)": "#2ecc71", "Default (1)": "#e74c3c"},
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Risk Score Distribution")
        fig_hist = px.histogram(
            predictions_df, x="PREDICTION_PROBA",
            nbins=50, color_discrete_sequence=["#3498db"],
            labels={"PREDICTION_PROBA": "Prediction Probability"}
        )
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red",
                           annotation_text="Decision Boundary")
        fig_hist.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ─── Drift Analysis ───────────────────────────────────
    st.subheader("📉 Feature Drift Analysis (PSI)")
    st.caption("Population Stability Index — compares prediction data vs training baseline")

    if drift_scores:
        fig_drift = go.Figure()
        colors = ["#2ecc71" if v < 0.1 else "#f39c12" if v < 0.2 else "#e74c3c"
                  for v in drift_scores.values()]
        fig_drift.add_trace(go.Bar(
            x=list(drift_scores.keys()),
            y=list(drift_scores.values()),
            marker_color=colors,
            text=[f"{v:.4f}" for v in drift_scores.values()],
            textposition="outside"
        ))
        fig_drift.add_hline(y=0.1, line_dash="dash", line_color="orange",
                            annotation_text="Warning (0.1)")
        fig_drift.add_hline(y=0.2, line_dash="dash", line_color="red",
                            annotation_text="Drift (0.2)")
        fig_drift.update_layout(
            yaxis_title="PSI Score", xaxis_title="Feature",
            margin=dict(t=40, b=20), height=400
        )
        st.plotly_chart(fig_drift, use_container_width=True)

        drift_df = pd.DataFrame([
            {"Feature": k, "PSI": v,
             "Status": "🟢 Stable" if v < 0.1 else ("🟡 Warning" if v < 0.2 else "🔴 Drift")}
            for k, v in drift_scores.items()
        ])
        st.dataframe(drift_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Could not compute drift — training baseline not available.")

    st.markdown("---")

    # ─── Feature Distribution Comparison ───────────────────
    st.subheader("📊 Feature Distribution: Training vs Predictions")

    if training_df is not None:
        selected_feature = st.selectbox("Select feature to compare", monitored_features, index=0)

        if selected_feature in training_df.columns and selected_feature in predictions_df.columns:
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Histogram(
                x=training_df[selected_feature].dropna(),
                name="Training Baseline",
                opacity=0.6, marker_color="#3498db",
                nbinsx=50, histnorm="probability"
            ))
            fig_compare.add_trace(go.Histogram(
                x=predictions_df[selected_feature].dropna(),
                name="Current Predictions",
                opacity=0.6, marker_color="#e74c3c",
                nbinsx=50, histnorm="probability"
            ))
            fig_compare.update_layout(
                barmode="overlay",
                xaxis_title=selected_feature,
                yaxis_title="Proportion",
                margin=dict(t=20, b=20), height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.warning("Training baseline not loaded. Cannot compare distributions.")

    st.markdown("---")

    # ─── Gender Fairness ──────────────────────────────────
    st.subheader("⚖️ Fairness Analysis by Gender")

    if "CODE_GENDER" in predictions_df.columns:
        gender_stats = predictions_df.groupby("CODE_GENDER").agg(
            count=("PREDICTION", "count"),
            default_rate=("PREDICTION", "mean"),
            avg_risk=("PREDICTION_PROBA", "mean")
        ).reset_index()
        gender_stats["default_rate"] = (gender_stats["default_rate"] * 100).round(1)
        gender_stats["avg_risk"] = (gender_stats["avg_risk"] * 100).round(1)
        gender_stats.columns = ["Gender", "Count", "Default Rate (%)", "Avg Risk Score (%)"]

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.dataframe(gender_stats, use_container_width=True, hide_index=True)
        with col_g2:
            fig_gender = px.bar(
                gender_stats, x="Gender", y="Default Rate (%)",
                color="Gender", text="Default Rate (%)"
            )
            fig_gender.update_layout(margin=dict(t=20, b=20), height=300)
            st.plotly_chart(fig_gender, use_container_width=True)

    st.markdown("---")

    # ─── Recent Predictions ────────────────────────────────
    st.subheader("🔎 Recent Predictions")
    display_cols = ["SK_ID_CURR", "CODE_GENDER", "AMT_INCOME_TOTAL",
                    "AMT_CREDIT", "PREDICTION", "PREDICTION_PROBA"]
    available_cols = [c for c in display_cols if c in predictions_df.columns]
    st.dataframe(
        predictions_df[available_cols].tail(20),
        use_container_width=True,
        hide_index=True
    )
