"""
Client View — Data Health Dashboard
"""

import streamlit as st
import pandas as pd


def render_client_view(predictions_df, drift_scores, health_status, health_color, monitored_features):

    st.markdown("## Client Dashboard")
    st.markdown("Monitor the health and quality of your data in real-time.")
    st.markdown("---")

    if health_color == "green":
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #0d5c2e, #1a8c4e);
                        padding: 50px; border-radius: 20px; text-align: center; margin: 10px 0;">
                <h1 style="color: white; font-size: 60px; margin: 0;">✅</h1>
                <h2 style="color: white; font-size: 36px; margin: 10px 0;">DATA HEALTHY</h2>
                <p style="color: #b8e6c8; font-size: 16px;">All data streams operational. No drift detected.</p>
            </div>
            """, unsafe_allow_html=True
        )
    elif health_color == "orange":
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #8c6d1f, #c9a227);
                        padding: 50px; border-radius: 20px; text-align: center; margin: 10px 0;">
                <h1 style="color: white; font-size: 60px; margin: 0;">⚠️</h1>
                <h2 style="color: white; font-size: 36px; margin: 10px 0;">DATA WARNING</h2>
                <p style="color: #f5e6b8; font-size: 16px;">Moderate data drift detected. Monitoring closely.</p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #8c1f1f, #c92727);
                        padding: 50px; border-radius: 20px; text-align: center; margin: 10px 0;">
                <h1 style="color: white; font-size: 60px; margin: 0;">🚨</h1>
                <h2 style="color: white; font-size: 36px; margin: 10px 0;">DATA DRIFT DETECTED</h2>
                <p style="color: #f5b8b8; font-size: 16px;">Significant data drift detected. Immediate attention required.</p>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records Processed", f"{len(predictions_df):,}")
    with col2:
        st.metric("Features Tracked", f"{len(monitored_features)}")
    with col3:
        drifted_count = sum(1 for v in drift_scores.values() if v >= 0.1)
        st.metric("Features Drifted", f"{drifted_count}/{len(drift_scores)}")
    with col4:
        st.metric("Data Status", health_status)

    st.markdown("---")

    st.subheader("📈 Data Quality Summary")
    if drift_scores:
        for feature, psi in drift_scores.items():
            col_a, col_b, col_c = st.columns([3, 5, 2])
            with col_a:
                st.markdown(f"**{feature}**")
            with col_b:
                st.progress(min(psi / 0.3, 1.0), text=f"PSI: {psi:.4f}")
            with col_c:
                if psi < 0.1:
                    st.markdown("🟢 Stable")
                elif psi < 0.2:
                    st.markdown("🟡 Warning")
                else:
                    st.markdown("🔴 Drift")

    st.markdown("---")

    st.subheader("🔎 Recent Data Sample")
    display_cols = ["SK_ID_CURR", "CODE_GENDER", "AMT_INCOME_TOTAL",
                    "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH"]
    available_cols = [c for c in display_cols if c in predictions_df.columns]
    st.dataframe(
        predictions_df[available_cols].head(10),
        use_container_width=True,
        hide_index=True
    )