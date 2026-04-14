import streamlit as st
import pandas as pd
from gcp_observability_sdk import ObservabilitySDK

st.set_page_config(page_title="ShiftHappens SDK Dashboard", layout="wide")

PAGE_TITLE = "ShiftHappens Observability SDK"
PAGE_SUBTITLE = (
    "Use this interface to load baseline data, run predictions, log observability events, "
    "and review client vs consultancy views."
)

st.title(PAGE_TITLE)
st.markdown(PAGE_SUBTITLE)

if "sdk" not in st.session_state:
    st.session_state.sdk = None
    st.session_state.df = None
    st.session_state.predictions = None
    st.session_state.latency_ms = None
    st.session_state.dq_report = None
    st.session_state.summary = None
    st.session_state.last_refresh = "Never"

project_id = st.sidebar.text_input("GCP Project ID", value="your-gcp-project-id")
dataset = st.sidebar.text_input("BigQuery Dataset", value="ml_observability")
model_path = st.sidebar.text_input("Model path", value="model.pkl")
data_path = st.sidebar.text_input("Data path", value="data.pkl")
client_id = st.sidebar.text_input("Client ID", value="client_1")
model_version = st.sidebar.text_input("Model version", value="v1")

st.sidebar.markdown("---")
st.sidebar.subheader("SDK Actions")

if st.sidebar.button("Initialize SDK"):
    try:
        st.session_state.sdk = ObservabilitySDK(
            project_id=project_id,
            dataset=dataset,
            model_path=model_path,
        )
        st.sidebar.success("SDK initialized successfully.")
    except Exception as exc:
        st.sidebar.error(f"Failed to initialize SDK: {exc}")

sdk = st.session_state.sdk

if sdk is not None:
    def refresh_summary():
        try:
            summary = sdk.summarize_predictions()
            st.session_state.summary = summary
            st.session_state.last_refresh = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            return summary
        except Exception as exc:
            st.sidebar.error(f"Failed to load summary: {exc}")
            return pd.DataFrame()

    if st.sidebar.button("Load baseline pickle"):
        try:
            df = sdk.load_pickle_to_dataframe(data_path)
            st.session_state.df = df
            st.sidebar.success("Baseline data loaded.")
        except Exception as exc:
            st.sidebar.error(f"Failed to load pickle: {exc}")

    if st.sidebar.button("Upload baseline to BigQuery"):
        if st.session_state.df is None:
            st.sidebar.error("Load data first.")
        else:
            try:
                result = sdk.upload_dataframe(
                    st.session_state.df,
                    table_name="raw_data",
                    write_disposition="WRITE_TRUNCATE",
                )
                st.sidebar.success(result)
            except Exception as exc:
                st.sidebar.error(f"Failed to upload baseline: {exc}")

    if st.sidebar.button("Run sample predictions"):
        if st.session_state.df is None:
            st.sidebar.error("Load data first.")
        else:
            try:
                sample = st.session_state.df.head(5)
                predictions, latency_ms = sdk.predict(sample)
                st.session_state.predictions = predictions
                st.session_state.latency_ms = latency_ms
                st.sidebar.success("Sample predictions completed.")
            except Exception as exc:
                st.sidebar.error(f"Failed to run predictions: {exc}")

    if st.sidebar.button("Log sample prediction events"):
        if st.session_state.df is None or st.session_state.predictions is None:
            st.sidebar.error("Run sample predictions first.")
        else:
            try:
                sample = st.session_state.df.head(5)
                for i in range(min(len(sample), len(st.session_state.predictions))):
                    sdk.log_prediction(
                        client_id=client_id,
                        model_version=model_version,
                        input_data=sample.iloc[i].to_dict(),
                        prediction=st.session_state.predictions[i],
                        latency_ms=st.session_state.latency_ms,
                    )
                st.sidebar.success("Logged sample predictions.")
                refresh_summary()
            except Exception as exc:
                st.sidebar.error(f"Failed to log prediction events: {exc}")

    if st.sidebar.button("Run data quality report"):
        if st.session_state.df is None:
            st.sidebar.error("Load data first.")
        else:
            try:
                st.session_state.dq_report = sdk.run_data_quality_check(st.session_state.df)
                st.sidebar.success("Data quality report generated.")
            except Exception as exc:
                st.sidebar.error(f"Failed to run data quality: {exc}")

    if st.sidebar.button("Refresh observability summary"):
        refresh_summary()

    if st.sidebar.button("Clear current session"):
        st.session_state.df = None
        st.session_state.predictions = None
        st.session_state.latency_ms = None
        st.session_state.dq_report = None
        st.session_state.summary = None
        st.session_state.last_refresh = "Never"
        st.sidebar.success("Session state cleared.")

    def get_health_status(summary: pd.DataFrame, dq_report: dict | None):
        if summary is None or summary.empty:
            return "Unknown", "gray", "No prediction logs yet."

        total_predictions = int(summary["total_predictions"].sum())
        error_count = int(summary["error_count"].sum()) if "error_count" in summary else 0
        if error_count > 0:
            return "Critical", "red", f"{error_count} logged errors across {total_predictions} predictions."

        if dq_report is not None:
            missing_values = sum(dq_report.get("missing_values", {}).values())
            duplicate_rows = dq_report.get("duplicate_rows", 0)
            if missing_values > 0 or duplicate_rows > 0:
                return (
                    "Warning",
                    "yellow",
                    f"{missing_values} missing values, {duplicate_rows} duplicates detected in baseline data.",
                )

        return "Healthy", "green", f"{total_predictions} predictions logged."

    summary = st.session_state.summary if st.session_state.summary is not None else refresh_summary()
    dq_report = st.session_state.dq_report
    health_label, health_color, health_message = get_health_status(summary, dq_report)

    tabs = st.tabs(["Client View", "Consultancy View", "SDK Console"])

    with tabs[0]:
        st.subheader("Client View")
        st.markdown(
            "This view is designed for business stakeholders: simplified health, key status, and trust signals."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall status", health_label, delta=None)
        if summary is not None and not summary.empty:
            c2.metric("Total predictions", int(summary["total_predictions"].sum()), delta=None)
            c3.metric("Average latency (ms)", round(float(summary["avg_latency"].mean()), 2), delta=None)
        else:
            c2.metric("Total predictions", 0, delta=None)
            c3.metric("Average latency (ms)", "N/A", delta=None)

        st.info(health_message)
        st.markdown("#### Notes")
        st.write(
            "- Green means the model is healthy and the log pipeline is successfully receiving events."
            "\n- Yellow means data quality issues were detected in the baseline dataset."
            "\n- Red means prediction errors were observed and the consultancy team should investigate."
        )

    with tabs[1]:
        st.subheader("Consultancy View")
        st.markdown(
            "This view is for engineers and analysts: detailed logging, prediction summary, and baseline quality insights."
        )
        if summary is not None and not summary.empty:
            st.write("### Prediction summary")
            st.dataframe(summary)
        else:
            st.warning("No prediction log data found. Trigger sample logging or send live traffic first.")

        if dq_report is not None:
            st.write("### Data quality report")
            st.json(dq_report)
        else:
            st.info("Load baseline data and run a data quality check to see pipeline health diagnostics.")

        st.markdown(f"**Last refresh:** {st.session_state.last_refresh}")

    with tabs[2]:
        st.subheader("SDK Console")
        st.markdown(
            "Use the buttons in the sidebar to connect the SDK, upload baseline data, run predictions, and log monitoring events."
        )
        if st.session_state.df is not None:
            st.write("### Baseline data preview")
            st.dataframe(st.session_state.df.head())
        else:
            st.info("No baseline data loaded yet.")

        if st.session_state.predictions is not None:
            st.write("### Last sample predictions")
            st.write(st.session_state.predictions)
            st.write(f"Latency: {st.session_state.latency_ms} ms")
        else:
            st.info("Run a sample prediction to see inference details here.")

else:
    st.warning("Initialize the SDK from the sidebar to get started.")

st.markdown("---")
st.write(
    "Set `GOOGLE_APPLICATION_CREDENTIALS` in your environment so the SDK can authenticate to BigQuery."
)
