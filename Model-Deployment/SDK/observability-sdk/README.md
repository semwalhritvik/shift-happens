# GCP Observability SDK

## Overview

This SDK provides a lightweight framework for:
- publishing prediction events asynchronously to Pub/Sub
- acting as a non-blocking interceptor around model inference
- enabling a downstream ingestion pipeline to move logs into BigQuery
- keeping production latency out of the client’s request path
- supporting an observability dashboard through event-driven logging

## Installation

```bash
cd observability-sdk
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you want to install the package locally for imports:

```bash
python -m pip install -e .
```

## Run the demo

From the `observability-sdk` folder, run:

```powershell
python examples/demo.py
```

If you use the setup script, it will already install the package editable.

Before running the demo, ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a valid GCP service account JSON file with BigQuery permissions.

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account.json"
python examples/demo.py
```

## Run the UI

From the `observability-sdk` folder, run:

```powershell
streamlit run ui.py
```

This app now supports two dashboard views:
- **Client View:** a simplified traffic-light-style health indicator for business stakeholders.
- **Consultancy View:** prediction summary, error counts, and baseline data quality metrics for engineers.

Then open the browser URL shown by Streamlit.

## Run the Prediction API

The prediction API wraps a `.pkl` model and publishes every request to Pub/Sub asynchronously.

Set the required environment variables before starting the server:

```powershell
$env:SHIFTHAPPENS_GCP_PROJECT = "your-gcp-project-id"
$env:SHIFTHAPPENS_PUBSUB_TOPIC = "your-pubsub-topic-name"
$env:SHIFTHAPPENS_MODEL_PATH = "Model-Development/models/final_model_debiased.pkl"
```

Run the server from the `observability-sdk` folder:

```powershell
python api_server.py
```

This starts the API on `http://127.0.0.1:8000`.

## API Client Demo

Send sample prediction requests to the API with:

```powershell
python examples/api_client.py --url http://127.0.0.1:8000/predict --client_id client_1 --model_version v1
```

If you have a CSV or JSON file with sample rows, pass `--data_path sample.csv` or `--data_path sample.json`.

## API Flow

1. Client data is posted to `/predict`.
2. The model makes a prediction.
3. The SDK publishes the prediction event to Pub/Sub.
4. A downstream Cloud Function ingests the Pub/Sub event into BigQuery.
5. The Streamlit dashboard reads the BigQuery drift metrics and displays health status.
## Ingestion pipeline

For the Pub/Sub architecture, deploy a Cloud Function that subscribes to the SDK topic and writes each prediction event into BigQuery.
Use `cloud_function/main.py` as the ingestion handler, and point it at `ml_observability.prediction_logs`.
Alternatively, in VS Code you can use the built-in tasks:

- `Setup environment`
- `Run demo`
- `Run UI`

## Connecting the SDK to your product

The SDK is meant to live inside the consultancy prediction service, not inside BigQuery or Pub/Sub itself.
The expected flow is:
1. The client sends new feature data to the consultancy prediction API.
2. The consultancy API loads the model and calls `sdk.predict(...)`.
3. After prediction, the API calls `sdk.log_prediction(...)` to record the event in BigQuery.
4. The Streamlit dashboard reads from BigQuery and shows the health status to both client and consultancy views.

If you want a future serverless event pipeline, the same pattern holds: the API publishes prediction logs to Pub/Sub, a Cloud Function transforms them, and BigQuery stores them for dashboard queries.

## Example usage

```python
from gcp_observability_sdk import ObservabilitySDK

sdk = ObservabilitySDK(
    project_id="your-gcp-project-id",
    dataset="ml_observability",
    model_path="model.pkl",
)

df = sdk.load_pickle_to_dataframe("data.pkl")
print(sdk.upload_dataframe(df, table_name="raw_data", write_disposition="WRITE_TRUNCATE"))

predictions, latency_ms = sdk.predict(df.head(5))
print(predictions, latency_ms)

sdk.log_prediction(
    client_id="client_1",
    model_version="v1",
    input_data=df.head(1).iloc[0].to_dict(),
    prediction=predictions[0],
    latency_ms=latency_ms,
)

print(sdk.run_data_quality_check(df))
print(sdk.summarize_predictions())
```

## Simulating live traffic

A new helper script is available to simulate prediction traffic and livestream events into BigQuery:

```powershell
python examples/stream_simulator.py \
  --project_id your-gcp-project-id \
  --dataset ml_observability \
  --model_path model.pkl \
  --data_path data.pkl \
  --client_id client_1 \
  --model_version v1 \
  --rows 50
```

This script loops through rows in `data.pkl`, calls the model, and logs each event through the SDK.

## Demo

Run the sample demo:

```bash
python examples/demo.py
```

## Notes

- Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service account JSON file with BigQuery permissions.
- The package uses BigQuery for dataset creation, data upload, prediction logging, and monitoring summary queries.

## BigQuery schema

A sample BigQuery schema file is available at `bigquery_schema.sql`.
Use it as a starting point to create your reporting and observability tables in your dataset.
