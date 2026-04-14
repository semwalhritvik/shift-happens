# Pub/Sub-to-BigQuery Cloud Function

This Cloud Function ingests prediction events published by the ShiftHappens SDK from Pub/Sub into BigQuery.

## Deployment

Set the following environment variables:

- `BQ_DATASET` (default: `ml_observability`)
- `BQ_TABLE` (default: `prediction_logs`)
- `BQ_LOCATION` (default: `US`)

Deploy the function with:

```bash
gcloud functions deploy shifthappens-pubsub-ingest \
  --region=us-central1 \
  --runtime=python311 \
  --trigger-topic YOUR_PUBSUB_TOPIC \
  --entry-point=publish_to_bigquery \
  --source=cloud_function \
  --set-env-vars="BQ_DATASET=ml_observability,BQ_TABLE=prediction_logs,BQ_LOCATION=US" \
  --allow-unauthenticated
```

## Message format

The function expects Pub/Sub messages whose payload is a JSON object with keys such as:

- `client_id`
- `model_version`
- `source_system`
- `prediction`
- `prediction_probability`
- `latency_ms`
- `status_code`
- `features`
- `timestamp`

The function stores the original feature payload in `input_payload`, extracts monitored features into dedicated columns, and writes a partitioned `ingest_timestamp`.

## Training baseline ingestion

To compute drift with BigQuery, load the training baseline into `ml_observability.training_baseline`.
Use `demo-environment/load_training_baseline_to_bigquery.py`:

```bash
python demo-environment/load_training_baseline_to_bigquery.py \
  --project_id YOUR_PROJECT_ID \
  --dataset ml_observability \
  --table training_baseline \
  --source_pickle Data-Pipeline/data/processed/application_train_merged.pkl
```

## Scheduled drift metrics query

Configure a BigQuery scheduled query using `cloud_function/drift_metrics_query.sql`.
This query calculates daily PSI per monitored feature and writes results to `ml_observability.drift_metrics_daily`.

A good schedule is every 5 or 10 minutes during the demo and hourly for production.

## Drift detection job for Cloud Run

A new drift detection job is available in `cloud_function/drift_detector.py`.
It reads:
- `ml_observability.training_baseline`
- `ml_observability.prediction_logs`

It computes:
- PSI
- KS
- average drift score
- `LOW` / `MEDIUM` / `HIGH` severity

It writes results to `ml_observability.drift_scores`.

### Run locally

```bash
python cloud_function/drift_detector.py
```

### Environment variables

- `BQ_PROJECT` or `GOOGLE_CLOUD_PROJECT`
- `BQ_DATASET=ml_observability`
- `BQ_BASELINE_TABLE=training_baseline`
- `BQ_LOG_TABLE=prediction_logs`
- `BQ_DRIFT_TABLE=drift_summary_daily`
- `BQ_LOCATION=US`
- `DRIFT_WINDOW_DAYS=1`
- `DRIFT_BINS=10`

### Deploy on Cloud Run

Build a container for the repo and use `python cloud_function/drift_detection_job.py` as the command.
Then schedule the Cloud Run job with Cloud Scheduler or Cloud Run Jobs to run whenever new prediction data is available.
