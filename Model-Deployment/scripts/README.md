# Demo Environment

This folder contains the Phase 2 demo scripts for clean and drifted traffic injection.

## Files

- `generate_drift.py`: Creates `test_data_clean.csv` and `test_data_drifted.csv` from your pipeline output.
- `simulate_live_traffic.py`: Publishes clean prediction events to Pub/Sub using the async SDK.
- `simulate_drift_traffic.py`: Publishes drifted prediction events to Pub/Sub using the async SDK.

## Usage

1. Generate demo CSVs:

```powershell
python demo-environment/generate_drift.py \
  --source_pickle Data-Pipeline/data/processed/application_train_merged.pkl \
  --output_dir demo-environment/uploads
```

2. Compute PSI for the drift dataset:

```powershell
python demo-environment/compute_psi.py \
  --baseline_pickle Data-Pipeline/data/processed/application_train_merged.pkl \
  --live_csv demo-environment/uploads/test_data_drifted.csv
```

3. Run clean traffic:

```powershell
python demo-environment/simulate_live_traffic.py \
  --project_id YOUR_PROJECT_ID \
  --topic_id YOUR_TOPIC_ID \
  --model_path Model-Development/models/final_model_debiased.pkl \
  --data_path demo-environment/uploads/test_data_clean.csv
```

3. Run drift traffic:

```powershell
python demo-environment/simulate_drift_traffic.py \
  --project_id YOUR_PROJECT_ID \
  --topic_id YOUR_TOPIC_ID \
  --model_path Model-Development/models/final_model_debiased.pkl \
  --data_path demo-environment/uploads/test_data_drifted.csv
```

## Phase 3: Ingestion Pipeline

Before running the simulators, deploy the Cloud Function in `cloud_function/main.py`.
It will ingest Pub/Sub messages into BigQuery table `ml_observability.prediction_logs`.

The Cloud Function uses these environment variables:
- `BQ_DATASET=ml_observability`
- `BQ_TABLE=prediction_logs`
- `BQ_LOCATION=US`

Before you can compute drift in BigQuery, load the training baseline into `ml_observability.training_baseline` with:

```powershell
python demo-environment/load_training_baseline_to_bigquery.py \
  --project_id YOUR_PROJECT_ID \
  --dataset ml_observability \
  --table training_baseline \
  --source_pickle Data-Pipeline/data/processed/application_train_merged.pkl
```

Then configure the scheduled query in `cloud_function/drift_metrics_query.sql` to populate `ml_observability.drift_metrics_daily`.

Once the function and baseline are ready, run the clean traffic script first, then the drift script.
