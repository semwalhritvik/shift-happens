CREATE SCHEMA IF NOT EXISTS `ml_observability`;

CREATE TABLE IF NOT EXISTS `ml_observability.training_baseline` (
  AMT_INCOME_TOTAL FLOAT64,
  AMT_CREDIT FLOAT64,
  AMT_ANNUITY FLOAT64,
  DAYS_BIRTH FLOAT64,
  DAYS_EMPLOYED FLOAT64,
  TARGET INT64
);

CREATE TABLE IF NOT EXISTS `ml_observability.prediction_logs` (
  request_id STRING NOT NULL,
  client_id STRING,
  timestamp TIMESTAMP NOT NULL,
  model_version STRING,
  source_system STRING,
  prediction STRING,
  prediction_probability FLOAT64,
  latency_ms FLOAT64,
  status_code INT64,
  input_payload STRING,
  AMT_INCOME_TOTAL FLOAT64,
  AMT_CREDIT FLOAT64,
  AMT_ANNUITY FLOAT64,
  DAYS_BIRTH FLOAT64,
  DAYS_EMPLOYED FLOAT64,
  ingest_timestamp TIMESTAMP NOT NULL
)
PARTITION BY DATE(ingest_timestamp);

CREATE TABLE IF NOT EXISTS `ml_observability.drift_metrics_daily` (
  date DATE,
  feature_name STRING,
  psi FLOAT64,
  training_total INT64,
  live_total INT64,
  training_buckets STRING,
  live_buckets STRING,
  created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `ml_observability.drift_summary_daily` (
  date DATE NOT NULL,
  feature_name STRING NOT NULL,
  psi FLOAT64,
  ks FLOAT64,
  avg_score FLOAT64,
  severity STRING,
  training_total INT64,
  live_total INT64,
  training_buckets STRING,
  live_buckets STRING,
  created_at TIMESTAMP NOT NULL
);
