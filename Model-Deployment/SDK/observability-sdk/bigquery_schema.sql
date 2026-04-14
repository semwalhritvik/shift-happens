-- BigQuery schema templates for the observability SDK
-- Adjust column types and partitioning as needed for your dataset.

CREATE TABLE IF NOT EXISTS `ml_observability.raw_data` (
  record_id STRING,
  client_id STRING,
  event_timestamp TIMESTAMP,
  feature_1 FLOAT64,
  feature_2 STRING,
  feature_3 INT64,
  -- add more feature columns based on your dataset
  source_system STRING
)
PARTITION BY DATE(event_timestamp);

CREATE TABLE IF NOT EXISTS `ml_observability.prediction_logs` (
  request_id STRING,
  client_id STRING,
  timestamp TIMESTAMP,
  model_version STRING,
  input_payload STRING,
  prediction STRING,
  prediction_probability FLOAT64,
  latency_ms FLOAT64,
  status_code INT64,
  source_system STRING
)
PARTITION BY DATE(timestamp);

CREATE TABLE IF NOT EXISTS `ml_observability.feature_distribution_daily` (
  date DATE,
  client_id STRING,
  feature_name STRING,
  distribution_json STRING,
  mean FLOAT64,
  stddev FLOAT64,
  min_value FLOAT64,
  max_value FLOAT64,
  null_rate FLOAT64
);

CREATE TABLE IF NOT EXISTS `ml_observability.prediction_distribution_daily` (
  date DATE,
  client_id STRING,
  model_version STRING,
  prediction_name STRING,
  count INT64,
  probability_avg FLOAT64
);

CREATE TABLE IF NOT EXISTS `ml_observability.data_quality_checks` (
  date DATE,
  client_id STRING,
  table_name STRING,
  row_count INT64,
  column_count INT64,
  missing_values STRING,
  duplicate_rows INT64,
  issues STRING
);

CREATE TABLE IF NOT EXISTS `ml_observability.model_versions` (
  model_version STRING,
  trained_on_date DATE,
  training_data_range STRING,
  feature_schema_version STRING,
  deployed_at TIMESTAMP,
  retired_at TIMESTAMP,
  notes STRING
);

CREATE TABLE IF NOT EXISTS `ml_observability.alert_events` (
  event_id STRING,
  alert_time TIMESTAMP,
  client_id STRING,
  severity STRING,
  alert_type STRING,
  message STRING,
  details STRING
);

CREATE TABLE IF NOT EXISTS `ml_observability.ground_truth_evaluations` (
  evaluation_id STRING,
  client_id STRING,
  timestamp TIMESTAMP,
  model_version STRING,
  actual_value STRING,
  predicted_value STRING,
  error FLOAT64,
  evaluation_notes STRING
);
