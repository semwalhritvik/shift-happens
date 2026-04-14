terraform {
  required_version = ">= 1.3.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.location
}

resource "google_bigquery_dataset" "ml_observability" {
  dataset_id = var.dataset_id
  project    = var.project_id
  location   = var.location
  description = "Dataset for observability tables and prediction logs."
}

resource "google_bigquery_table" "raw_data" {
  dataset_id = google_bigquery_dataset.ml_observability.dataset_id
  project    = google_bigquery_dataset.ml_observability.project
  table_id   = "raw_data"
  schema     = jsonencode([
    { name = "record_id", type = "STRING", mode = "NULLABLE" },
    { name = "client_id", type = "STRING", mode = "NULLABLE" },
    { name = "event_timestamp", type = "TIMESTAMP", mode = "NULLABLE" },
    { name = "feature_1", type = "FLOAT64", mode = "NULLABLE" },
    { name = "feature_2", type = "STRING", mode = "NULLABLE" },
    { name = "feature_3", type = "INT64", mode = "NULLABLE" },
    { name = "source_system", type = "STRING", mode = "NULLABLE" }
  ])
  time_partitioning {
    type = "DAY"
    field = "event_timestamp"
  }
}

resource "google_bigquery_table" "prediction_logs" {
  dataset_id = google_bigquery_dataset.ml_observability.dataset_id
  project    = google_bigquery_dataset.ml_observability.project
  table_id   = "prediction_logs"
  schema     = jsonencode([
    { name = "request_id", type = "STRING", mode = "NULLABLE" },
    { name = "client_id", type = "STRING", mode = "NULLABLE" },
    { name = "timestamp", type = "TIMESTAMP", mode = "NULLABLE" },
    { name = "model_version", type = "STRING", mode = "NULLABLE" },
    { name = "input_payload", type = "STRING", mode = "NULLABLE" },
    { name = "prediction", type = "STRING", mode = "NULLABLE" },
    { name = "prediction_probability", type = "FLOAT64", mode = "NULLABLE" },
    { name = "latency_ms", type = "FLOAT64", mode = "NULLABLE" },
    { name = "status_code", type = "INT64", mode = "NULLABLE" },
    { name = "source_system", type = "STRING", mode = "NULLABLE" }
  ])
  time_partitioning {
    type = "DAY"
    field = "timestamp"
  }
}

resource "google_bigquery_table" "model_versions" {
  dataset_id = google_bigquery_dataset.ml_observability.dataset_id
  project    = google_bigquery_dataset.ml_observability.project
  table_id   = "model_versions"
  schema     = jsonencode([
    { name = "model_version", type = "STRING", mode = "REQUIRED" },
    { name = "trained_on_date", type = "DATE", mode = "NULLABLE" },
    { name = "training_data_range", type = "STRING", mode = "NULLABLE" },
    { name = "feature_schema_version", type = "STRING", mode = "NULLABLE" },
    { name = "deployed_at", type = "TIMESTAMP", mode = "NULLABLE" },
    { name = "retired_at", type = "TIMESTAMP", mode = "NULLABLE" },
    { name = "notes", type = "STRING", mode = "NULLABLE" }
  ])
}

resource "google_bigquery_table" "data_quality_checks" {
  dataset_id = google_bigquery_dataset.ml_observability.dataset_id
  project    = google_bigquery_dataset.ml_observability.project
  table_id   = "data_quality_checks"
  schema     = jsonencode([
    { name = "date", type = "DATE", mode = "NULLABLE" },
    { name = "client_id", type = "STRING", mode = "NULLABLE" },
    { name = "table_name", type = "STRING", mode = "NULLABLE" },
    { name = "row_count", type = "INT64", mode = "NULLABLE" },
    { name = "column_count", type = "INT64", mode = "NULLABLE" },
    { name = "missing_values", type = "STRING", mode = "NULLABLE" },
    { name = "duplicate_rows", type = "INT64", mode = "NULLABLE" },
    { name = "issues", type = "STRING", mode = "NULLABLE" }
  ])
}
