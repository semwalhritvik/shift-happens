import base64
import json
import os
from datetime import datetime
from google.cloud import bigquery


BQ_DATASET = os.environ.get("BQ_DATASET", "ml_observability")
BQ_TABLE = os.environ.get("BQ_TABLE", "prediction_logs")
MONITORED_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]


def _create_bq_table_if_missing(client: bigquery.Client, dataset_id: str, table_id: str):
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = os.environ.get("BQ_LOCATION", "US")
        client.create_dataset(dataset, timeout=30)

    full_table_id = f"{client.project}.{dataset_id}.{table_id}"
    try:
        client.get_table(full_table_id)
    except Exception:
        schema = [
            bigquery.SchemaField("request_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("client_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("model_version", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("source_system", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("prediction", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("prediction_probability", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("latency_ms", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("status_code", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("input_payload", "STRING", mode="NULLABLE"),
        ]
        for feature in MONITORED_FEATURES:
            schema.append(bigquery.SchemaField(feature, "FLOAT", mode="NULLABLE"))
        schema.append(bigquery.SchemaField("ingest_timestamp", "TIMESTAMP", mode="REQUIRED"))

        table = bigquery.Table(full_table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(field="ingest_timestamp")
        client.create_table(table, exists_ok=True)


def _decode_pubsub_message(event):
    if "data" not in event:
        raise ValueError("No data field in Pub/Sub message")
    payload_bytes = base64.b64decode(event["data"])
    return json.loads(payload_bytes.decode("utf-8"))


def _normalize_payload(payload: dict) -> dict:
    features = payload.get("features", {}) or {}
    event = {
        "request_id": payload.get("request_id") or payload.get("client_id", "unknown") + "-" + datetime.utcnow().strftime("%Y%m%d%H%M%S%f"),
        "client_id": payload.get("client_id"),
        "timestamp": payload.get("timestamp") or datetime.utcnow().isoformat(),
        "model_version": payload.get("model_version"),
        "source_system": payload.get("source_system"),
        "prediction": str(payload.get("prediction", "")),
        "prediction_probability": payload.get("prediction_probability"),
        "latency_ms": payload.get("latency_ms"),
        "status_code": payload.get("status_code"),
        "input_payload": json.dumps(features, default=str),
        "ingest_timestamp": datetime.utcnow().isoformat(),
    }
    for feature in MONITORED_FEATURES:
        event[feature] = float(features.get(feature)) if features.get(feature) is not None else None
    return event


def publish_to_bigquery(event, context):
    """Cloud Function entrypoint triggered by Pub/Sub."""
    client = bigquery.Client()
    _create_bq_table_if_missing(client, BQ_DATASET, BQ_TABLE)

    pubsub_payload = _decode_pubsub_message(event)
    row = _normalize_payload(pubsub_payload)
    table_id = f"{client.project}.{BQ_DATASET}.{BQ_TABLE}"

    errors = client.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")
    return {
        "status": "success",
        "inserted_row": row,
    }
