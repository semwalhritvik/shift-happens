from datetime import datetime
import pandas as pd


class MonitoringManager:
    def __init__(self, bq_manager):
        self.bq_manager = bq_manager

    def log_prediction(
        self,
        client_id: str,
        model_version: str,
        input_data: dict,
        prediction,
        prediction_probability: float | None = None,
        latency_ms: float = 0.0,
        status_code: int = 200,
        source_system: str = "sdk",
    ) -> list[dict]:
        row = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "client_id": client_id,
                "model_version": model_version,
                "prediction": str(prediction),
                "prediction_probability": float(prediction_probability) if prediction_probability is not None else None,
                "latency_ms": float(latency_ms),
                "status_code": status_code,
                "source_system": source_system,
                "input_payload": str(input_data),
            }
        ]
        return self.bq_manager.insert_json_rows("prediction_logs", row)

    def run_data_quality_check(self, df: pd.DataFrame) -> dict:
        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
        }

    def summarize_predictions(self) -> pd.DataFrame:
        query = f"""
        SELECT
          client_id,
          COUNT(*) AS total_predictions,
          AVG(latency_ms) AS avg_latency,
          SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) AS error_count
        FROM `{self.bq_manager.project_id}.{self.bq_manager.dataset}.prediction_logs`
        GROUP BY client_id
        ORDER BY total_predictions DESC
        """
        return self.bq_manager.query_table(query)
