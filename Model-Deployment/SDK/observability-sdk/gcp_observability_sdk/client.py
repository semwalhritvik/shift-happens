from .bigquery_manager import BigQueryManager
from .model_manager import ModelManager
from .monitoring import MonitoringManager
from .publisher import ShiftHappensTracker
from .utils import load_pickle_data, normalize_dataframe


class ObservabilitySDK:
    """SDK wrapper that supports model inference, BigQuery ingestion, and observability logging."""

    def __init__(
        self,
        project_id: str,
        dataset: str,
        model_path: str,
        topic_id: str | None = None,
        location: str = "US",
    ):
        self.project_id = project_id
        self.dataset = dataset
        self.model_path = model_path
        self.location = location

        self.bq_manager = BigQueryManager(project_id, dataset, location)
        self.bq_manager.create_dataset_if_not_exists()

        self.model_manager = ModelManager(model_path)
        self.monitoring_manager = MonitoringManager(self.bq_manager)

        self.tracker = None
        if topic_id is not None:
            self.tracker = ShiftHappensTracker(project_id=project_id, topic_id=topic_id)

    def load_pickle_to_dataframe(self, file_path: str):
        df = load_pickle_data(file_path)
        return normalize_dataframe(df)

    def upload_dataframe(
        self,
        df,
        table_name: str,
        write_disposition: str = "WRITE_APPEND",
        schema=None,
    ):
        self.bq_manager.create_dataset_if_not_exists()
        return self.bq_manager.upload_dataframe(df, table_name, write_disposition, schema=schema)

    def predict(self, input_df):
        return self.model_manager.predict(input_df)

    def predict_with_probability(self, input_df):
        return self.model_manager.predict_with_proba(input_df)

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
    ):
        self.bq_manager.create_dataset_if_not_exists()
        result = self.monitoring_manager.log_prediction(
            client_id=client_id,
            model_version=model_version,
            input_data=input_data,
            prediction=prediction,
            prediction_probability=prediction_probability,
            latency_ms=latency_ms,
            status_code=status_code,
            source_system=source_system,
        )
        if self.tracker is not None:
            try:
                self.tracker.track_prediction(
                    features=input_data,
                    prediction=prediction,
                    model_version=model_version,
                    client_id=client_id,
                    prediction_probability=prediction_probability,
                    source_system=source_system,
                )
            except Exception:
                pass
        return result

    def run_data_quality_check(self, df):
        return self.monitoring_manager.run_data_quality_check(df)

    def summarize_predictions(self):
        return self.monitoring_manager.summarize_predictions()


__all__ = ["ShiftHappensTracker", "ObservabilitySDK"]
