from google.cloud import bigquery
import pandas as pd


class BigQueryManager:
    def __init__(self, project_id: str, dataset: str, location: str = "US"):
        self.project_id = project_id
        self.dataset = dataset
        self.location = location
        self.client = bigquery.Client(project=project_id)

    def dataset_id(self) -> str:
        return f"{self.project_id}.{self.dataset}"

    def table_id(self, table_name: str) -> str:
        return f"{self.project_id}.{self.dataset}.{table_name}"

    def create_dataset_if_not_exists(self) -> None:
        dataset_id = self.dataset_id()
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = self.location
        try:
            self.client.get_dataset(dataset_id)
        except Exception:
            self.client.create_dataset(dataset, timeout=30)

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        write_disposition: str = "WRITE_APPEND",
        schema: list[bigquery.SchemaField] | None = None,
    ) -> str:
        table_id = self.table_id(table_name)
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            schema=schema,
            autodetect=schema is None,
            source_format=bigquery.SourceFormat.PARQUET,
        )
        job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        return f"Uploaded {len(df)} rows to {table_id}"

    def query_table(self, query: str) -> pd.DataFrame:
        query_job = self.client.query(query)
        return query_job.to_dataframe()

    def insert_json_rows(self, table_name: str, rows: list[dict]) -> list[dict]:
        table_id = self.table_id(table_name)
        errors = self.client.insert_rows_json(table_id, rows)
        return errors
