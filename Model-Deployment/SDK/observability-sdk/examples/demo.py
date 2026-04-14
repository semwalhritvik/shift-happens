import pandas as pd
from gcp_observability_sdk import ObservabilitySDK

PROJECT_ID = "your-gcp-project-id"
DATASET = "ml_observability"
MODEL_PATH = "model.pkl"
DATA_PATH = "data.pkl"


def main():
    sdk = ObservabilitySDK(project_id=PROJECT_ID, dataset=DATASET, model_path=MODEL_PATH)

    df = sdk.load_pickle_to_dataframe(DATA_PATH)
    print("Loaded Data: \n", df.head(), "\n")

    print(sdk.upload_dataframe(df, table_name="raw_data", write_disposition="WRITE_TRUNCATE"))

    sample = df.head(5)
    predictions, latency_ms, prediction_probabilities = sdk.predict_with_probability(sample)
    print("Predictions:", predictions)
    print("Latency:", latency_ms, "ms")
    print("Probabilities:", prediction_probabilities)

    for i in range(len(sample)):
        sdk.log_prediction(
            client_id="client_1",
            model_version="v1",
            input_data=sample.iloc[i].to_dict(),
            prediction=predictions[i],
            prediction_probability=(prediction_probabilities[i] if prediction_probabilities is not None else None),
            latency_ms=latency_ms,
        )

    print("Data Quality Report:", sdk.run_data_quality_check(df))
    print("Prediction Summary:\n", sdk.summarize_predictions())


if __name__ == "__main__":
    main()
