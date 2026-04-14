import argparse
import time
import joblib
import pandas as pd
from gcp_observability_sdk import ShiftHappensTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate live prediction traffic through Pub/Sub.")
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument("--topic_id", required=True, help="Pub/Sub topic ID")
    parser.add_argument("--model_path", default="model.pkl", help="Path to the trained model pickle")
    parser.add_argument("--data_path", default="data.csv", help="Path to the input data CSV")
    parser.add_argument("--client_id", default="client_1", help="Client identifier to log")
    parser.add_argument("--model_version", default="v1", help="Model version tag")
    parser.add_argument("--rows", type=int, default=50, help="Number of rows to simulate")
    parser.add_argument("--sleep_seconds", type=float, default=0.2, help="Seconds to wait between rows")
    return parser.parse_args()


def main():
    args = parse_args()
    tracker = ShiftHappensTracker(project_id=args.project_id, topic_id=args.topic_id)
    model = joblib.load(args.model_path)

    df = pd.read_csv(args.data_path)
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    sample = df.head(args.rows)

    print(f"Simulating {len(sample)} prediction events through Pub/Sub...")
    for index, row in sample.iterrows():
        features = row.to_dict()
        prediction = model.predict(pd.DataFrame([features]))
        prediction_value = int(prediction[0]) if hasattr(prediction[0], "__int__") else str(prediction[0])

        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(pd.DataFrame([features]))
            if proba is not None:
                probability = float(proba[:, 1][0]) if proba.ndim == 2 and proba.shape[1] > 1 else float(proba.ravel()[0])

        tracker.track_prediction(
            features=features,
            prediction=prediction_value,
            model_version=args.model_version,
            client_id=args.client_id,
            prediction_probability=probability,
            source_system="simulator",
        )

        print(f"published row={index} prediction={prediction_value} probability={probability}")
        time.sleep(args.sleep_seconds)

    print("Simulation complete. Verify that Pub/Sub messages are being ingested downstream.")


if __name__ == "__main__":
    main()
