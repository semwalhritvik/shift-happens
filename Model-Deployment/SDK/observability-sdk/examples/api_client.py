import argparse
import json
import pandas as pd
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Send sample prediction requests to the ShiftHappens API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict", help="Prediction API URL")
    parser.add_argument("--client_id", default="client_1", help="Client identifier")
    parser.add_argument("--model_version", default="v1", help="Model version label")
    parser.add_argument("--data_path", default=None, help="Optional path to a CSV/JSON file with sample rows")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to send when using a file")
    return parser.parse_args()


def load_sample_data(path: str, rows: int = 5) -> list[dict]:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    else:
        raise ValueError("Unsupported file type. Use CSV or JSON.")
    return df.head(rows).to_dict(orient="records")


def main():
    args = parse_args()

    if args.data_path:
        rows = load_sample_data(args.data_path, args.rows)
    else:
        rows = [
            {
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 2478.5,
                "DAYS_BIRTH": -10436,
                "DAYS_EMPLOYED": 365243,
                "CODE_GENDER": "M",
            }
        ]

    for idx, features in enumerate(rows, start=1):
        payload = {
            "client_id": args.client_id,
            "model_version": args.model_version,
            "features": features,
        }
        response = requests.post(args.url, json=payload)
        if response.ok:
            print(f"Request {idx} succeeded:", json.dumps(response.json(), indent=2))
        else:
            print(f"Request {idx} failed: {response.status_code} {response.text}")


if __name__ == "__main__":
    main()
