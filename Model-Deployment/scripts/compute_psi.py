import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MONITORED_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute PSI drift scores between a training baseline pickle and live/demo CSV."    
    )
    parser.add_argument(
        "--baseline_pickle",
        default=os.path.join("Data-Pipeline", "data", "processed", "application_train_merged.pkl"),
        help="Path to the training baseline pickle file.",
    )
    parser.add_argument(
        "--live_csv",
        default=os.path.join("demo-environment", "uploads", "test_data_drifted.csv"),
        help="Path to the live or drifted demo CSV file.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of PSI buckets to use for the histogram.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=MONITORED_FEATURES,
        help="List of numeric feature names to compute PSI for.",
    )
    return parser.parse_args()


def load_baseline(baseline_pickle: str) -> pd.DataFrame:
    if not os.path.exists(baseline_pickle):
        raise FileNotFoundError(f"Baseline pickle not found: {baseline_pickle}")
    df = pd.read_pickle(baseline_pickle)
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    return df


def load_live(live_csv: str) -> pd.DataFrame:
    if not os.path.exists(live_csv):
        raise FileNotFoundError(f"Live CSV not found: {live_csv}")
    df = pd.read_csv(live_csv)
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    return df


def compute_bin_edges(training_values: pd.Series, bins: int) -> np.ndarray:
    values = training_values.dropna().values
    if values.size == 0:
        raise ValueError("Training feature contains no values.")
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    # Expand the end edges slightly so the max value falls into the last bucket.
    edges[0] = edges[0] - 1e-9
    edges[-1] = edges[-1] + 1e-9
    return edges


def compute_psi_for_feature(
    training_values: pd.Series,
    live_values: pd.Series,
    bins: int,
) -> Tuple[float, List[int], List[int]]:
    edges = compute_bin_edges(training_values, bins)
    training_counts, _ = np.histogram(training_values.dropna().values, bins=edges)
    live_counts, _ = np.histogram(live_values.dropna().values, bins=edges)

    training_total = training_counts.sum()
    live_total = live_counts.sum()
    if training_total == 0 or live_total == 0:
        raise ValueError("Training or live values have zero valid rows for this feature.")

    training_pct = (training_counts + 1) / (training_total + bins)
    live_pct = (live_counts + 1) / (live_total + bins)
    psi = np.sum((live_pct - training_pct) * np.log(live_pct / training_pct))
    return float(psi), training_counts.tolist(), live_counts.tolist()


def compute_psi_scores(
    baseline_df: pd.DataFrame,
    live_df: pd.DataFrame,
    features: List[str],
    bins: int,
) -> Dict[str, Dict]:
    psi_results = {}
    for feature in features:
        if feature not in baseline_df.columns:
            raise ValueError(f"Feature '{feature}' not found in baseline data.")
        if feature not in live_df.columns:
            raise ValueError(f"Feature '{feature}' not found in live data.")

        psi, training_counts, live_counts = compute_psi_for_feature(
            baseline_df[feature], live_df[feature], bins=bins
        )
        psi_results[feature] = {
            "psi": psi,
            "training_total": int(sum(training_counts)),
            "live_total": int(sum(live_counts)),
            "training_buckets": training_counts,
            "live_buckets": live_counts,
        }
    return psi_results


def print_results(psi_results: Dict[str, Dict]):
    print("\nPSI drift scores")
    print("------------------")
    for feature, result in psi_results.items():
        print(f"{feature}: {result['psi']:.6f}")
        print(f"  training_total={result['training_total']}, live_total={result['live_total']}")
        print(f"  training_buckets={result['training_buckets']}")
        print(f"  live_buckets={result['live_buckets']}")
    average_psi = np.mean([result["psi"] for result in psi_results.values()])
    max_psi = max(result["psi"] for result in psi_results.values())
    print("------------------")
    print(f"Average PSI: {average_psi:.6f}")
    print(f"Max PSI: {max_psi:.6f}")


def main():
    args = parse_args()
    baseline_df = load_baseline(args.baseline_pickle)
    live_df = load_live(args.live_csv)

    psi_results = compute_psi_scores(
        baseline_df=baseline_df,
        live_df=live_df,
        features=args.features,
        bins=args.bins,
    )
    print_results(psi_results)


if __name__ == "__main__":
    main()
