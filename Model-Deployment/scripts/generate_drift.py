import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate clean and drifted demo CSVs from the final pipeline output.")
    parser.add_argument(
        "--source_pickle",
        default=os.path.join("Data-Pipeline", "data", "processed", "application_train_merged.pkl"),
        help="Path to application_train_merged.pkl produced by your data pipeline.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("demo-environment", "uploads"),
        help="Directory where demo CSV files are written.",
    )
    parser.add_argument(
        "--holdout_fraction",
        type=float,
        default=0.2,
        help="Fraction of rows to reserve for the demo live holdout.",
    )
    parser.add_argument(
        "--income_multiplier",
        type=float,
        default=0.7,
        help="Multiplier applied to AMT_INCOME_TOTAL for the drifted dataset.",
    )
    parser.add_argument(
        "--credit_multiplier",
        type=float,
        default=1.4,
        help="Multiplier applied to AMT_CREDIT for the drifted dataset.",
    )
    return parser.parse_args()


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    return df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.source_pickle):
        raise FileNotFoundError(
            f"Source pickle not found: {args.source_pickle}.\n"
            "Run your data pipeline first or point --source_pickle to a valid application_train_merged.pkl file."
        )

    print(f"Loading source data from {args.source_pickle}...")
    df = pd.read_pickle(args.source_pickle)
    df = _prepare_df(df)

    holdout_size = int(len(df) * args.holdout_fraction)
    if holdout_size < 2:
        raise ValueError("Holdout fraction is too small for the dataset size.")

    holdout = df.tail(holdout_size).copy()
    mid_point = len(holdout) // 2

    clean_batch = holdout.iloc[:mid_point].copy()
    drift_batch = holdout.iloc[mid_point:].copy()

    if "AMT_INCOME_TOTAL" in drift_batch.columns:
        drift_batch["AMT_INCOME_TOTAL"] = drift_batch["AMT_INCOME_TOTAL"] * args.income_multiplier
    if "AMT_CREDIT" in drift_batch.columns:
        drift_batch["AMT_CREDIT"] = drift_batch["AMT_CREDIT"] * args.credit_multiplier
    if "DAYS_EMPLOYED" in drift_batch.columns:
        drift_batch["DAYS_EMPLOYED"] = (drift_batch["DAYS_EMPLOYED"] * 0.5).astype(int)

    clean_path = os.path.join(args.output_dir, "test_data_clean.csv")
    drift_path = os.path.join(args.output_dir, "test_data_drifted.csv")

    clean_batch.to_csv(clean_path, index=False)
    drift_batch.to_csv(drift_path, index=False)

    print("✅ Demo data generated successfully.")
    print(f"  clean data: {clean_path} ({len(clean_batch)} rows)")
    print(f"  drifted data: {drift_path} ({len(drift_batch)} rows)")


if __name__ == "__main__":
    main()
