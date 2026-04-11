import pandas as pd
import numpy as np
import os


# Load your final pipeline output
df = pd.read_pickle("Data-Pipeline/data/processed/application_train_merged.pkl")

# Use a 20% holdout for the "Live Demo"
# We'll split this holdout into two halves: one clean, one drifted
holdout = df.tail(int(len(df) * 0.2)).copy()
mid_point = len(holdout) // 2

clean_batch = holdout.iloc[:mid_point].copy()
drift_batch = holdout.iloc[mid_point:].copy()

# --- Save Clean Data ---
clean_batch.to_csv("Demo-Environment/data/test_data_clean.csv", index=False)

# --- Apply Drift to the second batch ---
# Simulate a 30% income drop (Recession)
drift_batch['AMT_INCOME_TOTAL'] = drift_batch['AMT_INCOME_TOTAL'] * 0.70

# Simulate a spike in loan size (Inflation/Risk)
drift_batch['AMT_CREDIT'] = drift_batch['AMT_CREDIT'] * 1.40

# Save Drifted Data
drift_batch.to_csv("Demo-Environment/data/test_data_drifted.csv", index=False)

print("Demo files generated in Demo-Environment/data")
print(f"   - Clean rows: {len(clean_batch)}")
print(f"   - Drifted rows: {len(drift_batch)}")