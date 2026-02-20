import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in a dataframe.
    Numerical columns: Replaced by median.
    Categorical columns: Replaced by mode.
    Empty columns (all NaNs) are left as NaNs or dropped based on preference.
    Here we leave empty columns as NaNs since median/mode cannot be computed.
    """
    df_imputed = df.copy()
    
    for col in df_imputed.columns:
        if df_imputed[col].isnull().all():
            logging.warning(f"Column '{col}' is entirely empty. Skipping imputation.")
            continue
            
        if pd.api.types.is_numeric_dtype(df_imputed[col]):
            median_val = df_imputed[col].median()
            if not pd.isna(median_val):
                df_imputed[col] = df_imputed[col].fillna(median_val)
        else:
            # For categorical/object columns, use the mode
            mode_series = df_imputed[col].mode()
            if not mode_series.empty:
                mode_val = mode_series.iloc[0]
                df_imputed[col] = df_imputed[col].fillna(mode_val)
                
    return df_imputed
