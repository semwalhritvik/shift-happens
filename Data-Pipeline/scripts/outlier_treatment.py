import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def treat_days_employed_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the DAYS_EMPLOYED column and replaces any values 
    equal to 365243 with NaN.
    """
    if 'DAYS_EMPLOYED' not in df.columns:
        logging.warning("Column 'DAYS_EMPLOYED' not found in the dataframe.")
        return df

    df_clean = df.copy()
    outlier_value = 365243
    
    outlier_mask = df_clean['DAYS_EMPLOYED'] == outlier_value
    outliers_count = outlier_mask.sum()
    
    if outliers_count > 0:
        logging.info(f"Replacing {outliers_count} outliers ({outlier_value}) in DAYS_EMPLOYED with NaN.")
        df_clean.loc[outlier_mask, 'DAYS_EMPLOYED'] = np.nan
    else:
        logging.info("No outliers found in DAYS_EMPLOYED column.")
        
    return df_clean

def process_and_save_data(input_path: str, output_path: str) -> None:
    """
    Reads the CSV, applies outlier treatment, and saves to a pickle file.
    """
    logging.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"File {input_path} not found.")
        return

    df_cleaned = treat_days_employed_outliers(df)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logging.info(f"Saving cleaned dataframe to {output_path}...")
    df_cleaned.to_pickle(output_path)
    logging.info("Save complete.")

if __name__ == "__main__":
    # Example usage:
    # process_and_save_data('data/raw/application_train.csv', 'data/processed/application_train_cleaned.pkl')
    pass
