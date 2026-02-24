import os
import time
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_file(path: str) -> pd.DataFrame:
    if path.endswith('.pkl'):
        return pd.read_pickle(path)
    return pd.read_csv(path)

def _aggregate_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Helper function to aggregate numerical features by SK_ID_CURR.
    """
    logging.info(f"Identifying numerical columns for {prefix} dataset...")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if 'SK_ID_CURR' not in num_cols:
        logging.error(f"'SK_ID_CURR' not found or is not numeric in the {prefix} dataset.")
        return pd.DataFrame()
        
    # Exclude primary keys and other common ID columns from aggregation
    cols_to_aggregate = [col for col in num_cols if col not in ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']]
    
    if not cols_to_aggregate:
        logging.warning(f"No numerical columns available to aggregate in the {prefix} dataset.")
        return pd.DataFrame()

    logging.info(f"Aggregating {prefix} data (mean, sum, max, min, count) by SK_ID_CURR...")
    agg_funcs = ['mean', 'sum', 'max', 'min', 'count']
    agg_dict = {col: agg_funcs for col in cols_to_aggregate}
    
    # Perform the group by and aggregation
    df_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    
    # Flatten multi-level columns
    df_agg.columns = pd.Index([f"{prefix}_{c[0]}_{c[1].upper()}" for c in df_agg.columns.tolist()])
    
    # Reset index to make SK_ID_CURR a normal column again
    df_agg = df_agg.reset_index()
    
    logging.info(f"Aggregated {prefix} data shape: {df_agg.shape}")
    return df_agg

def aggregate_bureau(bureau_path: str, output_path: str) -> str:
    start_time = time.time()
    try:
        logging.info(f"Loading bureau table from {bureau_path}...")
        bureau = load_file(bureau_path)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return ""
        
    bureau_agg = _aggregate_table(bureau, "BUREAU")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bureau_agg.to_pickle(output_path)
    logging.info(f"Bureau aggregation completed in {time.time() - start_time:.2f} seconds. Saved to {output_path}")
    return output_path

def aggregate_previous_applications(prev_app_path: str, output_path: str) -> str:
    start_time = time.time()
    try:
        logging.info(f"Loading previous applications table from {prev_app_path}...")
        prev_app = load_file(prev_app_path)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return ""
        
    prev_app_agg = _aggregate_table(prev_app, "PREV_APP")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prev_app_agg.to_pickle(output_path)
    logging.info(f"Previous applications aggregation completed in {time.time() - start_time:.2f} seconds. Saved to {output_path}")
    return output_path

def aggregate_installments(installments_path: str, output_path: str) -> str:
    start_time = time.time()
    try:
        logging.info(f"Loading installments table from {installments_path}...")
        installments = load_file(installments_path)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return ""
        
    installments_agg = _aggregate_table(installments, "INSTALLMENTS")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    installments_agg.to_pickle(output_path)
    logging.info(f"Installments aggregation completed in {time.time() - start_time:.2f} seconds. Saved to {output_path}")
    return output_path

def merge_features(app_train_path: str, bureau_agg_path: str, prev_app_agg_path: str, installments_agg_path: str, output_path: str) -> str:
    start_time = time.time()
    try:
        logging.info(f"Loading main application table from {app_train_path}...")
        merged_df = load_file(app_train_path)
    except FileNotFoundError as e:
        logging.error(f"Main application table not found: {e}")
        return ""
        
    paths_and_names = [
        (bureau_agg_path, "Bureau"),
        (prev_app_agg_path, "Previous Applications"),
        (installments_agg_path, "Installments")
    ]
    
    for path, name in paths_and_names:
        if path and os.path.exists(path):
            try:
                logging.info(f"Merging {name} features from {path}...")
                agg_df = load_file(path)
                merged_df = merged_df.merge(agg_df, on='SK_ID_CURR', how='left')
            except Exception as e:
                logging.error(f"Error merging {name} features: {e}")
        else:
            logging.warning(f"Path for {name} aggregation ({path}) does not exist. Skipping.")
            
    logging.info(f"Final merged dataframe shape: {merged_df.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_pickle(output_path)
    logging.info(f"Merge completed successfully in {time.time() - start_time:.2f} seconds. Saved to {output_path}")
    
    return output_path
