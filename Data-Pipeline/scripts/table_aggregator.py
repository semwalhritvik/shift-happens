import os
import time
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def aggregate_and_merge_dfs(app_train: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Core logic: aggregates numerical features from bureau by SK_ID_CURR, 
    and merges them into application_train.
    """
    logging.info("Identifying numerical columns in the bureau dataset...")
    # Filter only numeric columns
    num_cols = bureau.select_dtypes(include=['number']).columns.tolist()
    
    if 'SK_ID_CURR' not in num_cols:
        logging.error("'SK_ID_CURR' not found or is not numeric in the bureau dataset.")
        return app_train
        
    # We do not want to aggregate the primary key itself or other IDs like SK_ID_BUREAU
    cols_to_aggregate = [col for col in num_cols if col not in ['SK_ID_CURR', 'SK_ID_BUREAU']]
    
    if not cols_to_aggregate:
        logging.warning("No numerical columns available to aggregate in the bureau dataset.")
        return app_train

    logging.info("Aggregating bureau data (mean, sum, max, min, count) by SK_ID_CURR...")
    agg_funcs = ['mean', 'sum', 'max', 'min', 'count']
    
    # Create the aggregation dictionary
    agg_dict = {col: agg_funcs for col in cols_to_aggregate}
    
    # Perform the group by and aggregation
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_dict)
    
    # Flatten multi-level columns
    # resulting columns will look like: BUREAU_DAYS_CREDIT_MEAN, BUREAU_DAYS_CREDIT_MAX, etc.
    bureau_agg.columns = pd.Index([f"BUREAU_{c[0]}_{c[1].upper()}" for c in bureau_agg.columns.tolist()])
    
    # Reset index to make SK_ID_CURR a normal column again
    bureau_agg = bureau_agg.reset_index()
    
    logging.info(f"Aggregated bureau data shape: {bureau_agg.shape}")
    
    logging.info("Merging aggregated features back into the main application dataframe...")
    # Perform a left join so we don't lose any application_train records
    merged_df = app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    
    logging.info(f"Merged dataframe shape: {merged_df.shape}")
    return merged_df

def aggregate_and_merge(app_train_path: str, bureau_path: str, output_path: str) -> None:
    """
    Reads application_train and bureau data, aggregates numerical features 
    from bureau by SK_ID_CURR, and merges them into application_train.
    """
    start_time = time.time()
    
    def load_file(path: str) -> pd.DataFrame:
        if path.endswith('.pkl'):
            return pd.read_pickle(path)
        return pd.read_csv(path)
        
    try:
        logging.info(f"Loading main application table from {app_train_path}...")
        app_train = load_file(app_train_path)
        
        logging.info(f"Loading secondary bureau table from {bureau_path}...")
        bureau = load_file(bureau_path)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return

    merged_df = aggregate_and_merge_dfs(app_train, bureau)
    
    logging.info(f"Saving merged dataframe to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_pickle(output_path)  # Saving to pickle for faster subsequent reads
    
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Aggregation and merge completed successfully in {execution_time:.2f} seconds.")

if __name__ == "__main__":
    aggregate_and_merge(
        app_train_path='data/raw/application_train.csv',
        bureau_path='data/raw/bureau.csv',
        output_path='data/processed/application_train_with_bureau.pkl'
    )
