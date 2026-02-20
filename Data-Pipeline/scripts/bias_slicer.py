import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, false_positive_rate

# Configure logging to output to a file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bias_evaluation.log"),
        logging.StreamHandler()
    ]
)

def evaluate_bias(df: pd.DataFrame, 
                  y_true_col: str = 'TARGET', 
                  y_pred_col: str = 'PREDICTION', 
                  sensitive_feature_col: str = 'CODE_GENDER') -> None:
    """
    Evaluates the model's performance (accuracy and false positive rate)
    across different groups defined by the sensitive feature (e.g., gender)
    to detect potential bias.
    """
    logging.info(f"Evaluating bias across the sensitive feature: '{sensitive_feature_col}'")
    
    # Check if necessary columns exist
    for col in [y_true_col, y_pred_col, sensitive_feature_col]:
        if col not in df.columns:
            logging.error(f"Column '{col}' not found in the dataframe. Aborting evaluation.")
            return

    # Define the metrics we want to calculate
    # Note: false_positive_rate is available in fairlearn.metrics
    metrics = {
        'accuracy': accuracy_score,
        'false_positive_rate': false_positive_rate
    }

    try:
        # Create a MetricFrame to compute metrics for different subgroups
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=df[y_true_col],
            y_pred=df[y_pred_col],
            sensitive_features=df[sensitive_feature_col]
        )
        
        # Log overall metrics
        logging.info("--- Overall Model Metrics ---")
        overall_metrics = metric_frame.overall
        for metric_name, value in overall_metrics.items():
            logging.info(f"Overall {metric_name}: {value:.4f}")

        # Log metrics by group
        logging.info("--- Metrics by Group ---")
        by_group_metrics = metric_frame.by_group
        for group_name, row in by_group_metrics.iterrows():
            logging.info(f"Group '{group_name}':")
            for metric_name, value in row.items():
                logging.info(f"  - {metric_name}: {value:.4f}")

        # Highlight maximum differences (potential bias)
        logging.info("--- Metric Differences across Groups ---")
        diffs = metric_frame.difference()
        for metric_name, value in diffs.items():
            logging.info(f"Maximum difference in {metric_name} between any two groups: {value:.4f}")
            if value > 0.05:  # Arbitrary threshold to flag potential concern
                logging.warning(f"High disparity found in '{metric_name}' across groups! Max difference: {value:.4f}")

    except Exception as e:
        logging.error(f"An error occurred during bias evaluation: {e}")

if __name__ == "__main__":
    # Example usage with simulated data:
    logging.info("Starting bias evaluation script...")
    
    # Simulating a dataframe with predictions and a sensitive demographic feature
    np.random.seed(42)
    n_samples = 1000
    
    dummy_data = {
        'TARGET': np.random.randint(0, 2, n_samples),
        'PREDICTION': np.random.randint(0, 2, n_samples),
        'CODE_GENDER': np.random.choice(['M', 'F'], n_samples)
    }
    
    df_simulated = pd.DataFrame(dummy_data)
    
    # Introduce some synthetic bias for demonstration purposes
    # For instance, let's make predictions for 'F' more often wrong (False Positives)
    # when target is 0
    f_mask = (df_simulated['CODE_GENDER'] == 'F') & (df_simulated['TARGET'] == 0)
    
    # Generate random predictions for the entire dataframe, but only apply them where f_mask is true
    random_preds = np.random.choice([0, 1], size=len(df_simulated), p=[0.2, 0.8])
    df_simulated['PREDICTION'] = np.where(f_mask, random_preds, df_simulated['PREDICTION'])
    
    evaluate_bias(df=df_simulated)
    
    logging.info("Bias evaluation script finished.")
