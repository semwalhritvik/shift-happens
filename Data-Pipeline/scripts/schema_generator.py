import logging
import tensorflow_data_validation as tfdv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_schema_and_validate(train_data_path: str, eval_data_path: str = None) -> None:
    """
    Generates a schema based on train_data_path using TFDV.
    If eval_data_path is provided, checks it for anomalies against the schema.
    """
    logging.info(f"Generating statistics and inferring schema for {train_data_path}...")
    try:
        train_stats = tfdv.generate_statistics_from_csv(data_location=train_data_path)
        schema = tfdv.infer_schema(statistics=train_stats)
        logging.info("Schema successfully generated.")
    except Exception as e:
        logging.error(f"Failed to generate schema: {e}")
        return

    if eval_data_path:
        logging.info(f"Validating {eval_data_path} against the inferred schema...")
        try:
            eval_stats = tfdv.generate_statistics_from_csv(data_location=eval_data_path)
            anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)
            
            # TFDV anomalies object is a protobuf. We check if there are any anomalies listed.
            if len(anomalies.anomaly_info) > 0:
                logging.warning("Anomalies found in the evaluation data!")
                for feature_name, anomaly_info in anomalies.anomaly_info.items():
                    logging.warning(f"Feature: {feature_name}, Analysis: {anomaly_info.description}")
            else:
                logging.info("No anomalies found. Data matches the schema.")
        except Exception as e:
            logging.error(f"Failed to validate data: {e}")

if __name__ == "__main__":
    # Example usage:
    # generate_schema_and_validate('data/raw/application_train.csv', 'data/raw/application_test.csv')
    pass
