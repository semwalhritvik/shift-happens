import os
import zipfile
import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_and_extract_data():
    competition_name = 'home-credit-default-risk'
    raw_data_dir = 'data/raw'
    
    # Create the raw data directory if it doesn't exist
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Attempt to authenticate with Kaggle API
    try:
        api = KaggleApi()
        api.authenticate()
        logging.info("Successfully authenticated with Kaggle API.")
    except Exception as e:
        logging.error(f"Failed to authenticate with Kaggle API: {e}")
        logging.error("Please ensure your 'kaggle.json' file is placed in '~/.kaggle/' and has the correct permissions.")
        return

    # Attempt to download the dataset/competition files
    try:
        logging.info(f"Downloading data for competition: '{competition_name}'...")
        # Note: If it's a dataset, use api.dataset_download_cli instead. 
        # 'home-credit-default-risk' is a competition.
        api.competition_download_cli(competition_name, path=raw_data_dir)
        logging.info("Download completed.")
    except Exception as e:
        logging.error(f"Failed to download data from Kaggle: {e}")
        return

    # Extract the zip file
    zip_filename = f"{competition_name}.zip"
    zip_filepath = os.path.join(raw_data_dir, zip_filename)
    
    if os.path.exists(zip_filepath):
        try:
            logging.info(f"Extracting '{zip_filename}' into '{raw_data_dir}'...")
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(raw_data_dir)
            logging.info("Extraction complete.")
            
            # Clean up the zip file to save space
            os.remove(zip_filepath)
            logging.info("Removed zip file.")
        except Exception as e:
            logging.error(f"Failed to extract or remove the zip file: {e}")
    else:
        logging.warning("No zip file found. The files may have been downloaded uncompressed or under a different name.")

if __name__ == "__main__":
    download_and_extract_data()
