import os
import sys
import pandas as pd
import yaml
import logging
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Add project root to path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.connections.s3_connection import S3Operations

# Load environment variables
load_dotenv()

def load_params(params_path: str) -> dict:
    """
    Load parameters from YAML file
    """
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
            logging.info("Parameters loaded successfully")
            return params
    except FileNotFoundError:
        logging.error(f"Parameters file not found at {params_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data: map Score to Sentiment as per exp.ipynb logic
    """
    try:
        logging.info("Preprocessing data...")
        
        if 'Score' in df.columns:
            def get_sentiment(score):
                if score > 3:
                    return 'Positive'
                elif score < 3:
                    return 'Negative'
                else:
                    return 'Neutral'
            df['Sentiment'] = df['Score'].apply(get_sentiment)
            
        logging.info("Data preprocessing completed")
        return df
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, raw_data_path: str):
    """
    Save train and test CSVs
    """
    try:
        os.makedirs(raw_data_path, exist_ok=True)
        train_df.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.info(f"Data saved to {raw_data_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    try:
        # Load params from YAML if available
        params = load_params("params.yaml")
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        raw_data_path = "data/raw"

        # Initialize S3 connection using environment variables
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if not aws_access_key or not aws_secret_key:
            logging.error("AWS credentials not found in environment variables.")
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment variables.")

        s3 = S3Operations(
            bucket_name="data-info-s3",
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key
        )

        # Fetch data from S3
        df = s3.fetch_file_from_s3("data.csv")
        if df is None:
            logging.error("No data fetched from S3, exiting...")
            return

        # Preprocess
        df = preprocess_data(df)

        # Split train/test
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        # Save data
        save_data(train_df, test_df, raw_data_path)

        logging.info("Data ingestion completed successfully")

    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()    