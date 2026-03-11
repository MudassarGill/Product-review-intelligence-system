import os
import sys
import yaml
import logging
from dotenv import load_dotenv

# Add project root to path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import logging
from src.model.train_model import train_logistic_regression

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
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def main():
    try:
        # Load params
        params = load_params("params.yaml")
        model_params = params.get('model_building', {})
        
        # Define paths
        X_train_path = os.path.join("data", "features", "X_train.npy")
        y_train_path = os.path.join("data", "features", "y_train.npy")
        model_save_path = os.path.join("models", "logistic_regression.pkl")

        # Train model
        train_logistic_regression(
            X_train_path=X_train_path,
            y_train_path=y_train_path,
            model_save_path=model_save_path,
            params=model_params
        )

        logging.info("Model building completed successfully")

    except Exception as e:
        logging.error(f"Model building failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
