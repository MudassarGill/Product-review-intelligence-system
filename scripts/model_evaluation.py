import os
import sys
import numpy as np
import logging
import joblib
import dagshub
import mlflow
import mlflow.sklearn
import json
import time
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure UTF-8 output to avoid charmap errors with emojis
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


# Load environment variables
load_dotenv()

# Add project root to path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import logging

def evaluate_model(X_test_path: str, y_test_path: str, model_path: str):
    """
    Load test data and trained model, predict, and calculate evaluation metrics.
    We are logging results to MLflow.
    """
    try:
        logging.info("Starting model evaluation process")
        
        # Load model parameters for logging
        model_params = {}
        try:
            with open("params.yaml", "r") as f:
                params = yaml.safe_load(f)
                model_params = params.get("model_building", {})
        except Exception as e:
            logging.warning(f"Could not load params.yaml: {e}")
        
        # Load data
        logging.info(f"Loading test data from {X_test_path} and {y_test_path}")
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path, allow_pickle=True)
        
        # Ensure model exists
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logging.info("Loading pre-trained Logistic Regression model")
        model = joblib.load(model_path)
        
        logging.info("Predicting on test data")
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        logging.info("Evaluation metrics calculated")
        
        # Start MLflow tracking using environment variables
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "mudassarhussain6533")
        repo_name = os.getenv("DAGSHUB_REPO_NAME", "Product-review-intelligence-system")
        dagshub_token = os.getenv("CAPSTONE_TEST")

        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST (DagsHub Token) not found in environment variables.")

        # Explicitly set MLflow credentials using the token
        os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        
        experiment_name = "Logistic-Regression-baseline"
        mlflow.set_experiment(experiment_name)
        
        # Disable autologging to ensure manual control and avoid conflicts
        # mlflow.sklearn.autolog()
        
        with mlflow.start_run(run_name="DVC-Repro-Run") as run:
            run_id = run.info.run_id
            logging.info(f"Started MLflow run: {run_id}")
            
            # Log model parameters for traceability
            if model_params:
                mlflow.log_params(model_params)
                logging.info(f"Logged model params: {model_params}")
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            
            # Log the model explicitly and ensure it's detectable
            logging.info("Logging model to MLflow...")
            mlflow.sklearn.log_model(model, "model")
            
            logging.info(f"Model logged to MLflow as 'model' artifact in run {run_id}")
            logging.info(f'Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # Wait for DagsHub to fully sync the artifacts before the next stage reads them
        logging.info("Waiting for artifact sync to complete...")
        time.sleep(5)

        # Save metrics locally
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        metrics_file = os.path.join("reports", "metrics.json")
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved locally to {metrics_file}")

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def main():
    try:
        X_test_path = os.path.join("data", "features", "X_test.npy")
        y_test_path = os.path.join("data", "features", "y_test.npy")
        model_path = os.path.join("models", "logistic_regression.pkl")
        evaluate_model(X_test_path, y_test_path, model_path)
        logging.info("Model evaluation completed successfully")
    except Exception as e:
        # Final fallback log for critical failures
        sys.stderr.write(f"CRITICAL FAILURE: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
