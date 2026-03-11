import os
import sys
import argparse
import dagshub
import mlflow
import time
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv()

# Add project root to path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import logging

def register_best_model(experiment_name, model_name, stage, max_retries=3, retry_delay=10):
    try:
        logging.info("Starting model registry process")
        
        # Connect to DagsHub using environment variables
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
        
        # Allow DagsHub connection to stabilize
        time.sleep(5)
        
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            # List all experiments to help debugging
            all_experiments = client.search_experiments()
            exp_names = [e.name for e in all_experiments]
            logging.error(f"Experiment '{experiment_name}' not found. Available experiments: {exp_names}")
            raise ValueError(f"Experiment '{experiment_name}' not found.")

        # Retry logic — DagsHub artifact propagation can be slow
        best_run = None
        for attempt in range(1, max_retries + 1):
            logging.info(f"Attempt {attempt}/{max_retries}: Searching for runs in experiment '{experiment_name}' (ID: {experiment.experiment_id})")
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.accuracy DESC", "attributes.start_time DESC"],
                max_results=20
            )
            
            logging.info(f"Found {len(runs)} runs in experiment.")
            
            for run in runs:
                run_id = run.info.run_id
                try:
                    artifacts = client.list_artifacts(run_id)
                    artifact_paths = [art.path for art in artifacts]
                    logging.info(f"Checking run {run_id}: artifacts = {artifact_paths}")
                    
                    if "model" in artifact_paths:
                        best_run = run
                        logging.info(f"Found valid run {run_id} with 'model' artifact.")
                        break
                except Exception as e:
                    logging.warning(f"Could not list artifacts for run {run_id}: {e}")
            
            if best_run:
                break
            
            if attempt < max_retries:
                logging.info(f"No valid run found yet. Waiting {retry_delay}s before retry...")
                time.sleep(retry_delay)
        
        if best_run:
            run_id = best_run.info.run_id
            accuracy = best_run.data.metrics.get("accuracy", 0)
            # Build the source URI pointing to the artifact location on the tracking server
            model_source = f"{tracking_uri}/artifacts/{experiment.experiment_id}/{run_id}/artifacts/model"
            
            logging.info(f"Registering best model from run {run_id} with accuracy {accuracy}")
            
            # Ensure the registered model exists (create if not)
            try:
                client.get_registered_model(model_name)
                logging.info(f"Registered model '{model_name}' already exists.")
            except Exception:
                client.create_registered_model(model_name)
                logging.info(f"Created registered model '{model_name}'.")
            
            # Create a new version using the client API directly
            # This avoids the mlflow.register_model() logged_model metadata check
            # that fails on DagsHub remote tracking servers
            result = client.create_model_version(
                name=model_name,
                source=model_source,
                run_id=run_id
            )
            version = result.version
            logging.info(f"Created model version {version}")
            
            # Transition to stage
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True
            )
            logging.info(f"Model {model_name} version {version} transitioned to {stage}")
            
            # Save status
            os.makedirs("reports", exist_ok=True)
            with open("reports/registry_status.txt", "w") as f:
                f.write(f"VERSION: {version}\nACCURACY: {accuracy}\nRUN_ID: {run_id}")
            logging.info("Registry status saved to reports/registry_status.txt")
        else:
            logging.error(f"No valid runs found with 'model' artifact in experiment '{experiment_name}' after {max_retries} attempts.")
            raise ValueError(f"No valid runs found with 'model' artifact after {max_retries} attempts.")

    except Exception as e:
        logging.error(f"Error during model registry: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    
    register_best_model(args.experiment_name, args.model_name, args.stage)

if __name__ == "__main__":
    main()
