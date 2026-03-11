import os
import sys
import logging
import dagshub
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv()

# Add project root to path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import logging

def promote_model(model_name: str):
    try:
        logging.info(f"Starting model promotion process for '{model_name}'")
        
        # Connect to DagsHub
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "mudassarhussain6533")
        repo_name = os.getenv("DAGSHUB_REPO_NAME", "Product-review-intelligence-system")
        dagshub_token = os.getenv("CAPSTONE_TEST")

        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST (DagsHub Token) not found in environment variables.")

        os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
        
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging", "Production"])
        
        staging_version = next((v for v in versions if v.current_stage == "Staging"), None)
        production_version = next((v for v in versions if v.current_stage == "Production"), None)
        
        if not staging_version:
            logging.info("No model found in Staging. Nothing to promote.")
            return

        staging_run = client.get_run(staging_version.run_id)
        staging_acc = staging_run.data.metrics.get("accuracy", 0)
        
        production_acc = 0
        if production_version:
            production_run = client.get_run(production_version.run_id)
            production_acc = production_run.data.metrics.get("accuracy", 0)

        # Guardrails
        ACCURACY_THRESHOLD = 0.50
        
        if staging_acc > production_acc:
            if staging_acc >= ACCURACY_THRESHOLD:
                client.transition_model_version_stage(
                    name=model_name, version=staging_version.version,
                    stage="Production", archive_existing_versions=True
                )
                logging.info(f"Promoted version {staging_version.version} to Production.")
            else:
                logging.info(f"Staging accuracy {staging_acc:.4f} below threshold {ACCURACY_THRESHOLD}")
        else:
            logging.info("Production model is already better or equal.")

    except Exception as e:
        logging.error(f"Error during model promotion: {e}")
        raise

def main():
    model_name = "Product-Review-Sentiment-Model"
    promote_model(model_name)

if __name__ == "__main__":
    main()
