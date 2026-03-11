import mlflow
import dagshub
import os
from dotenv import load_dotenv

load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
repo_name = os.getenv("DAGSHUB_REPO_NAME")
dagshub_token = os.getenv("CAPSTONE_TEST")

os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

print(f"Initializing DagsHub for {repo_owner}/{repo_name}...")
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
print("DagsHub initialized.")

tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
mlflow.set_tracking_uri(tracking_uri)
print(f"Tracking URI set to {tracking_uri}")

experiment_name = "Logistic-Regression-baseline"
print(f"Setting experiment to {experiment_name}...")
mlflow.set_experiment(experiment_name)
print("Experiment set successfully.")
