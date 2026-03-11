import os
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "mudassarhussain6533")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "Product-review-intelligence-system")
dagshub_token = os.getenv("CAPSTONE_TEST")
os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
client = MlflowClient(tracking_uri=tracking_uri)

experiment_name = "Logistic-Regression-baseline"
experiment = client.get_experiment_by_name(experiment_name)
if experiment:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=3, order_by=["attributes.start_time DESC"])
    for run in runs:
        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")
        artifacts = client.list_artifacts(run.info.run_id)
        for art in artifacts:
            print(f"  Artifact: {art.path}")
            if art.path == "model":
                sub_arts = client.list_artifacts(run.info.run_id, path="model")
                for sub in sub_arts:
                    print(f"    - {sub.path}")
