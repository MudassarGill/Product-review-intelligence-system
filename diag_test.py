import sys
import os
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

print("Imports successful")

try:
    from src.logger import logging
    logging.info("Diagnostic: Logger imported successfully")
except Exception as e:
    print(f"Diagnostic: Logger import failed: {e}")

print("Diagnostic: End of script")
