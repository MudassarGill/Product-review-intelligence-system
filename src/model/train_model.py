import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
import logging

def train_logistic_regression(X_train_path: str, y_train_path: str, model_save_path: str, params: dict):
    """
    Train a Logistic Regression model and save it.
    """
    try:
        logging.info(f"Loading features from {X_train_path} and {y_train_path}")
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path, allow_pickle=True)

        logging.info("Initializing Logistic Regression model with params: %s", params)
        model = LogisticRegression(
            max_iter=params.get('max_iter', 1000),
            random_state=params.get('random_state', 42)
        )

        logging.info("Fitting model...")
        model.fit(X_train, y_train)

        logging.info(f"Saving model to {model_save_path}")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
        
        return model

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise
