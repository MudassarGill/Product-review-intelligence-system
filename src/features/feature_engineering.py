import pandas as pd
import numpy as np
import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import joblib
import logging
from sklearn.feature_extraction.text import CountVectorizer

def build_features(train_path: str, test_path: str, out_dir: str, max_features: int = 50):
    """
    Load preprocessed data, extract features using CountVectorizer,
    and save the arrays and the trained vectorizer.
    """
    try:
        logging.info(f"Loading processed data from {train_path} and {test_path}")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Handle potential missing values in text
        train_df['Cleaned_Text'] = train_df['Cleaned_Text'].fillna('')
        test_df['Cleaned_Text'] = test_df['Cleaned_Text'].fillna('')
        
        # Identify the target column dynamically based on existing data
        target_col = None
        for col in ['Sentiment', 'summary', 'Target']:
            if col in train_df.columns:
                target_col = col
                break
                
        if not target_col:
            logging.warning("Target column not found. Defaulting to 'summary'")
            target_col = 'summary'
            
        y_train = train_df[target_col].values if target_col in train_df.columns else None
        y_test = test_df[target_col].values if target_col in test_df.columns else None
        
        logging.info(f"Initializing CountVectorizer with max_features={max_features}")
        cv = CountVectorizer(max_features=max_features)
        
        logging.info("Fitting and transforming training data")
        X_train = cv.fit_transform(train_df['Cleaned_Text']).toarray()
        
        logging.info("Transforming testing data")
        X_test = cv.transform(test_df['Cleaned_Text']).toarray()
        
        # Create directories
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Save the trained vectorizer
        vectorizer_path = os.path.join('models', 'count_vectorizer.pkl')
        joblib.dump(cv, vectorizer_path)
        logging.info(f"Vectorizer model saved to {vectorizer_path}")
        
        # Save feature arrays
        np.save(os.path.join(out_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(out_dir, 'X_test.npy'), X_test)
        if y_train is not None:
            np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
        if y_test is not None:
            np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
            
        logging.info(f"Feature arrays saved to {out_dir}")

    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise

def main():
    try:
        train_path = os.path.join("data", "processed", "train.csv")
        test_path = os.path.join("data", "processed", "test.csv")
        out_dir = os.path.join("data", "features")
        
        if not os.path.exists(train_path):
            logging.error(f"Processed data not found at {train_path}. Please run data_preprocessing.py first.")
            return
            
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        
        max_features = params['feature_engineering']['max_features']
            
        build_features(train_path, test_path, out_dir, max_features)
        logging.info("Feature engineering script completed successfully.")
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
