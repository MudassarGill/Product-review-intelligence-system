import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import yaml
from src.logger import logging

import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from src.data import data_ingestion



def preprocess_text(text):
    """
    Clean the text by lowercasing, removing HTML tags, punctuation, numbers, and stopwords.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercasing
    text = text.lower()
    
    # Removing HTML tags
    text = re.sub('<.*?>', '', text)
    
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Removing numbers
    text = re.sub(r'\d+', '', text)
    
    # Removing stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    try:
        words = [lemmatizer.lemmatize(w) for w in words]
    except Exception:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

def preprocess_data(input_path: str, output_path: str):
    """
    Load data from input_path, apply preprocessing, and save to output_path.
    """
    try:
        logging.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # We assume the data has 'Sentiment_Summary' or 'Text'
        # Looking at ingestion, it keeps all columns. In exp.ipynb it used 'Text'
        target_column = 'Text' if 'Text' in df.columns else 'summary'
        
        logging.info(f"Preprocessing column: {target_column}")
        df['Cleaned_Text'] = df[target_column].apply(preprocess_text)
        
        # Remove rows where Cleaned_Text is empty
        df = df[df['Cleaned_Text'] != ""]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Preprocessed data saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error during preprocessing of {input_path}: {e}")
        raise

def main():
    try:
        raw_train_path = os.path.join("data", "raw", "train.csv")
        raw_test_path = os.path.join("data", "raw", "test.csv")
        
        processed_train_path = os.path.join("data", "processed", "train.csv")
        processed_test_path = os.path.join("data", "processed", "test.csv")
        
        if not os.path.exists(raw_train_path):
            logging.error(f"Raw train data not found at {raw_train_path}. Please run data_ingestion first.")
            return

        preprocess_data(raw_train_path, processed_train_path)
        preprocess_data(raw_test_path, processed_test_path)
        
        logging.info("Data preprocessing script completed successfully.")

    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")

if __name__ == "__main__":
    main()