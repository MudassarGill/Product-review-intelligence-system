import pandas as pd
import numpy as np 
import setuptools
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import re  # Added missing import

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # Added CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import scipy.sparse

import warnings
warnings.filterwarnings("ignore")

#=====================CONFIGRATION=====================

CONFIG = {
    "data_path": r"C:\Users\LAPTOPS HUB\Desktop\Product-review-intelligence-system\notebooks\Reviews.csv",
    "test_size": 0.2,
    "random_state": 42,
    "mlflow_tracking_uri": "",
    "dagshub_username": "",
    "dagshub_repo": "Product-review-intelligence-system",
    "Experiment_name": "TF-IDF"
}

#==================MLFLOW setup and DASGHUB========================

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
mlflow.set_experiment(CONFIG["Experiment_name"])
dagshub.init(
    repo_owner=CONFIG["dagshub_username"],
    repo_name=CONFIG["dagshub_repo"],
    mlflow=True
)

print("MLFLOW setup and DAGSHUB initialization complete")

#==================TEXT PREPROCESSING========================

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))  # Fixed variable name
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_number(text):
    return text.translate(str.maketrans('', '', string.digits))

def lower_case(text):
    return text.lower()

def removing_url(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_digits(text):
    return text.translate(str.maketrans('', '', string.digits))

def remove_whitespace(text):
    return " ".join(text.split())

def normalize_text(df):
    try:
        df = df.copy()  # Added to avoid warnings
        df['Text'] = df['Text'].apply(lower_case)
        df['Text'] = df['Text'].apply(removing_url)
        df['Text'] = df['Text'].apply(remove_punctuation)
        df['Text'] = df['Text'].apply(remove_digits)
        df['Text'] = df['Text'].apply(remove_whitespace)
        df['Text'] = df['Text'].apply(lemmatization)
        df['Text'] = df['Text'].apply(remove_stopwords)
        return df
    except Exception as e:
        raise e

#============================LOAD and PREPROCASS DATA========================

def load_and_preprocess_data():  # Fixed function name
    try:
        df = pd.read_csv(CONFIG["data_path"])
        print("Data loaded successfully")
        print(df.head())
        df = normalize_text(df)
        df = df[df['Summary'].isin(['positive', 'negative'])]
        df['Summary'] = df['Summary'].replace({'positive': 1, 'negative': 0})
        print("Data preprocessed successfully")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e

#========================FEATURE ENGINEERING========================

VECTORIZER = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {  # Fixed spelling from ALGROTHIM to ALGORITHMS
    'MultinomialNB': MultinomialNB(),
    'LogisticRegression': LogisticRegression(),
    'XGBClassifier': XGBClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

#=====================================TRAINING AND EVALUATION=====================================

def train_and_evaluate(df):
    with mlflow.start_run(run_name="All experiments") as parent_run:
        for algo_name, algorithm in ALGORITHMS.items():  # Fixed variable name
            for vectorizer_name, vectorizer in VECTORIZER.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vectorizer_name}", nested=True) as child_run:
                    try: 
                        #feature extraction
                        X = vectorizer.fit_transform(df['Text'])
                        y = df['Summary']
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, 
                            test_size=CONFIG["test_size"], 
                            random_state=CONFIG["random_state"]
                        )
                        #logs prepocessing paramerts
                        mlflow.log_params({  # Fixed from log_param to log_params
                            'vectorizer': vectorizer_name,
                            'algorithm': algo_name,
                            'test_size': CONFIG["test_size"],
                            'random_state': CONFIG["random_state"]
                        })
                        #train model
                        model = algorithm
                        model.fit(X_train, y_train)
                        log_model_params(algo_name, model)  # Fixed function name
                        y_pred = model.predict(X_test)
                        metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='binary'),
                            'recall': recall_score(y_test, y_pred, average='binary'),
                            'f1': f1_score(y_test, y_pred, average='binary')
                        }
                        mlflow.log_metrics(metrics)
                        if scipy.sparse.issparse(X_test):
                            input_example = X_test[:5].toarray()
                        else:
                            input_example = X_test[:5]
                        mlflow.sklearn.log_model(model=model, artifact_path="model", input_example=input_example)
                        print(f"Vectorizer: {vectorizer_name}, Algorithm: {algo_name}, Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
                    except Exception as e:
                        print(f"Error in {algo_name} with {vectorizer_name}: {e}")
                        mlflow.log_param("error", str(e))
                        continue

def log_model_params(algo_name, model):  # Fixed function name
    """ Logs hyper-parameters of the trained model to MLflow """
    param_to_log = {}  # Fixed from list to dict
    
    if algo_name == 'LogisticRegression':
        param_to_log['C'] = model.C
    elif algo_name == 'XGBClassifier':
        param_to_log['n_estimators'] = model.n_estimators
        param_to_log['learning_rate'] = model.learning_rate
        param_to_log['max_depth'] = model.max_depth
    elif algo_name == 'RandomForestClassifier':
        param_to_log['n_estimators'] = model.n_estimators
        param_to_log['max_depth'] = model.max_depth
        param_to_log['min_samples_split'] = model.min_samples_split
        param_to_log['min_samples_leaf'] = model.min_samples_leaf
    elif algo_name == 'GradientBoostingClassifier':
        param_to_log['n_estimators'] = model.n_estimators
        param_to_log['learning_rate'] = model.learning_rate
        param_to_log['max_depth'] = model.max_depth
    
    if param_to_log:  # Only log if there are parameters
        mlflow.log_params(param_to_log)

#============================MAIN============================

if __name__ == "__main__":  # Fixed syntax
    df = load_and_preprocess_data()  # Fixed function name
    train_and_evaluate(df)