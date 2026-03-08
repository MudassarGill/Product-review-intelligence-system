import pandas as pd
import numpy as np 
import setuptools
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import re
import os
import joblib

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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
    "Experiment_name": "Experiment-3-GridSearchCV",
    "cv_folds": 3,  # Reduced from 5 to 3 for small dataset
    "scoring": "f1",
    "min_samples_per_class": 2  # Minimum samples required per class
}

#==================MLFLOW setup and DASGHUB========================

# First initialize DAGsHub
dagshub.init(
    repo_owner=CONFIG["dagshub_username"],
    repo_name=CONFIG["dagshub_repo"],
    mlflow=True
)

# Set MLflow tracking URI
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])

# Try to set experiment, if it fails, run locally
try:
    mlflow.set_experiment(CONFIG["Experiment_name"])
    print(f"MLFLOW experiment '{CONFIG['Experiment_name']}' set successfully")
except Exception as e:
    print(f"Warning: Could not set MLflow experiment remotely: {e}")
    print("Running in local mode...")
    # Set local tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(CONFIG["Experiment_name"])

print("MLFLOW setup and DAGSHUB initialization complete")

#==================TEXT PREPROCESSING========================

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
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
        df = df.copy()
        df['Text'] = df['Text'].astype(str).apply(lower_case)
        df['Text'] = df['Text'].apply(removing_url)
        df['Text'] = df['Text'].apply(remove_punctuation)
        df['Text'] = df['Text'].apply(remove_digits)
        df['Text'] = df['Text'].apply(remove_whitespace)
        df['Text'] = df['Text'].apply(lemmatization)
        df['Text'] = df['Text'].apply(remove_stopwords)
        return df
    except Exception as e:
        print(f"Error in normalize_text: {e}")
        raise e

#============================LOAD and PREPROCESS DATA========================

def load_and_preprocess_data():
    try:
        # Check if file exists
        if not os.path.exists(CONFIG["data_path"]):
            # Try alternative path
            alt_path = "../" + CONFIG["data_path"]
            if os.path.exists(alt_path):
                CONFIG["data_path"] = alt_path
            else:
                print(f"Warning: Data file not found at {CONFIG['data_path']}")
                # Create larger sample data for testing
                print("Creating larger sample data for testing...")
                sample_data = pd.DataFrame({
                    'Text': [
                        'This product is amazing! I love it.',
                        'Very poor quality, waste of money.',
                        'Good product, works as expected.',
                        'Terrible customer service, would not recommend.',
                        'Excellent value for money, highly recommended!',
                        'Not worth the price, disappointed.',
                        'Best purchase ever!',
                        'Really bad experience with this product.',
                        'Average product, nothing special.',
                        'Highly recommend this to everyone!',
                        'Works great, very satisfied.',
                        'Complete waste of money, avoid.',
                        'Perfect product, exactly what I needed.',
                        'Disappointing quality, broke quickly.',
                        'Great value for the price.',
                        'Poor performance, would not buy again.',
                        'Excellent quality, very durable.',
                        'Not as described, very disappointed.',
                        'Love this product, works perfectly.',
                        'Terrible quality, returned immediately.'
                    ],
                    'Summary': [
                        'positive', 'negative', 'positive', 'negative', 'positive', 
                        'negative', 'positive', 'negative', 'positive', 'positive',
                        'positive', 'negative', 'positive', 'negative', 'positive',
                        'negative', 'positive', 'negative', 'positive', 'negative'
                    ]
                })
                sample_data.to_csv(CONFIG["data_path"], index=False)
                print(f"Sample data created at {CONFIG['data_path']} with {len(sample_data)} rows")
        
        df = pd.read_csv(CONFIG["data_path"])
        print("Data loaded successfully")
        print(f"Data shape: {df.shape}")
        print("First 5 rows:")
        print(df.head())
        
        df = normalize_text(df)
        
        # Handle different column name cases
        if 'Summary' not in df.columns and 'summary' in df.columns:
            df = df.rename(columns={'summary': 'Summary'})
        
        df['Summary'] = df['Summary'].astype(str).str.lower()
        df = df[df['Summary'].isin(['positive', 'negative'])]
        df['Summary'] = df['Summary'].map({'positive': 1, 'negative': 0})
        
        # Check class distribution
        class_counts = df['Summary'].value_counts()
        print("Class distribution after preprocessing:")
        print(class_counts)
        
        # Check if we have enough samples for cross-validation
        min_class_size = class_counts.min()
        if min_class_size < CONFIG["min_samples_per_class"]:
            print(f"Warning: Smallest class has only {min_class_size} samples. Adjusting CV folds...")
            CONFIG["cv_folds"] = min(3, min_class_size)  # Adjust CV folds based on smallest class
            print(f"CV folds adjusted to: {CONFIG['cv_folds']}")
        
        print("Data preprocessed successfully")
        print("First 5 rows after preprocessing:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e

#========================HYPERPARAMETER GRIDS========================

# Simplified hyperparameter grids for small dataset
PARAM_GRIDS = {
    'MultinomialNB': {
        'alpha': [0.1, 0.5, 1.0],
        'fit_prior': [True, False]
    },
    
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [1000]
    },
    
    'XGBClassifier': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.3],
        'subsample': [1.0],
        'colsample_bytree': [1.0]
    },
    
    'RandomForestClassifier': {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    
    'GradientBoostingClassifier': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.3],
        'subsample': [1.0]
    }
}

# Base algorithms with default parameters
BASE_ALGORITHMS = {
    'MultinomialNB': MultinomialNB(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=CONFIG["random_state"]),
    'RandomForestClassifier': RandomForestClassifier(random_state=CONFIG["random_state"]),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=CONFIG["random_state"])
}

VECTORIZER = {
    'BoW': CountVectorizer(max_features=1000),  # Limit features for small dataset
    'TF-IDF': TfidfVectorizer(max_features=1000)
}

#=====================================TRAINING WITH GRIDSEARCHCV=====================================

def train_with_gridsearchcv(df):
    """Train models using GridSearchCV and track with MLflow"""
    
    best_models = {}  # Store best models for each combination
    
    with mlflow.start_run(run_name="All Experiments with GridSearchCV") as parent_run:
        
        for algo_name, algorithm in BASE_ALGORITHMS.items():
            for vectorizer_name, vectorizer in VECTORIZER.items():
                
                # Create a nested run for each algorithm-vectorizer combination
                with mlflow.start_run(run_name=f"GridSearch_{algo_name}_{vectorizer_name}", nested=True) as child_run:
                    
                    try: 
                        print(f"\n{'='*70}")
                        print(f"GridSearchCV for {algo_name} with {vectorizer_name}")
                        print(f"{'='*70}")
                        
                        # Feature extraction
                        print("Extracting features...")
                        X = vectorizer.fit_transform(df['Text'])
                        y = df['Summary'].values
                        
                        # Check if we have enough samples for stratification
                        class_counts = pd.Series(y).value_counts()
                        min_class_count = class_counts.min()
                        
                        # Adjust test_size based on smallest class
                        if min_class_count < 2:
                            print(f"Warning: Smallest class has {min_class_count} samples. Cannot use stratification.")
                            # Split without stratification
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, 
                                test_size=CONFIG["test_size"], 
                                random_state=CONFIG["random_state"],
                                stratify=None  # No stratification
                            )
                        else:
                            # Split with stratification
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, 
                                test_size=CONFIG["test_size"], 
                                random_state=CONFIG["random_state"],
                                stratify=y
                            )
                        
                        # Log preprocessing parameters
                        mlflow.log_params({
                            'vectorizer': vectorizer_name,
                            'algorithm': algo_name,
                            'test_size': CONFIG["test_size"],
                            'random_state': CONFIG["random_state"],
                            'cv_folds': CONFIG["cv_folds"],
                            'scoring_metric': CONFIG["scoring"],
                            'total_samples': len(df),
                            'features_count': X.shape[1],
                            'train_samples': len(y_train),
                            'test_samples': len(y_test),
                            'positive_samples': sum(y == 1),
                            'negative_samples': sum(y == 0)
                        })
                        
                        # Setup GridSearchCV
                        print(f"Setting up GridSearchCV with {CONFIG['cv_folds']}-fold CV...")
                        print(f"Parameter grid: {PARAM_GRIDS[algo_name]}")
                        
                        # Adjust CV strategy based on class distribution
                        if min_class_count >= CONFIG["cv_folds"]:
                            cv_strategy = StratifiedKFold(
                                n_splits=CONFIG["cv_folds"], 
                                shuffle=True, 
                                random_state=CONFIG["random_state"]
                            )
                        else:
                            # Use regular KFold if classes are too small for stratification
                            from sklearn.model_selection import KFold
                            cv_strategy = KFold(
                                n_splits=CONFIG["cv_folds"], 
                                shuffle=True, 
                                random_state=CONFIG["random_state"]
                            )
                            print(f"Using regular KFold instead of StratifiedKFold due to small class sizes")
                        
                        grid_search = GridSearchCV(
                            estimator=algorithm,
                            param_grid=PARAM_GRIDS[algo_name],
                            cv=cv_strategy,
                            scoring=CONFIG["scoring"],
                            n_jobs=-1,
                            verbose=1,
                            return_train_score=True
                        )
                        
                        # Train with GridSearchCV
                        print("Training with GridSearchCV...")
                        grid_search.fit(X_train, y_train)
                        
                        # Get best model
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        best_score = grid_search.best_score_
                        
                        print(f"\n✓ Best parameters: {best_params}")
                        print(f"✓ Best cross-validation {CONFIG['scoring']}: {best_score:.4f}")
                        
                        # Log best parameters
                        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                        mlflow.log_metric("best_cv_score", best_score)
                        
                        # Evaluate on test set
                        y_pred = best_model.predict(X_test)
                        
                        # Calculate metrics
                        metrics = {
                            'test_accuracy': accuracy_score(y_test, y_pred),
                            'test_precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                            'test_recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                            'test_f1': f1_score(y_test, y_pred, average='binary', zero_division=0)
                        }
                        
                        # Log test metrics
                        mlflow.log_metrics(metrics)
                        
                        # Log confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        cm_df = pd.DataFrame(
                            cm, 
                            index=['Actual Negative', 'Actual Positive'], 
                            columns=['Predicted Negative', 'Predicted Positive']
                        )
                        cm_path = f"confusion_matrix_{algo_name}_{vectorizer_name}.csv"
                        cm_df.to_csv(cm_path)
                        mlflow.log_artifact(cm_path)
                        
                        # Log model
                        if scipy.sparse.issparse(X_test):
                            input_example = X_test[:5].toarray()
                        else:
                            input_example = X_test[:5]
                        
                        mlflow.sklearn.log_model(
                            sk_model=best_model, 
                            artifact_path=f"model_{algo_name}_{vectorizer_name}", 
                            input_example=input_example
                        )
                        
                        # Save best model info
                        best_models[f"{algo_name}_{vectorizer_name}"] = {
                            'model': best_model,
                            'best_params': best_params,
                            'best_cv_score': best_score,
                            'test_metrics': metrics,
                            'vectorizer': vectorizer
                        }
                        
                        print(f"\n✓ Test Set Performance:")
                        print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
                        print(f"  Precision: {metrics['test_precision']:.4f}")
                        print(f"  Recall: {metrics['test_recall']:.4f}")
                        print(f"  F1-Score: {metrics['test_f1']:.4f}")
                        
                    except Exception as e:
                        print(f"✗ Error in GridSearchCV for {algo_name} with {vectorizer_name}: {e}")
                        mlflow.log_param("error", str(e))
                        import traceback
                        traceback.print_exc()
                        continue
        
        # Find and log the overall best model
        if best_models:
            best_model_name = max(best_models, key=lambda x: best_models[x]['test_metrics']['test_f1'])
            best_model_info = best_models[best_model_name]
            
            print(f"\n{'='*70}")
            print(f"🎯 OVERALL BEST MODEL: {best_model_name}")
            print(f"{'='*70}")
            print(f"Best Parameters: {best_model_info['best_params']}")
            print(f"Best CV F1-Score: {best_model_info['best_cv_score']:.4f}")
            print(f"Test F1-Score: {best_model_info['test_metrics']['test_f1']:.4f}")
            print(f"Test Accuracy: {best_model_info['test_metrics']['test_accuracy']:.4f}")
            
            # Log overall best model info in parent run
            mlflow.log_params({
                "overall_best_model": best_model_name,
                "overall_best_params": str(best_model_info['best_params']),
                "overall_best_cv_score": best_model_info['best_cv_score'],
                "overall_best_test_f1": best_model_info['test_metrics']['test_f1']
            })
            
            # Save the best model and vectorizer
            joblib.dump(best_model_info['model'], "best_model.pkl")
            joblib.dump(best_model_info['vectorizer'], "best_vectorizer.pkl")
            mlflow.log_artifact("best_model.pkl")
            mlflow.log_artifact("best_vectorizer.pkl")
            
            return best_models

#=====================================ALTERNATIVE: RUN WITHOUT MLFLOW=====================================

def run_without_mlflow(df):
    """Run GridSearchCV without MLflow tracking (fallback)"""
    
    best_models = {}
    
    for algo_name, algorithm in BASE_ALGORITHMS.items():
        for vectorizer_name, vectorizer in VECTORIZER.items():
            try:
                print(f"\n{'='*70}")
                print(f"GridSearchCV for {algo_name} with {vectorizer_name}")
                print(f"{'='*70}")
                
                # Feature extraction
                print("Extracting features...")
                X = vectorizer.fit_transform(df['Text'])
                y = df['Summary'].values
                
                # Check class distribution
                class_counts = pd.Series(y).value_counts()
                min_class_count = class_counts.min()
                
                # Split data
                if min_class_count < 2:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=CONFIG["test_size"], 
                        random_state=CONFIG["random_state"],
                        stratify=None
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=CONFIG["test_size"], 
                        random_state=CONFIG["random_state"],
                        stratify=y
                    )
                
                # Setup GridSearchCV
                print(f"Setting up GridSearchCV with {CONFIG['cv_folds']}-fold CV...")
                
                # Adjust CV strategy
                if min_class_count >= CONFIG["cv_folds"]:
                    cv_strategy = StratifiedKFold(
                        n_splits=CONFIG["cv_folds"], 
                        shuffle=True, 
                        random_state=CONFIG["random_state"]
                    )
                else:
                    from sklearn.model_selection import KFold
                    cv_strategy = KFold(
                        n_splits=CONFIG["cv_folds"], 
                        shuffle=True, 
                        random_state=CONFIG["random_state"]
                    )
                
                grid_search = GridSearchCV(
                    estimator=algorithm,
                    param_grid=PARAM_GRIDS[algo_name],
                    cv=cv_strategy,
                    scoring=CONFIG["scoring"],
                    n_jobs=-1,
                    verbose=1,
                    return_train_score=True
                )
                
                # Train with GridSearchCV
                print("Training with GridSearchCV...")
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                
                print(f"\n✓ Best parameters: {best_params}")
                print(f"✓ Best cross-validation {CONFIG['scoring']}: {best_score:.4f}")
                
                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
                
                print(f"\n✓ Test Set Performance:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                
                best_models[f"{algo_name}_{vectorizer_name}"] = {
                    'model': best_model,
                    'best_params': best_params,
                    'best_cv_score': best_score,
                    'test_metrics': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                }
                
            except Exception as e:
                print(f"✗ Error in {algo_name} with {vectorizer_name}: {e}")
                continue
    
    # Find overall best model
    if best_models:
        best_model_name = max(best_models, key=lambda x: best_models[x]['test_metrics']['f1'])
        best_model_info = best_models[best_model_name]
        
        print(f"\n{'='*70}")
        print(f"🎯 OVERALL BEST MODEL: {best_model_name}")
        print(f"{'='*70}")
        print(f"Best Parameters: {best_model_info['best_params']}")
        print(f"Best CV F1-Score: {best_model_info['best_cv_score']:.4f}")
        print(f"Test F1-Score: {best_model_info['test_metrics']['f1']:.4f}")
    
    return best_models

#============================MAIN============================

if __name__ == "__main__":
    print("="*70)
    print("PRODUCT REVIEW INTELLIGENCE SYSTEM - EXPERIMENT 3")
    print("GridSearchCV for Hyperparameter Tuning")
    print("="*70)
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        print(f"\n✅ Final dataset shape: {df.shape}")
        print(f"✅ Classes: {df['Summary'].unique()}")
        print(f"✅ Class distribution:\n{df['Summary'].value_counts()}")
        
        # Check if we have enough data for meaningful GridSearchCV
        if len(df) < 10:
            print("\n⚠️ Warning: Very small dataset detected. Results may not be reliable.")
        
        # Try with MLflow first, fallback to without
        try:
            print("\n🚀 Starting GridSearchCV with MLflow tracking...")
            best_models = train_with_gridsearchcv(df)
        except Exception as e:
            print(f"\n⚠️ MLflow tracking failed: {e}")
            print("🚀 Running GridSearchCV without MLflow tracking...")
            best_models = run_without_mlflow(df)
        
        print("\n" + "="*70)
        print("✅ EXPERIMENT 3 COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()