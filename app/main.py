import os
import sys
import nltk
import string
import re
import joblib
import mlflow
import dagshub
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Global models
vectorizer = None
model = None

# Initialize templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Define request body schema
class ReviewRequest(BaseModel):
    text: str



# Ensure NLTK resources are available
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: NLTK downloading failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup. Fetches from MLflow Registry with local fallback."""
    global vectorizer, model
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        vectorizer_path = os.path.join(project_root, 'models', 'count_vectorizer.pkl')
        local_model_path = os.path.join(project_root, 'models', 'logistic_regression.pkl')
        
        # 1. Load the local Vectorizer (Essential)
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            print("Vectorizer loaded successfully from local file.")
        else:
            print("CRITICAL: Vectorizer not found locally at models/count_vectorizer.pkl")
            
        # 2. Attempt to load Model from MLflow Registry
        try:
            print("Connecting to DagsHub MLflow Server...")
            # Load environment variables
            load_dotenv()
            repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "mudassarhussain6533")
            repo_name = os.getenv("DAGSHUB_REPO_NAME", "Product-review-intelligence-system")
            dagshub_token = os.getenv("CAPSTONE_TEST")

            if dagshub_token:
                # Explicitly set MLflow credentials using the token
                os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
                os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
                dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            else:
                print("Warning: CAPSTONE_TEST not found. Model registry access may fail.")
                dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            
            model_name = "Product-Review-Sentiment-Model"
            stage = "Production"
            
            print(f"Attempting to fetch '{model_name}' at stage '{stage}' from Registry...")
            model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
            print("Model loaded successfully from MLflow Remote Registry (Production).")
        except Exception as remote_err:
            print(f"Remote Registry load failed: {remote_err}")
            print("Falling back to local model loading...")
            
            # 3. Fallback to local .pkl file
            if os.path.exists(local_model_path):
                # We load it using joblib. Since we need it to behave like a pyfunc for our predict logic,
                # we'll keep it as the raw model. The predict logic handles both.
                model = joblib.load(local_model_path)
                print(f"Model loaded successfully from local file: {local_model_path}")
            else:
                print("CRITICAL ERROR: No local model found at models/logistic_regression.pkl. Prediction will fail.")
        
    except Exception as e:
        print(f"Lifespan error: {e}")
        
    yield  # Hand over control to the FastAPI application
    print("Shutting down application...")

# Initialize app with lifespan
app = FastAPI(title="Product Review Intelligence API", lifespan=lifespan)

def preprocess_input(text: str) -> str:
    """Apply the exact same preprocessing logic from the training pipeline."""
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the frontend HTML dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_sentiment(request: ReviewRequest):
    """API Endpoint to predict sentiment of a single review."""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Models are not loaded. Server is unavailable.")
        
    try:
        # Preprocess text
        cleaned_text = preprocess_input(request.text)
        if not cleaned_text:
            return {"sentiment": "Neutral", "confidence": 0.0}
            
        # Transform using CountVectorizer
        X_infer = vectorizer.transform([cleaned_text]).toarray()
        
        # Predict
        prediction = model.predict(X_infer)[0]
        
        # Get probabilities (if available on the model, LogReg does)
        proba = model.predict_proba(X_infer)[0]
        confidence = float(max(proba))
        
        return {
            "sentiment": prediction,
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Product Review Intelligence App...")
    # Add the project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    # We run 'main:app' and specify the 'app' directory as the search path
    # This is more robust for reload on Windows.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir="app")
