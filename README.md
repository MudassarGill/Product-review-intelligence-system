# Product Review Intelligence System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![DVC](https://img.shields.io/badge/DVC-Pipeline-orange)
![DagsHub](https://img.shields.io/badge/DagsHub-Integration-purple)

An end-to-end Natural Language Processing (NLP) system that analyzes product reviews to extract sentiment, detect key product features, and identify common customer issues using Machine Learning.

## 📌 Overview

This project implements a complete MLOps lifecycle from data ingestion to model deployment. It leverages a Logistic Regression model with a `CountVectorizer` to classify the sentiment of product reviews. The system is designed to be fully reproducible and trackable.

### Key Technologies
- **Model:** Logistic Regression & CountVectorizer (scikit-learn)
- **NLP Processing:** NLTK (Stopwords, Lemmatization)
- **Pipeline Orchestration:** DVC (Data Version Control)
- **Experiment Tracking & Model Registry:** MLflow & DagsHub
- **API & Web Interface:** FastAPI, Uvicorn, Jinja2 Templates

---

## 🏗️ Project Architecture & Pipeline

The machine learning pipeline is orchestrated using **DVC**. The pipeline steps are defined in `dvc.yaml`:

1.  **`data_ingestion`**: Reads raw data and splits into train/test sets (`src/data/data_ingestion.py`).
2.  **`data_preprocessing`**: Cleans text, removes HTML tags, punctuation, numbers, and applies lemmatization (`src/data/data_preprocessing.py`).
3.  **`feature_engineering`**: Transforms raw text into token counts using `CountVectorizer` (`src/features/feature_engineering.py`).
4.  **`model_building`**: Trains a Logistic Regression model on the processed features (`scripts/model_building.py`).
5.  **`model_evaluation`**: Evaluates the model on test data and logs metrics (`scripts/model_evaluation.py`).
6.  **`model_registry`**: Registers the trained model into the MLflow model registry via DagsHub (`scripts/model_regestry.py`).
7.  **`model_promotion`**: Promotes the best model to the "Production" stage based on performance metrics (`scripts/model_promotion.py`).

Hyperparameters and pipeline configurations are managed in `params.yaml`.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/mudassarhussain6533/Product-review-intelligence-system.git
cd Product-review-intelligence-system
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv myvenv
# On Windows:
myvenv\Scripts\activate
# On macOS/Linux:
source myvenv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables (DagsHub/MLflow Integration)
Create a `.env` file in the root directory (or ensure the correct environment variables are set) for remote tracking:
```env
DAGSHUB_REPO_OWNER=your_dagshub_username
DAGSHUB_REPO_NAME=Product-review-intelligence-system
CAPSTONE_TEST=your_dagshub_access_token
```

---

## 🧠 Running the ML Pipeline

To execute the entire end-to-end pipeline (Ingestion ➡️ Deployment):

```bash
dvc repro
```

This will run all stages defined in `dvc.yaml`, automatically skipping stages with unchanged dependencies.

---

## 🌐 Running the Application (FastAPI)

The project includes a web application to serve real-time predictions. 

Start the server:
```bash
python app/main.py
# Or directly using Uvicorn:
# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the API
- **Web Interface (UI)**: [http://localhost:8000/](http://localhost:8000/)
- **API Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Example API Request
You can easily test the prediction endpoint via cURL or Python `requests`:

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is absolutely amazing! Highly recommend it."}'
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.95
}
```

---

## 📂 Project Structure

```text
├── .github/workflows/   # CI/CD pipelines
├── app/                 # FastAPI application
│   ├── main.py          # API and app logic
│   └── templates/       # HTML frontend templates
├── data/                # Data storage (Raw, Processed, Features - tracked by DVC)
├── models/              # Saved Models (e.g. vectorizer, logistic regression)
├── notebooks/           # Jupyter notebooks for experimentation
├── reports/             # Evaluation metrics and registry logs
├── scripts/             # ML execution scripts (train, evaluate, promote)
├── src/                 # Source code (ingestion, preprocessing, features)
├── README.md            # Project documentation
├── dvc.yaml             # DVC pipeline configuration
├── params.yaml          # Pipeline parameters
└── requirements.txt     # Python dependencies
```

## 📜 License

This project is licensed under the [LICENSE](LICENSE) file located in the root directory.
