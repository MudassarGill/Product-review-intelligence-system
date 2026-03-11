import joblib
import os
import numpy as np
import pytest

def test_model_loading():
    model_path = "models/logistic_regression.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert model is not None
        assert hasattr(model, 'predict')

def test_vectorizer_loading():
    vec_path = "models/count_vectorizer.pkl"
    if os.path.exists(vec_path):
        vec = joblib.load(vec_path)
        assert vec is not None
        assert hasattr(vec, 'transform')

def test_prediction_shape():
    model_path = "models/logistic_regression.pkl"
    vec_path = "models/count_vectorizer.pkl"
    
    if os.path.exists(model_path) and os.path.exists(vec_path):
        model = joblib.load(model_path)
        vec = joblib.load(vec_path)
        
        sample_text = ["This product is great!"]
        X = vec.transform(sample_text)
        prediction = model.predict(X)
        
        assert len(prediction) == 1
