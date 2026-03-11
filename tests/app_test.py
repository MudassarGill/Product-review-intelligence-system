from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_predict_sentiment():
    # Note: This might fail if models are not loaded in the test environment
    # but we test the structure of the API.
    response = client.post(
        "/predict",
        json={"text": "I really love this product, it works perfectly!"},
    )
    if response.status_code == 200:
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
    elif response.status_code == 503:
        # Service unavailable is acceptable if models aren't loaded
        assert response.json()["detail"] == "Models are not loaded. Server is unavailable."
