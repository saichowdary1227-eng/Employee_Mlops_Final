import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(APP_DIR)

if not (APP_DIR / "model.pkl").exists():
    subprocess.run([sys.executable, "train.py"], cwd=APP_DIR, check=True)

from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test if the health check endpoint returns 200 OK."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json == {"status": "healthy"}


def test_home_page(client):
    """Test if the home page loads successfully."""
    response = client.get("/")

    assert response.status_code == 200
    assert b"Employee Salary Prediction Form" in response.data


def test_predict_endpoint(client):
    """Test if the predict endpoint accepts employee data and returns HTML."""

    mock_form_data = {
        "years_experience": "5",
        "age": "30",
        "department": "IT",
        "education_level": "Master",
        "job_role": "Senior",
        "performance_score": "4.2",
    }

    response = client.post("/predict", data=mock_form_data)

    assert response.status_code == 200
    assert b"Prediction Result" in response.data
    assert b"predicted salary" in response.data

