import os
import joblib


def test_project_structure():
    assert os.path.exists("app/streamlit_app.py")
    assert os.path.exists("src/preprocessing.py")
    assert os.path.exists("src/feature_engineering.py")
    assert os.path.exists("src/predict.py")
    assert os.path.exists("src/risk_interpretation.py")


def test_models_exist():
    assert os.path.exists("models/nephropathy_model.joblib")
    assert os.path.exists("models/angiopathy_model.joblib")


def test_load_nephropathy_model():
    model = joblib.load(
        "models/nephropathy_model.joblib"
    )
    assert model is not None


def test_load_angiopathy_model():
    model = joblib.load(
        "models/angiopathy_model.joblib"
    )
    assert model is not None

