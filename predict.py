from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.pkl"
METADATA_PATH = ROOT_DIR / "models" / "model_metrics.json"
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
DEFAULT_CUSTOMER: dict[str, Any] = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}


@lru_cache(maxsize=1)
def load_artifacts(
    model_path: str = str(MODEL_PATH),
    preprocessor_path: str = str(PREPROCESSOR_PATH),
) -> tuple[Any, Any]:
    if not Path(model_path).exists() or not Path(preprocessor_path).exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run `python train_model.py` before launching the app."
        )
    return joblib.load(model_path), joblib.load(preprocessor_path)


@lru_cache(maxsize=1)
def load_metadata(path: str = str(METADATA_PATH)) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(
            "Model metadata is missing. Run `python train_model.py` before launching the app."
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict_churn(customer_data: dict[str, Any]) -> dict[str, Any]:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in customer_data]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Customer data is missing required fields: {missing_text}")
    model, preprocessor = load_artifacts()
    customer_frame = pd.DataFrame([{column: customer_data[column] for column in FEATURE_COLUMNS}])
    numeric_columns = customer_frame.select_dtypes(include=["number"]).columns
    if not numeric_columns.empty:
        customer_frame[numeric_columns] = customer_frame[numeric_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
    processed_customer = preprocessor.transform(customer_frame)
    prediction = int(model.predict(processed_customer)[0])
    churn_probability = float(model.predict_proba(processed_customer)[0][1])
    if churn_probability < 0.30:
        risk_level = "Low risk"
    elif churn_probability < 0.60:
        risk_level = "Medium risk"
    else:
        risk_level = "High risk"
    return {
        "prediction": prediction,
        "outcome": "Likely to churn" if prediction == 1 else "Likely to stay",
        "churn_probability": churn_probability,
        "retention_probability": 1 - churn_probability,
        "risk_level": risk_level,
    }


if __name__ == "__main__":
    prediction_result = predict_churn(DEFAULT_CUSTOMER)
    print(f"Outcome: {prediction_result['outcome']}")
    print(f"Churn probability: {prediction_result['churn_probability']:.2%}")
    print(f"Risk level: {prediction_result['risk_level']}")
