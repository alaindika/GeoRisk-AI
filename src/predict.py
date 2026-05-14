import json
import joblib
import pandas as pd

from src.config import MODEL_PATH, FEATURE_COLUMNS_PATH, LABEL_MAP


def load_model():
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_columns = json.load(f)

    return model, feature_columns


def predict_risk(input_data: dict):
    model, feature_columns = load_model()

    X_new = pd.DataFrame([input_data])[feature_columns]
    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0]

    return {
        "risk_label": int(prediction),
        "risk_category": LABEL_MAP[int(prediction)],
        "probabilities": {
            LABEL_MAP[i]: float(probabilities[i])
            for i in range(len(probabilities))
        },
    }


if __name__ == "__main__":
    sample = {
        "rainfall_48hr": 85,
        "pressure_drop_3hr": -12,
        "temp_anomaly": 2,
        "wind_speed": 95,
        "humidity": 80,
        "snowfall_rate": 0,
        "season": 2,
    }

    result = predict_risk(sample)
    print(result)