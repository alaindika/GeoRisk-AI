import joblib
import pandas as pd
import shap

from src.config import MODEL_PATH, FEATURE_COLUMNS
from src.predict import predict_risk


def load_trained_model():
    return joblib.load(MODEL_PATH)


def get_shap_explanation(input_data: dict):
    """
    Generate local SHAP explanations for one prediction.
    """
    model = load_trained_model()
    X = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    prediction = predict_risk(input_data)
    predicted_label = prediction["risk_label"]

    # For multiclass models, SHAP returns one array per class.
    class_shap_values = shap_values[:, :, predicted_label][0]

    explanation_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Input Value": X.iloc[0].values,
        "SHAP Value": class_shap_values,
        "Absolute Impact": abs(class_shap_values),
    })

    explanation_df = explanation_df.sort_values(
        by="Absolute Impact",
        ascending=False
    )

    return prediction, explanation_df


def generate_plain_language_explanation(input_data: dict):
    prediction, explanation_df = get_shap_explanation(input_data)

    top_features = explanation_df.head(3)["Feature"].tolist()
    risk = prediction["risk_category"]

    explanation = (
        f"The model predicted {risk} risk mainly because of "
        f"{top_features[0]}, {top_features[1]}, and {top_features[2]}."
    )

    return {
        "prediction": prediction,
        "top_contributors": explanation_df.head(3).to_dict(orient="records"),
        "full_explanation": explanation_df.to_dict(orient="records"),
        "plain_language_explanation": explanation,
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

    result = generate_plain_language_explanation(sample)

    print(result["prediction"])
    print("\nTop SHAP contributors:")
    for item in result["top_contributors"]:
        print(item)

    print("\nExplanation:")
    print(result["plain_language_explanation"])