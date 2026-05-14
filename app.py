import streamlit as st
import pandas as pd

from src.predict import predict_risk
from src.explain import generate_plain_language_explanation

st.set_page_config(
    page_title="GeoRisk-AI",
    page_icon="🌩️",
    layout="wide"
)

st.title("🌩️ GeoRisk-AI")
st.subheader("Explainable Severe Weather Risk Forecasting Dashboard")

st.markdown(
    """
    GeoRisk-AI predicts localized severe-weather risk from engineered meteorological signals.
    The system prioritizes detection of high-risk events while providing interpretable SHAP-based
    explanations for each prediction.
    """
)

st.divider()

left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("Weather Inputs")

    rainfall_48hr = st.slider("48-hour Rainfall (mm)", 0.0, 150.0, 20.0)
    pressure_drop_3hr = st.slider("3-hour Pressure Drop (hPa)", -20.0, 10.0, -2.0)
    temp_anomaly = st.slider("Temperature Anomaly (°C)", -15.0, 15.0, 0.0)
    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 150.0, 20.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
    snowfall_rate = st.slider("Snowfall Rate (cm/hr)", 0.0, 10.0, 0.0)

    season = st.selectbox(
        "Season",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Winter",
            1: "Spring",
            2: "Summer",
            3: "Fall",
        }[x],
    )

    input_data = {
        "rainfall_48hr": rainfall_48hr,
        "pressure_drop_3hr": pressure_drop_3hr,
        "temp_anomaly": temp_anomaly,
        "wind_speed": wind_speed,
        "humidity": humidity,
        "snowfall_rate": snowfall_rate,
        "season": season,
    }

    run_prediction = st.button("Predict Weather Risk", use_container_width=True)

with right_col:
    st.header("Scenario Summary")

    st.metric("Rainfall 48hr", f"{rainfall_48hr:.1f} mm")
    st.metric("Pressure Drop 3hr", f"{pressure_drop_3hr:.1f} hPa")
    st.metric("Wind Speed", f"{wind_speed:.1f} km/h")
    st.metric("Humidity", f"{humidity:.1f}%")

if run_prediction:
    result = predict_risk(input_data)
    explanation = generate_plain_language_explanation(input_data)

    risk = result["risk_category"]
    probabilities = result["probabilities"]

    st.divider()
    st.header("Prediction Result")

    if risk == "High":
        st.error("🚨 Predicted Risk Level: HIGH")
        st.markdown("**Action:** Conditions may justify urgent monitoring or alert escalation.")
    elif risk == "Medium":
        st.warning("⚠️ Predicted Risk Level: MEDIUM")
        st.markdown("**Action:** Conditions should be monitored closely.")
    else:
        st.success("✅ Predicted Risk Level: LOW")
        st.markdown("**Action:** No severe-weather alert is suggested by the model.")

    prob_df = pd.DataFrame(
        {
            "Risk Level": list(probabilities.keys()),
            "Probability": list(probabilities.values()),
        }
    )

    st.subheader("Risk Probabilities")
    st.bar_chart(prob_df.set_index("Risk Level"))

    st.subheader("Top SHAP Contributors")

    contrib_df = pd.DataFrame(explanation["top_contributors"])

    display_df = contrib_df[
        [
            "Feature",
            "Input Value",
            "SHAP Value",
            "Absolute Impact",
        ]
    ]

    st.dataframe(display_df, use_container_width=True)

    shap_chart_df = display_df[["Feature", "Absolute Impact"]].set_index("Feature")

    st.subheader("SHAP Feature Impact")
    st.bar_chart(shap_chart_df)

    st.subheader("Plain-Language Explanation")
    st.info(explanation["plain_language_explanation"])

    st.divider()
    st.caption(
        "Prototype note: This dashboard is a decision-support tool, not a replacement for official meteorological warnings."
    )