# GeoRisk-AI

## Explainable Severe Weather Risk Forecasting Dashboard

GeoRisk-AI is a machine learning prototype that classifies localized severe-weather risk as **Low**, **Medium**, or **High** using engineered meteorological features. The system combines a trained Random Forest model, SHAP-based explainability, error analysis, and an interactive Streamlit dashboard.

---

## Project Motivation

Severe weather alerts need to be timely, localized, and interpretable. This project translates weather signals such as rainfall accumulation, pressure drop, wind speed, humidity, temperature anomaly, snowfall rate, and season into a risk-level prediction that can support early decision-making.

---

## Key Features

- Synthetic weather data generation for supervised prototype development
- Feature engineering using meteorological risk indicators
- Model comparison: Logistic Regression, Decision Tree, Random Forest
- Recall-focused evaluation for high-risk events
- SHAP-based local explanation for each prediction
- Streamlit dashboard for interactive risk forecasting
- Confusion matrix and error analysis workflow

---

## Dashboard Preview

### Home Dashboard
![Dashboard Home](images/dashboard_home.png)

### Prediction Results
![Prediction Result](images/prediction_result.png)

### SHAP Explainability
![SHAP Analysis](images/shap_analysis.png)

### Error Analysis
![Confusion Matrix](images/confusion_matrix.png)

---

## Machine Learning Task

- **Task:** Multiclass classification
- **Target:** `Low`, `Medium`, `High` weather risk
- **Primary metric:** High-risk recall
- **Main model:** Random Forest Classifier

---

## Input Features

| Feature | Description |
|---|---|
| `rainfall_48hr` | 48-hour accumulated rainfall |
| `pressure_drop_3hr` | 3-hour atmospheric pressure change |
| `temp_anomaly` | Temperature deviation from expected conditions |
| `wind_speed` | Current wind speed |
| `humidity` | Relative humidity |
| `snowfall_rate` | Snowfall accumulation rate |
| `season` | Encoded seasonal context |

---

## Current Results

| Model | Accuracy | Weighted F1 | High-Risk Recall |
|---|---:|---:|---:|
| Random Forest | 0.9000 | 0.9003 | 0.8817 |
| Decision Tree | 0.8944 | 0.8956 | 0.8495 |
| Logistic Regression | 0.8800 | 0.8799 | 0.8710 |

The error analysis shows that most confusion occurs around boundary cases between **Low**, **Medium**, and **High** risk, which is realistic for weather-risk classification.

---

## Project Structure

- `app.py` — Streamlit dashboard
- `requirements.txt` — Python dependencies
- `README.md` — project documentation
- `src/` — ML pipeline source code
- `data/` — processed and raw data folders
- `images/` — dashboard screenshots
- `artifacts/` — saved model artifacts
- `reports/` — reports and outputs
- `notebooks/` — analysis notebooks

---

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/alaindika/GeoRisk-AI.git
cd GeoRisk-AI
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Generate synthetic weather data:

```bash
python -m src.data_gen
```

5. Train the model:

```bash
python -m src.train
```

6. Run the dashboard:

```bash
streamlit run app.py
```

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- SHAP
- Streamlit
- matplotlib
- seaborn
- joblib

---

## Limitations

This prototype is trained primarily on synthetic weather data. The next major improvement is to integrate real Environment Canada weather records and validate the model against real severe-weather events.

---

## Future Improvements

- Integrate real Environment Canada weather data
- Add real-event labeling
- Deploy the dashboard online
- Add advanced SHAP waterfall plots
- Extend the system toward time-series forecasting

---

## Author

**Alain Dika**  
Applied Mathematician | AI Graduate Student | ML/AI Engineering Portfolio
