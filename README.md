\# GeoRisk-AI



\## Explainable Severe Weather Risk Forecasting Dashboard



GeoRisk-AI is a machine learning prototype that classifies localized severe-weather risk as \*\*Low\*\*, \*\*Medium\*\*, or \*\*High\*\* using engineered meteorological features. The system combines a trained Random Forest model, SHAP-based explainability, error analysis, and an interactive Streamlit dashboard.



\---



\## Project Motivation



Severe weather alerts need to be timely, localized, and interpretable. This project translates weather signals such as rainfall accumulation, pressure drop, wind speed, humidity, temperature anomaly, snowfall rate, and season into a risk-level prediction that can support early decision-making.



\---



\## Key Features



\- Synthetic weather data generation for supervised prototype development

\- Feature engineering using meteorological risk indicators

\- Model comparison: Logistic Regression, Decision Tree, Random Forest

\- Recall-focused evaluation for high-risk events

\- SHAP-based local explanation for each prediction

\- Streamlit dashboard for interactive risk forecasting

\- Confusion matrix and error analysis workflow



\---



\## Dashboard Preview



\### Home Dashboard



!\[Dashboard Home](images/dashboard\_home.png)



\### Prediction Results



!\[Prediction Result](images/prediction\_result.png)



\### SHAP Explainability



!\[SHAP Analysis](images/shap\_analysis.png)



\### Error Analysis



!\[Confusion Matrix](images/confusion\_matrix.png)



\---



\## Machine Learning Task



\- \*\*Task:\*\* Multiclass classification

\- \*\*Target:\*\* Low / Medium / High weather risk

\- \*\*Primary metric:\*\* High-risk recall

\- \*\*Main model:\*\* Random Forest Classifier



\---



\## Project Structure



```text

GeoRisk-AI/

│

├── app.py

├── README.md

├── requirements.txt

│

├── src/

│   ├── config.py

│   ├── data\_gen.py

│   ├── data\_loader.py

│   ├── features.py

│   ├── train.py

│   ├── predict.py

│   ├── explain.py

│   ├── model\_comparison.py

│   └── error\_analysis.py

│

├── data/

├── images/

├── artifacts/

├── reports/

└── notebooks/

```



\---



\## Technologies Used



\- Python

\- Scikit-learn

\- SHAP

\- Pandas

\- NumPy

\- Matplotlib

\- Streamlit



\---



\## Future Improvements



\- Integration with real Environment Canada weather data

\- Live API-based weather ingestion

\- Online deployment with Streamlit Cloud

\- Advanced SHAP waterfall plots

\- Time-series forecasting extension



\---



\## Author



Alain Dika

