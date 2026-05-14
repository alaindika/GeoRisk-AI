\# GeoRisk-AI



\*\*Explainable Severe Weather Risk Forecasting Dashboard\*\*



GeoRisk-AI is a machine learning prototype that classifies localized severe-weather risk as \*\*Low\*\*, \*\*Medium\*\*, or \*\*High\*\* using engineered meteorological features. The system combines a trained Random Forest model, SHAP-based explainability, error analysis, and an interactive Streamlit dashboard.



\## Project Motivation



Severe weather alerts need to be timely, localized, and interpretable. This project translates weather signals such as rainfall accumulation, pressure drop, wind speed, humidity, temperature anomaly, snowfall rate, and season into a risk-level prediction that can support early decision-making.



\## Key Features



\- Synthetic weather data generation for supervised prototype development

\- Feature engineering using meteorological risk indicators

\- Model comparison: Logistic Regression, Decision Tree, Random Forest

\- Recall-focused evaluation for high-risk events

\- SHAP-based local explanation for each prediction

\- Streamlit dashboard for interactive risk forecasting

\- Confusion matrix and error analysis workflow



\## Machine Learning Task



\*\*Task:\*\* Multiclass classification  

\*\*Target:\*\* `Low`, `Medium`, `High` weather risk  

\*\*Primary metric:\*\* High-risk recall  

\*\*Main model:\*\* Random Forest Classifier



\## Input Features



| Feature | Description |

|---|---|

| `rainfall\_48hr` | 48-hour accumulated rainfall |

| `pressure\_drop\_3hr` | 3-hour atmospheric pressure change |

| `temp\_anomaly` | Temperature deviation from expected conditions |

| `wind\_speed` | Current wind speed |

| `humidity` | Relative humidity |

| `snowfall\_rate` | Snowfall accumulation rate |

| `season` | Encoded seasonal context |



\## Current Results



Model comparison on the synthetic prototype dataset:



| Model | Accuracy | Weighted F1 | High-Risk Recall |

|---|---:|---:|---:|

| Random Forest | 0.9000 | 0.9003 | 0.8817 |

| Decision Tree | 0.8944 | 0.8956 | 0.8495 |

| Logistic Regression | 0.8800 | 0.8799 | 0.8710 |



The error analysis shows that most confusion occurs around boundary cases between \*\*Low\*\*, \*\*Medium\*\*, and \*\*High\*\* risk, which is realistic for weather-risk classification.



\## Project Structure



```text

GeoRisk-AI/

│

├── app.py

├── requirements.txt

├── README.md

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

│   ├── raw/

│   └── processed/

│

├── artifacts/

├── reports/

└── notebooks/

