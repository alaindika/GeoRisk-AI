import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_synthetic_data
from src.features import split_features_target


# Load data
df = load_synthetic_data()

X, y = split_features_target(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
}

results = []

for model_name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    high_risk_recall = recall_score(
        y_test,
        y_pred,
        labels=[2],
        average="macro"
    )

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Weighted F1": round(f1, 4),
        "High-Risk Recall": round(high_risk_recall, 4)
    })

# Results dataframe
results_df = pd.DataFrame(results)

print("\nModel Comparison")
print("=" * 60)

print(results_df.sort_values(
    by="High-Risk Recall",
    ascending=False
))