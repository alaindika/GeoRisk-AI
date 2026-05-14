import json
import joblib

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
)
from src.data_loader import load_synthetic_data
from src.features import split_features_target, validate_feature_columns


def train_model():
    df = load_synthetic_data()
    validate_feature_columns(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train_sm, y_train_sm)

    y_pred = model.predict(X_test)

    print("Model evaluation")
    print("=" * 40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"High-risk recall: {recall_score(y_test, y_pred, labels=[2], average='macro'):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    print()
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved feature columns to: {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    train_model()