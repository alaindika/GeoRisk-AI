import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN


def split_features_target(df: pd.DataFrame):
    """
    Split the dataset into model inputs X and target y.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def validate_feature_columns(df: pd.DataFrame):
    """
    Check that all required model features exist in the dataset.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return True