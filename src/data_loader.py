import pandas as pd

from src.config import PROCESSED_DATA_DIR


def load_synthetic_data():
    """
    Load the generated synthetic weather dataset.
    """
    file_path = PROCESSED_DATA_DIR / "synthetic_weather_data.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            "Synthetic data not found. Run: python .\\src\\data_gen.py"
        )

    return pd.read_csv(file_path)