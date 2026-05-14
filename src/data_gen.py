import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED,
    N_SAMPLES,
    PROCESSED_DATA_DIR,
)

def generate_synthetic_weather_data(n_samples=N_SAMPLES, random_seed=RANDOM_SEED):
    np.random.seed(random_seed)

    seasons = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.25, 0.25, 0.25])

    rainfall_48hr = np.random.gamma(shape=2, scale=10, size=n_samples)
    pressure_drop_3hr = np.random.normal(loc=0, scale=4, size=n_samples)
    temp_anomaly = np.random.normal(loc=0, scale=6, size=n_samples)
    wind_speed = np.random.gamma(shape=2, scale=18, size=n_samples)
    humidity = np.clip(np.random.beta(a=5, b=3, size=n_samples) * 100, 10, 100)

    snowfall_rate = np.where(
        seasons == 0,
        np.random.gamma(shape=1.5, scale=0.8, size=n_samples),
        0.0,
    )

    random_draw = np.random.random(n_samples)
    labels = np.where(random_draw < 0.10, 2, np.where(random_draw < 0.35, 1, 0))

    high_mask = labels == 2
    medium_mask = labels == 1

    rainfall_48hr[high_mask] += np.random.uniform(40, 80, high_mask.sum())
    pressure_drop_3hr[high_mask] -= np.random.uniform(6, 15, high_mask.sum())
    wind_speed[high_mask] += np.random.uniform(50, 80, high_mask.sum())

    snowfall_rate[high_mask] += np.where(
        seasons[high_mask] == 0,
        np.random.uniform(1.5, 4, high_mask.sum()),
        0,
    )

    rainfall_48hr[medium_mask] += np.random.uniform(15, 40, medium_mask.sum())
    pressure_drop_3hr[medium_mask] -= np.random.uniform(2, 6, medium_mask.sum())
    wind_speed[medium_mask] += np.random.uniform(20, 50, medium_mask.sum())

    rainfall_48hr += np.random.normal(0, 3, n_samples)
    pressure_drop_3hr += np.random.normal(0, 1, n_samples)
    wind_speed += np.random.normal(0, 5, n_samples)

    rainfall_48hr = np.clip(rainfall_48hr, 0, None)
    wind_speed = np.clip(wind_speed, 0, None)

    df = pd.DataFrame({
        "rainfall_48hr": rainfall_48hr,
        "pressure_drop_3hr": pressure_drop_3hr,
        "temp_anomaly": temp_anomaly,
        "wind_speed": wind_speed,
        "humidity": humidity,
        "snowfall_rate": snowfall_rate,
        "season": seasons,
        "risk_label": labels,
    })

    df["risk_category"] = df["risk_label"].map({0: "Low", 1: "Medium", 2: "High"})
    return df


if __name__ == "__main__":
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_weather_data()
    output_path = PROCESSED_DATA_DIR / "synthetic_weather_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(df.shape)
    print(df["risk_category"].value_counts())