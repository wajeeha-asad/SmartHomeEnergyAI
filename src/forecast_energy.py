"""Train and evaluate short-term household power forecasting models.

Uses a chronological holdout split. The persistence baseline predicts the
power observed at the same time on the previous day. Random Forest uses
historical lags and calendar features. No future readings are used as features.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "refit_house1_15min.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

INTERVALS_PER_DAY = 96
TEST_FRACTION = 0.20
MIN_ROWS = 14 * INTERVALS_PER_DAY
RANDOM_STATE = 42


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features using only current timestamp and past power values."""
    result = pd.DataFrame(index=data.index)
    timestamps = data["timestamp"]

    result["lag_1"] = data["aggregate_power_w"].shift(1)
    result["lag_4"] = data["aggregate_power_w"].shift(4)
    result["lag_96"] = data["aggregate_power_w"].shift(INTERVALS_PER_DAY)
    result["lag_672"] = data["aggregate_power_w"].shift(7 * INTERVALS_PER_DAY)

    minutes_in_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    angle = 2 * np.pi * minutes_in_day / 1440
    result["hour_sin"] = np.sin(angle)
    result["hour_cos"] = np.cos(angle)

    day_angle = 2 * np.pi * timestamps.dt.dayofweek / 7
    result["weekday_sin"] = np.sin(day_angle)
    result["weekday_cos"] = np.cos(day_angle)
    result["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
    return result


def metrics(actual, predicted) -> dict:
    return {
        "mae_w": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse_w": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
    }


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(
            f"Processed data not found: {DATA_FILE}\n"
            "Run src/preprocess_refit.py first."
        )

    df = pd.read_csv(DATA_FILE)
    required = {"timestamp", "aggregate_power_w"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["aggregate_power_w"] = pd.to_numeric(
        df["aggregate_power_w"], errors="coerce"
    )
    df = df.dropna(subset=["timestamp", "aggregate_power_w"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp").sort_index()

    # Verify the data is on the expected 15-minute grid. Missing timestamps
    # are inserted as NaN; we do not pretend gaps are zero consumption.
    full_index = pd.date_range(
        start=df.index.min().floor("15min"),
        end=df.index.max().floor("15min"),
        freq="15min",
    )
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    df = df.reset_index()

    # Missing target values remain missing. Lag features naturally become
    # missing after gaps and those rows are excluded from model fitting.
    features = make_features(df)
    target = df["aggregate_power_w"]

    dataset = features.copy()
    dataset["target_w"] = target
    dataset["timestamp"] = df["timestamp"]
    dataset["baseline_w"] = features["lag_96"]
    dataset = dataset.dropna(
        subset=list(features.columns) + ["target_w", "baseline_w"]
    )
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

    if len(dataset) < MIN_ROWS:
        raise SystemExit(
            f"Only {len(dataset)} usable rows after creating lag features. "
            f"At least {MIN_ROWS} are required for this initial experiment. "
            "Use a longer date range or inspect missing-data coverage."
        )

    # Chronological split: earlier observations train, later observations test.
    split_at = int(len(dataset) * (1 - TEST_FRACTION))
    train = dataset.iloc[:split_at].copy()
    test = dataset.iloc[split_at:].copy()

    if len(train) < 100 or len(test) < 20:
        raise SystemExit("Not enough rows for a reliable train/test split.")

    feature_columns = list(features.columns)
    model = RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train[feature_columns], train["target_w"])

    baseline_predictions = test["baseline_w"].to_numpy()
    model_predictions = model.predict(test[feature_columns])

    results = {
        "dataset": "REFIT House 1",
        "target": "aggregate household average power (W)",
        "interval_minutes": 15,
        "test_fraction": TEST_FRACTION,
        "split_timestamp": test["timestamp"].iloc[0].isoformat(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": feature_columns,
        "baseline_same_time_previous_day": metrics(
            test["target_w"], baseline_predictions
        ),
        "random_forest": metrics(test["target_w"], model_predictions),
        "note": (
            "Results apply to the selected House 1 time-based holdout. "
            "They do not establish generalization to other households."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "refit_house1_random_forest.joblib"
    dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "interval_minutes": 15,
            "target_column": "aggregate_power_w",
        },
        model_path,
    )

    metrics_path = REPORT_DIR / "forecast_metrics_house1.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    predictions = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "actual_power_w": test["target_w"].to_numpy(),
        "baseline_power_w": baseline_predictions,
        "random_forest_power_w": model_predictions,
    })
    predictions_path = REPORT_DIR / "forecast_predictions_house1.csv"
    predictions.to_csv(predictions_path, index=False)

    # Plot a readable sample from the start of the test period.
    plot_rows = min(len(test), 7 * INTERVALS_PER_DAY)
    plot_data = predictions.iloc[:plot_rows]

    plt.figure(figsize=(13, 5))
    plt.plot(
        plot_data["timestamp"], plot_data["actual_power_w"],
        label="Actual", linewidth=1.2
    )
    plt.plot(
        plot_data["timestamp"], plot_data["baseline_power_w"],
        label="Previous-day baseline", linewidth=1
    )
    plt.plot(
        plot_data["timestamp"], plot_data["random_forest_power_w"],
        label="Random Forest", linewidth=1
    )
    plt.title("REFIT House 1: Forecast vs Actual Power (Test Period)")
    plt.xlabel("Time")
    plt.ylabel("Average power (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_path = REPORT_DIR / "forecast_comparison_house1.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print("\nFORECAST EVALUATION COMPLETE")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Test begins: {test['timestamp'].iloc[0]}")
    print("\nMetrics (lower is better):")
    print(json.dumps({
        "baseline_same_time_previous_day": results[
            "baseline_same_time_previous_day"
        ],
        "random_forest": results["random_forest"],
    }, indent=2))
    print("\nSaved files:")
    print(f"  Model: {model_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Chart: {chart_path}")


if __name__ == "__main__":
    main()
