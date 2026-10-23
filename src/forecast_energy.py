
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "refit_house1_15min.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

INTERVALS_PER_DAY = 96
TEST_FRACTION = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load and validate the data
# --------------------------------------------------

if not DATA_FILE.exists():
    raise SystemExit(
        f"Processed data not found:\n{DATA_FILE}\n"
        "Run src/preprocess_refit.py first."
    )

df = pd.read_csv(DATA_FILE)

required = {"timestamp", "aggregate_power_w"}
missing = required - set(df.columns)

if missing:
    raise SystemExit(
        f"Required columns are missing: {sorted(missing)}"
    )

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)
df["aggregate_power_w"] = pd.to_numeric(
    df["aggregate_power_w"], errors="coerce"
)

df = df.dropna(
    subset=["timestamp", "aggregate_power_w"]
)
df = df.sort_values("timestamp")
df = df.drop_duplicates("timestamp", keep="last")
df = df.set_index("timestamp")

if df.empty:
    raise SystemExit("No valid readings remain after cleaning.")

# Ensure every row represents one 15-minute interval.
# Missing intervals remain missing; they are not filled with zero.
df = df[["aggregate_power_w"]].resample("15min").mean()

# REFIT aggregate power should be non-negative.
df.loc[df["aggregate_power_w"] < 0, "aggregate_power_w"] = np.nan

print(f"Time range: {df.index.min()} to {df.index.max()}")
print(f"15-minute time slots: {len(df):,}")
print(
    "Missing aggregate readings: "
    f"{df['aggregate_power_w'].isna().sum():,}"
)


# --------------------------------------------------
# 3. Create forecasting features
# --------------------------------------------------

# At time t, use earlier readings to predict power at t.
# lag_96 is the reading from the same time yesterday.
df["lag_1"] = df["aggregate_power_w"].shift(1)
df["lag_4"] = df["aggregate_power_w"].shift(4)
df["lag_96"] = df["aggregate_power_w"].shift(
    INTERVALS_PER_DAY
)

minutes_in_day = (
    df.index.hour * 60 + df.index.minute
)
day_fraction = 2 * np.pi * minutes_in_day / 1440

df["hour_sin"] = np.sin(day_fraction)
df["hour_cos"] = np.cos(day_fraction)
df["day_of_week"] = df.index.dayofweek

feature_columns = [
    "lag_1",
    "lag_4",
    "lag_96",
    "hour_sin",
    "hour_cos",
    "day_of_week",
]

# Keep the target separate from its input features.
target = df["aggregate_power_w"]

# Drop rows with missing target or missing historical features.
dataset = df[feature_columns].copy()
dataset["target_power_w"] = target
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()

if len(dataset) < 300:
    raise SystemExit(
        f"Only {len(dataset)} usable rows remain. "
        "Check REFIT coverage and whether the dataset "
        "contains enough valid readings."
    )


# --------------------------------------------------
# 4. Split by time, not randomly
# --------------------------------------------------

# The last 20% of the timeline is reserved for testing.
# The model never trains on those future target readings.
cut_position = int(len(df) * (1 - TEST_FRACTION))
cutoff_time = df.index[cut_position]

train = dataset.loc[dataset.index < cutoff_time]
test = dataset.loc[dataset.index >= cutoff_time].copy()

if len(train) < 200 or len(test) < 50:
    raise SystemExit(
        "Not enough training or test rows. "
        "Use a longer period of valid REFIT data."
    )

X_train = train[feature_columns]
y_train = train["target_power_w"]

X_test = test[feature_columns]
y_test = test["target_power_w"]

print(f"\nTraining rows: {len(train):,}")
print(f"Testing rows: {len(test):,}")
print(f"Test period begins: {cutoff_time}")


# --------------------------------------------------
# 5. Baseline: use the same time yesterday
# --------------------------------------------------

# lag_96 is one day earlier, so it is a simple
# benchmark for a 15-minute-resolution dataset.
baseline_predictions = X_test["lag_96"].to_numpy()


# --------------------------------------------------
# 6. Train the Random Forest model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=18,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train, y_train)
model_predictions = model.predict(X_test)

# Power cannot be negative. Clip predictions to zero.
model_predictions = np.maximum(model_predictions, 0)


# --------------------------------------------------
# 7. Evaluate both methods on the same test rows
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    return {
        "MAE_W": round(float(mae), 4),
        "RMSE_W": round(float(rmse), 4),
    }


baseline_metrics = calculate_metrics(
    y_test, baseline_predictions
)
model_metrics = calculate_metrics(
    y_test, model_predictions
)

baseline_mae = baseline_metrics["MAE_W"]
model_mae = model_metrics["MAE_W"]

if baseline_mae > 0:
    mae_improvement_percent = (
        (baseline_mae - model_mae) / baseline_mae
    ) * 100
else:
    mae_improvement_percent = None

results = {
    "dataset": "REFIT House 1",
    "target": "aggregate_power_w",
    "interval": "15 minutes",
    "test_fraction": TEST_FRACTION,
    "test_start": str(test.index.min()),
    "test_end": str(test.index.max()),
    "training_rows": int(len(train)),
    "testing_rows": int(len(test)),
    "baseline_same_time_yesterday": baseline_metrics,
    "random_forest": model_metrics,
    "mae_improvement_percent_vs_baseline": (
        round(float(mae_improvement_percent), 2)
        if mae_improvement_percent is not None
        else None
    ),
}


# --------------------------------------------------
# 8. Save metrics, predictions, model, and chart
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with (REPORT_DIR / "forecast_metrics.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(results, file, indent=2)

predictions = pd.DataFrame(
    {
        "timestamp": test.index,
        "actual_power_w": y_test.to_numpy(),
        "baseline_power_w": baseline_predictions,
        "random_forest_power_w": model_predictions,
    }
)

predictions.to_csv(
    REPORT_DIR / "forecast_predictions.csv",
    index=False,
)

dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "target": "aggregate_power_w",
        "interval": "15min",
    },
    MODEL_DIR / "random_forest_energy.joblib",
)

# Plot a readable sample of the test period.
plot_count = min(7 * INTERVALS_PER_DAY, len(test))
plot_data = predictions.iloc[:plot_count]

plt.figure(figsize=(14, 6))
plt.plot(
    plot_data["timestamp"],
    plot_data["actual_power_w"],
    label="Actual demand",
    linewidth=1.5,
)
plt.plot(
    plot_data["timestamp"],
    plot_data["baseline_power_w"],
    label="Yesterday baseline",
    alpha=0.8,
)
plt.plot(
    plot_data["timestamp"],
    plot_data["random_forest_power_w"],
    label="Random Forest",
    alpha=0.8,
)
plt.title("REFIT House 1: Electricity-Demand Forecast")
plt.xlabel("Time")
plt.ylabel("Average power (W)")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()

chart_path = REPORT_DIR / "forecast_comparison.png"
plt.savefig(chart_path, dpi=160)
plt.close()


# --------------------------------------------------
# 9. Print the experiment summary
# --------------------------------------------------

print("\n" + "=" * 55)
print("FORECASTING EXPERIMENT COMPLETE")
print("=" * 55)
print(json.dumps(results, indent=2))
print(f"\nMetrics: {REPORT_DIR / 'forecast_metrics.json'}")
print(f"Predictions: {REPORT_DIR / 'forecast_predictions.csv'}")
print(f"Chart: {chart_path}")
print(f"Model: {MODEL_DIR / 'random_forest_energy.joblib'}")

if mae_improvement_percent is not None:
    if mae_improvement_percent > 0:
        print(
            "\nRandom Forest has lower MAE than the baseline "
            f"by {mae_improvement_percent:.2f}%."
        )
    elif mae_improvement_percent < 0:
        print(
            "\nRandom Forest has higher MAE than the baseline "
            f"by {abs(mae_improvement_percent):.2f}%."
        )
    else:
        print("\nBoth methods have the same MAE.")

print(
    "\nImportant: these metrics apply to this test period "
    "and this household, not all homes."
)
