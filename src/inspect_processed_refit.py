
from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parent.parent

data_file = (
    project_dir
    / "data"
    / "processed"
    / "refit_house1_15min.csv"
)

if not data_file.exists():
    raise SystemExit(
        "Processed dataset not found. "
        "Run preprocess_refit.py first."
    )

df = pd.read_csv(
    data_file,
    parse_dates=["timestamp"],
)

print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nDate range:")
print(df["timestamp"].min())
print(df["timestamp"].max())

print("\nTime interval counts:")
print(
    df["timestamp"]
    .diff()
    .value_counts()
    .head()
)

print("\nCoverage summary:")
print(df["coverage_fraction"].describe())

print("\nMissing values:")
print(df.isna().sum())

print("\nAggregate power summary:")
print(df["aggregate_power_w"].describe())
