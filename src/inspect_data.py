
from pathlib import Path

import pandas as pd

data_file = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "simulated_household.csv"
)

df = pd.read_csv(data_file, parse_dates=["timestamp"])

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df[["total_power_w", "energy_kwh"]].describe())

print("\nMissing values:")
print(df.isna().sum())
