
from pathlib import Path

import numpy as np
import pandas as pd


# Keep the example reproducible.
rng = np.random.default_rng(42)

# Seven days, measured every 15 minutes.
intervals = 7 * 24 * 4

timestamps = pd.date_range(
    start="2026-01-05 00:00:00",
    periods=intervals,
    freq="15min",
)

df = pd.DataFrame({"timestamp": timestamps})

hour = df["timestamp"].dt.hour
minute = df["timestamp"].dt.minute
day_of_week = df["timestamp"].dt.dayofweek

# Simulated background household consumption.
morning_peak = 180 * np.exp(-((hour - 8) / 2) ** 2)
evening_peak = 300 * np.exp(-((hour - 20) / 3) ** 2)


base_load = (
    180
    + morning_peak
    + evening_peak
    + rng.normal(0, 20, intervals)
).clip(lower=80)

# Illustrative appliance profiles.
# These are synthetic data, not real household measurements.

# Refrigerator: simplified cycling pattern.
refrigerator_w = np.where(
    (minute < 30),
    150,
    45,
)

# Washing machine: one simulated 1-hour cycle each day.
washing_machine_w = np.zeros(intervals)

for day in range(7):
    start = day * 96 + 40
    washing_machine_w[start:start + 4] = 500

# AC: higher usage during warm afternoon/evening hours.
ac_probability = np.where(
    (hour >= 13) & (hour <= 22),
    0.65,
    0.08,
)

ac_on = rng.random(intervals) < ac_probability
air_conditioner_w = ac_on.astype(int) * 1200

# Water heater: a short simulated morning cycle.
water_heater_w = np.where(
    (hour == 6) & (minute < 30),
    1500,
    0,
)

df["base_load_w"] = base_load
df["refrigerator_w"] = refrigerator_w
df["washing_machine_w"] = washing_machine_w
df["air_conditioner_w"] = air_conditioner_w
df["water_heater_w"] = water_heater_w

appliance_columns = [
    "base_load_w",
    "refrigerator_w",
    "washing_machine_w",
    "air_conditioner_w",
    "water_heater_w",
]

df["total_power_w"] = df[appliance_columns].sum(axis=1)

# Every row represents 15 minutes.
df["energy_kwh"] = df["total_power_w"] * 0.25 / 1000

output_dir = Path(__file__).resolve().parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "simulated_household.csv"
df.to_csv(output_file, index=False)

print("Sample household dataset generated.")
print(f"Saved to: {output_file}")
print(f"Intervals: {len(df)}")
print(f"Total simulated energy: {df['energy_kwh'].sum():.2f} kWh")
print(f"Average power: {df['total_power_w'].mean():.2f} W")
print(f"Peak power: {df['total_power_w'].max():.2f} W")
