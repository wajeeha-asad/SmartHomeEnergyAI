
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent

data_file = (
    project_dir
    / "data"
    / "processed"
    / "refit_house1_15min.csv"
)

df = pd.read_csv(
    data_file,
    parse_dates=["timestamp"],
)

# Plot the first seven calendar days containing valid readings.
valid = df.dropna(subset=["aggregate_power_w"])

if valid.empty:
    raise SystemExit(
        "No valid aggregate power readings are available."
    )

start = valid["timestamp"].min().normalize()
end = start + pd.Timedelta(days=7)

week = valid[
    (valid["timestamp"] >= start)
    & (valid["timestamp"] < end)
]

if week.empty:
    raise SystemExit("No readings found in the selected period.")

plt.figure(figsize=(13, 5))

plt.plot(
    week["timestamp"],
    week["aggregate_power_w"],
    linewidth=0.8,
)

plt.title("REFIT House 1: Household Power Demand")
plt.xlabel("Time")
plt.ylabel("Average power (W)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

report_dir = project_dir / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

output_file = report_dir / "refit_house1_demand.png"

plt.savefig(output_file, dpi=150)
plt.show()

print(f"Chart saved to: {output_file}")
