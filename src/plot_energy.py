
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
data_file = project_dir / "data" / "simulated_household.csv"

df = pd.read_csv(data_file, parse_dates=["timestamp"])

plt.figure(figsize=(12, 5))
plt.plot(
    df["timestamp"],
    df["total_power_w"],
    linewidth=1,
)

plt.title("Simulated Household Electricity Demand")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_file = project_dir / "reports" / "sample_demand.png"
output_file.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_file, dpi=150)
plt.show()

print(f"Chart saved to: {output_file}")
