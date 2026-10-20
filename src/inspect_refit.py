
from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
raw_dir = project_dir / "data" / "raw"

csv_files = list(raw_dir.rglob("*.csv"))

if not csv_files:
    print("No CSV files found.")
    print(f"Place the extracted REFIT CSV files inside: {raw_dir}")
    raise SystemExit(1)

print(f"Found {len(csv_files)} CSV file(s).")

for file_path in csv_files:
    print("\n" + "=" * 70)
    print(f"File: {file_path.relative_to(raw_dir)}")
    print(f"Size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

    try:
        # Read only a small sample first.
        sample = pd.read_csv(file_path, nrows=5)

        print("\nColumns:")
        print(sample.columns.tolist())

        print("\nFirst five rows:")
        print(sample.to_string(index=False))

        print("\nData types:")
        print(sample.dtypes)

    except Exception as error:
        print(f"Could not inspect this file: {error}")
