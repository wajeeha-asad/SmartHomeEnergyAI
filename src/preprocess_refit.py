
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

HOUSE_NUMBER = 1
CHUNK_SIZE = 200_000
INTERVAL = "15min"

# REFIT readings are approximately 8 seconds apart.
EXPECTED_READINGS_PER_INTERVAL = 900 / 8
MIN_COVERAGE = 0.80


# --------------------------------------------------
# 2. Find the selected household file
# --------------------------------------------------

matches = list(RAW_DIR.rglob(f"CLEAN_House{HOUSE_NUMBER}.csv"))

if not matches:
    print("Could not find the expected REFIT household file.")
    print(f"Expected filename: CLEAN_House{HOUSE_NUMBER}.csv")
    print(f"Search location: {RAW_DIR}")
    print("CSV files found:")

    for path in RAW_DIR.rglob("*.csv"):
        print(f"  - {path.relative_to(RAW_DIR)}")

    raise SystemExit(
        "Check your extracted filenames before continuing."
    )

input_file = matches[0]

if len(matches) > 1:
    print("Multiple matching files were found:")
    for path in matches:
        print(f"  - {path}")
    print(f"Using the first match: {input_file}")

print(f"Input file: {input_file}")


# --------------------------------------------------
# 3. Inspect column names before loading large data
# --------------------------------------------------

header = pd.read_csv(input_file, nrows=0)
original_columns = list(header.columns)


def normalize_name(name):
    """Normalize column names for easier matching."""
    return " ".join(str(name).strip().lower().split())


normalized_columns = {
    normalize_name(column): column
    for column in original_columns
}

datetime_candidates = [
    "datetime",
    "date time",
    "timestamp",
    "time",
]

timestamp_column = None

for candidate in datetime_candidates:
    if candidate in normalized_columns:
        timestamp_column = normalized_columns[candidate]
        break

if timestamp_column is None:
    raise SystemExit(
        "No recognizable datetime column was found. "
        f"Available columns: {original_columns}"
    )

power_columns = []

for column in original_columns:
    normalized = normalize_name(column)

    if normalized == "aggregate":
        power_columns.append(column)
    elif normalized.startswith("appliance"):
        power_columns.append(column)

if not any(
    normalize_name(column) == "aggregate"
    for column in power_columns
):
    raise SystemExit(
        "The Aggregate power column is missing. "
        f"Available columns: {original_columns}"
    )

print(f"Timestamp column: {timestamp_column}")
print(f"Power columns: {power_columns}")


# --------------------------------------------------
# 4. Process the file in manageable chunks
# --------------------------------------------------

# Store small interval-level summaries instead of the
# entire high-frequency dataset in memory.
sum_parts = []
count_parts = []

reader = pd.read_csv(
    input_file,
    usecols=[timestamp_column] + power_columns,
    chunksize=CHUNK_SIZE,
)

total_rows = 0
invalid_timestamps = 0

for chunk_number, chunk in enumerate(reader, start=1):

    total_rows += len(chunk)

    raw_time = chunk[timestamp_column]

    # Support Unix timestamps as well as readable dates.
    if pd.api.types.is_numeric_dtype(raw_time):
        timestamps = pd.to_datetime(
            raw_time,
            unit="s",
            errors="coerce",
        )
    else:
        numeric_time = pd.to_numeric(
            raw_time,
            errors="coerce",
        )

        numeric_fraction = numeric_time.notna().mean()

        if numeric_fraction > 0.95:
            timestamps = pd.to_datetime(
                numeric_time,
                unit="s",
                errors="coerce",
            )
        else:
            timestamps = pd.to_datetime(
                raw_time,
                errors="coerce",
            )

    invalid_timestamps += int(timestamps.isna().sum())

    valid = timestamps.notna()

    if not valid.any():
        print(f"Chunk {chunk_number}: no valid timestamps.")
        continue

    timestamps = timestamps.loc[valid]
    power = chunk.loc[valid, power_columns].copy()

    # Convert readings to numbers and discard invalid values.
    for column in power_columns:
        power[column] = pd.to_numeric(
            power[column],
            errors="coerce",
        )

    # REFIT active power values are in watts.
    # Negative or implausible readings should not silently
    # become zero; flag them as missing for this pipeline.
    for column in power_columns:
        invalid_power = (
            (power[column] < 0)
            | (power[column] > 5000)
        )
        power.loc[invalid_power, column] = np.nan

    power.index = pd.DatetimeIndex(timestamps)

    # Bin readings into 15-minute intervals.
    interval_sum = power.resample(INTERVAL).sum(min_count=1)
    interval_count = power.resample(INTERVAL).count()

    sum_parts.append(interval_sum)
    count_parts.append(interval_count)

    print(
        f"Processed chunk {chunk_number:,} "
        f"({total_rows:,} rows read)"
    )


# --------------------------------------------------
# 5. Combine interval summaries from all chunks
# --------------------------------------------------

if not sum_parts:
    raise SystemExit("No usable timestamped readings were found.")

total_sum = pd.concat(sum_parts).groupby(level=0).sum(min_count=1)
total_count = pd.concat(count_parts).groupby(level=0).sum()

# Mean power within each 15-minute interval.
mean_power = total_sum.div(total_count.where(total_count > 0))

# Expected coverage based on nominal 8-second sampling.
aggregate_column = next(
    column
    for column in power_columns
    if normalize_name(column) == "aggregate"
)

sample_count = total_count[aggregate_column]

coverage = (
    sample_count / EXPECTED_READINGS_PER_INTERVAL
).clip(upper=1.0)

# Mark poorly covered intervals as missing.
# Do not replace them with zero.
poor_coverage = coverage < MIN_COVERAGE

mean_power.loc[poor_coverage, :] = np.nan


# --------------------------------------------------
# 6. Build the processed dataset
# --------------------------------------------------

processed = pd.DataFrame(index=mean_power.index)
processed.index.name = "timestamp"

for column in power_columns:
    normalized = normalize_name(column)

    if normalized == "aggregate":
        output_name = "aggregate_power_w"
    else:
        output_name = normalized.replace(" ", "_") + "_w"

    processed[output_name] = mean_power[column]

processed["sample_count"] = sample_count
processed["coverage_fraction"] = coverage

# One 15-minute interval = 0.25 hours.
processed["energy_kwh"] = (
    processed["aggregate_power_w"] * 0.25 / 1000
)

processed = processed.reset_index()


# --------------------------------------------------
# 7. Save results and report data quality
# --------------------------------------------------

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

output_file = (
    PROCESSED_DIR / f"refit_house{HOUSE_NUMBER}_15min.csv"
)

processed.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("REFIT PREPROCESSING COMPLETE")
print("=" * 60)
print(f"Source: {input_file.name}")
print(f"Rows read: {total_rows:,}")
print(f"Invalid timestamps: {invalid_timestamps:,}")
print(f"15-minute intervals: {len(processed):,}")
print(
    "Intervals with low coverage: "
    f"{int(poor_coverage.sum()):,}"
)
print(f"Output: {output_file}")

print("\nProcessed dataset preview:")
print(processed.head())

print("\nMissing values per column:")
print(processed.isna().sum())

print("\nAggregate power summary (W):")
print(processed["aggregate_power_w"].describe())

print(
    "\nTotal energy across valid intervals: "
    f"{processed['energy_kwh'].sum(min_count=1):.2f} kWh"
)
