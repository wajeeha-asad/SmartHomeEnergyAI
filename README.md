# AI-Driven Smart Home Energy Forecasting and Appliance Scheduling Using a 3D Digital Twin

A software-only Final Year Project exploring household electricity-demand forecasting, constraint-aware appliance scheduling, and interactive 3D smart-home simulation.

## Project Overview

The goal is to develop a virtual household energy-management system that predicts electricity demand, schedules flexible appliances within defined operational constraints, and visualizes the resulting decisions in a 3D digital twin.

The system will be evaluated against conventional scheduling strategies to determine whether forecast-informed scheduling can reduce peak demand or electricity cost without violating appliance requirements.

## Current Progress

### Phase 1: Project Setup and Synthetic Energy Simulation

- Created a Python virtual environment.
- Generated a seven-day synthetic household electricity dataset.
- Simulated background household load, refrigerator, washing machine, air conditioner, and water heater.
- Inspected the dataset and generated a demand chart.

**Research integrity:** the synthetic dataset is for development and is not real household measurement data.

### Phase 2: Real Household Data

- Added chunk-based REFIT preprocessing.
- Resampled House 1 readings to 15-minute intervals.
- Recorded sampling coverage and retained missing-value information.
- Added data-inspection and visualization scripts.

Raw REFIT files should remain local and excluded from Git. Processed datasets should only be committed after reviewing licensing, size, and data quality.

### Phase 3: Initial Forecasting Pipeline

- Added `src/forecast_energy.py` to compare a same-time-previous-day baseline with a Random Forest regressor.
- Uses a chronological train/test split rather than randomly mixing time-series observations.
- Calculates MAE and RMSE and saves predictions, metrics, a comparison plot, and a serialized model.
- Initial evaluation targets House 1 only. This does **not** establish generalization to other homes; multi-household evaluation is planned.
- The script requires `data/processed/refit_house1_15min.csv` to exist locally before it can run.


## Phase 4: Electricity-Demand Forecasting

- Prepared time-based features from processed REFIT House 1 readings.
- Implemented a same-time-yesterday baseline.
- Trained a Random Forest regression model.
- Used a chronological train/test split to evaluate future-period predictions.
- Compared both approaches using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Saved the evaluation metrics and forecast comparison chart under `reports/`.

The experiment's measured results are stored in
`reports/forecast_metrics.json`. Performance conclusions should
be based on those results and the selected test period.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn and Joblib
- Google OR-Tools (planned)
- FastAPI (planned)
- React and Vite (planned)
- Three.js / React Three Fiber (planned)

## Project Structure

```text
SmartHomeEnergyAI/
├── data/
│   ├── simulated_household.csv
│   └── processed/                 # local processed REFIT data; verify .gitignore
├── models/                        # generated locally by forecasting script
├── reports/                       # generated charts and evaluation outputs
├── src/
│   ├── generate_sample_data.py
│   ├── inspect_data.py
│   ├── plot_energy.py
│   ├── inspect_refit.py
│   ├── preprocess_refit.py
│   ├── inspect_processed_refit.py
│   ├── plot_processed_refit.py
│   └── forecast_energy.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```powershell
git clone https://github.com/wajeeha-asad/SmartHomeEnergyAI.git
cd SmartHomeEnergyAI
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Generate and inspect the synthetic dataset

```powershell
python src/generate_sample_data.py
python src/inspect_data.py
python src/plot_energy.py
```

### 5. Preprocess REFIT data

Download REFIT from its official dataset page, extract it locally, and configure the input path in `src/preprocess_refit.py` to match your local file. Then run:

```powershell
python src/preprocess_refit.py
python src/inspect_processed_refit.py
python src/plot_processed_refit.py
```

### 6. Train and evaluate the initial forecasting model

Ensure `data/processed/refit_house1_15min.csv` exists and contains `timestamp` and `aggregate_power_w` columns. Then run:

```powershell
python src/forecast_energy.py
```

The script writes local outputs to `models/` and `reports/`. Review the metrics and generated chart before making any performance claims.

## Planned Research and Development

1. Validate preprocessing across additional REFIT households.
2. Compare forecasting models with leakage-safe chronological evaluation.
3. Implement rule-based and constraint-optimized appliance scheduling.
4. Integrate forecasting and scheduling through a Python API.
5. Develop an interactive 3D smart-home digital twin.
6. Compare forecasting accuracy, peak demand, energy use, cost (when tariff assumptions are defined), and operational constraints.

## Research Integrity

Real measurements, synthetic data, and simulated scenarios will be clearly distinguished. Performance claims will be based on reproducible experiments and held-out evaluation data. Shifting appliance usage can reduce peak demand or cost under suitable constraints, but does not automatically reduce total energy consumption.

## Project Status

In development — synthetic data simulation, REFIT preprocessing tools, and an initial forecasting pipeline are in place. The forecasting script has not yet been verified against the user's local processed dataset; run it locally and inspect outputs before reporting results.