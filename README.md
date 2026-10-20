# AI-Driven Smart Home Energy Forecasting and Appliance Scheduling Using a 3D Digital Twin

A software-only Final Year Project exploring household electricity-demand forecasting, constraint-aware appliance scheduling, and interactive 3D smart-home simulation.

## Project Overview

The goal is to develop a virtual household energy-management system that predicts electricity demand, schedules flexible appliances within defined operational constraints, and visualizes the resulting decisions in a 3D digital twin.

The system will be evaluated against conventional scheduling strategies to determine whether forecast-informed scheduling can reduce peak demand or electricity cost without violating appliance requirements.

## Current Progress

**Phase 1: Project Setup and Synthetic Energy Simulation**

- Created a Python virtual environment.
- Installed Pandas, NumPy, and Matplotlib.
- Generated a seven-day synthetic household electricity dataset.
- Simulated five load components: background household load, refrigerator, washing machine, air conditioner, and water heater.
- Calculated interval energy consumption, average power, and peak power.
- Inspected the dataset and generated an electricity-demand chart.

**Important:** The initial dataset is synthetic and is used only for development and testing. It is not real household measurement data.

## Phase 2: Real Household Data

- Added a chunk-based REFIT preprocessing pipeline.
- Resampled selected household readings to 15-minute intervals.
- Recorded sampling coverage and retained missing-value information.
- Generated a demand chart from processed REFIT measurements.

The selected household is House 1. Raw dataset files are stored locally
and excluded from Git version control. Processed outputs and results
will be reviewed before being committed.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn (planned)
- Google OR-Tools (planned)
- FastAPI (planned)
- React and Vite (planned)
- Three.js / React Three Fiber (planned)

## Project Structure

```text
SmartHomeEnergyAI/
├── data/
│   └── simulated_household.csv
├── reports/
│   └── sample_demand.png
├── src/
│   ├── generate_sample_data.py
│   ├── inspect_data.py
│   └── plot_energy.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Generate the synthetic dataset

```powershell
python src/generate_sample_data.py
```

### 5. Inspect the data and create a chart

```powershell
python src/inspect_data.py
python src/plot_energy.py
```

## Planned Research and Development

1. Acquire and preprocess real household electricity measurements.
2. Build and evaluate short-term electricity-demand forecasting models.
3. Implement rule-based and constraint-optimized appliance scheduling.
4. Integrate the forecasting and scheduling modules through a Python API.
5. Develop an interactive 3D smart-home digital twin.
6. Compare forecasting accuracy, peak demand, energy consumption, cost, and operational constraints.

## Research Integrity

Real measurements, synthetic data, and simulated scenarios will be clearly distinguished. Performance claims will be based on reproducible experiments and held-out evaluation data.

## Project Status

In development — initial Python setup, synthetic energy simulation, data inspection, and demand visualization completed.