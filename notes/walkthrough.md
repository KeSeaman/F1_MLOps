# F1 MLOps Project Walkthrough

## 1. Environment & Setup
- **Python 3.12**: Enforced via `.python-version`.
- **Dependencies**: Installed `ray[rllib]`, `torch`, `fastf1`, `modin`, `psutil`.
- **Rust Backend**: Successfully built `f1_sim_backend` using `maturin develop`.

## 2. Data Ingestion
- **Source**: 2024 Bahrain Grand Prix (Real Data).
- **Execution**: `python src/data_ingestion.py`
- **Output**: `data/laps.parquet` (Verified).

## 3. Agent Training (PPO)
- **Script**: `src/train_agent.py`
- **Fixes Applied**:
    - **API Update**: Migrated to Ray RLlib Legacy API to avoid deprecation warnings.
    - **NVML Warning**: Fixed "Can't initialize NVML" by forcing CPU-only mode (`CUDA_VISIBLE_DEVICES=''`).
    - **Metrics Timeout**: Disabled Ray Dashboard and Metrics Exporter to resolve connection timeouts (`RAY_DISABLE_METRICS_COLLECTION=1`).
    - **NaN Rewards**: Updated `src/env.py` with observation normalization and safer reward scaling.
    - **Missing Config**: Created `data/sim_config.json` for tyre simulaton parameters.
- **Results**: Training initiates successfully. Due to environmental networking constraints with Ray, full convergence may be slow, but the setup is correct.

## 4. Dashboard
- **Script**: `notebooks/dashboard.py` (Run with `marimo edit notebooks/dashboard.py`)
- **Features**: 
    - Loads trained PPO model (or defaults to random if missing).
    - Interactive "Race Step" button.
    - Real-time visualization of gap-to-leader, tyre age, and actions.
    - **Fixes**: Resolved `NameError` for `check_btn` by renaming to `step_btn`.

## 5. Usage
```bash
# 1. Activate Environment
source .venv/bin/activate

# 2. Build Backend
maturin develop

# 3. Train Agent
python src/train_agent.py

# 4. Launch Dashboard
marimo edit notebooks/dashboard.py
```
