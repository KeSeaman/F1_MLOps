# F1 MLOps Project Walkthrough

## 1. Environment & Setup
- **Python 3.12**: Enforced via `.python-version`.
- **Dependencies**: Installed `ray[rllib]`, `torch`, `fastf1`, `modin`, `psutil`.
- **Rust Backend**: Successfully built `f1_sim_backend` using `maturin develop`.

## 2. Data Ingestion
- **Source**: 2024 Bahrain Grand Prix (Real Data).
- **Execution**: `python src/data_ingestion.py`
- **Output**: `data/laps.parquet` (Verified).

## 3. Simulation Calibration
- **Script**: `src/calibrate.py`
- **Process**: Linear regression on real tyre data → `data/sim_config.json`
- **Output**: Per-compound degradation rates and pace offsets.

## 4. Agent Training (PPO)
- **Script**: `src/train_agent.py`
- **Fixes Applied**:
    - **API Update**: Migrated to Ray RLlib Legacy API to avoid deprecation warnings.
    - **NVML Warning**: Fixed "Can't initialize NVML" by forcing CPU-only mode (`CUDA_VISIBLE_DEVICES=''`).
    - **Metrics Timeout**: Disabled Ray Dashboard and Metrics Exporter to resolve connection timeouts (`RAY_DISABLE_METRICS_COLLECTION=1`).
    - **NaN Rewards**: Updated `src/env.py` with observation normalization and safer reward scaling.
    - **Missing Config**: Created `data/sim_config.json` for tyre simulation parameters.
    - **Results Output**: Enhanced to output both `data/training_results.json` (structured) and `data/training_results.txt` (human-readable).
- **Results**: Training initiates successfully with 3 parallel workers.

## 5. Dashboard
- **Script**: `notebooks/dashboard.py` (Run with `marimo edit notebooks/dashboard.py`)
- **Features**: 
    - Loads trained PPO model (or defaults to random if missing).
    - Interactive "Race Step" button.
    - Real-time visualization of gap-to-leader, tyre age, and actions.

## 6. Documentation Restructure
- **README.md**: Slimmed down to high-level project overview only (architecture, quick start, tech stack).
- **notes/Artefact.md**: Created with detailed implementation specifics:
    - Data pipeline details and schema
    - Simulation calibration parameters
    - Rust backend deep-dive (types, simulation logic, PyO3 exports)
    - Gymnasium environment design (observation/action spaces, reward function)
    - Training configuration and hyperparameters
    - Dashboard usage guide
    - Development guide and debugging tips

## 7. Usage
```bash
# 1. Install Dependencies
uv sync

# 2. Build Backend
source .venv/bin/activate
maturin develop

# 3. Run Full Pipeline
python src/data_ingestion.py
python src/calibrate.py
python src/train_agent.py

# 4. Launch Dashboard
marimo edit notebooks/dashboard.py
```
