# The Undercut Oracle: F1 MLOps Portfolio

A production-ready Multi-Agent Reinforcement Learning (MARL) system for F1 race strategy optimization, built with a polyglot architecture combining Rust's performance with Python's ML ecosystem.

## 🏎️ Overview

This project demonstrates end-to-end MLOps for F1 race strategy, featuring:
- **Real F1 Data**: 2024 Bahrain Grand Prix via FastF1
- **Multi-Agent RL**: PPO agents learning pit stop strategies
- **High-Performance Simulation**: Rust backend with PyO3 bindings
- **Interactive Dashboard**: Real-time race visualization with Marimo

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         F1 MLOps Stack                          │
├─────────────────────────────────────────────────────────────────┤
│  Frontend    │ Marimo Dashboard (plotly visualizations)        │
├──────────────┼──────────────────────────────────────────────────┤
│  ML/RL       │ Ray RLlib (PPO), PyTorch, Gymnasium              │
├──────────────┼──────────────────────────────────────────────────┤
│  Features    │ Feast Feature Store, Modin (parallel pandas)    │
├──────────────┼──────────────────────────────────────────────────┤
│  Backend     │ Rust Simulation Engine (pyo3, serde, rand)      │
├──────────────┼──────────────────────────────────────────────────┤
│  Data        │ FastF1 API → Parquet → Feature Store            │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
F1_MLOps/
├── backend/              # Rust simulation engine
│   └── src/
│       ├── lib.rs        # PyO3 module exports
│       ├── types.rs      # TyreCompound, DriverState, RaceConfig
│       └── sim.rs        # Race simulation logic
├── src/                  # Python ML pipeline
│   ├── data_ingestion.py # FastF1 → Parquet
│   ├── env.py            # Gymnasium environment wrapper
│   ├── train_agent.py    # Ray RLlib PPO training
│   └── calibrate.py      # Model calibration utilities
├── notebooks/
│   └── dashboard.py      # Marimo interactive dashboard
├── feature_repo/         # Feast feature definitions
├── data/                 # Cached data and parquet files
└── models/               # Trained model checkpoints
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- Rust toolchain (for backend compilation)
- uv package manager

### 1. Create Environment & Install Dependencies
```bash
uv sync
```

### 2. Build Rust Backend
```bash
source .venv/bin/activate
maturin develop
```

### 3. Ingest Real F1 Data
```bash
python src/data_ingestion.py
```
This downloads 2024 Bahrain GP data via FastF1 and creates `data/laps.parquet`.

### 4. Train the RL Agent
```bash
python src/train_agent.py
```
Trains a PPO agent using Ray RLlib for 5 iterations (configurable).

### 5. Launch Dashboard
```bash
marimo edit notebooks/dashboard.py
```

## 🔧 Key Components

### Rust Backend (`backend/`)
The simulation engine handles:
- **Tyre Degradation**: Per-compound degradation curves
- **Pit Stop Logic**: Time penalties and compound switching
- **Race Progression**: Lap-by-lap state updates

```rust
pub enum TyreCompound { Soft, Medium, Hard, Intermediate, Wet }

pub struct DriverState {
    driver_id: String,
    position: u32,
    tyre_compound: TyreCompound,
    tyre_age: u32,
    gap_to_leader: f32,
    // ...
}
```

### Python Environment (`src/env.py`)
Gymnasium-compatible environment that:
- Loads real race data from parquet
- Wraps Rust simulation via PyO3
- Provides observation/action spaces for RL

### Data Pipeline (`src/data_ingestion.py`)
- Fetches telemetry via FastF1
- Computes derived features (gap, cumulative time)
- Saves to Parquet for feature store integration

## 📊 Features & Observations

| Feature | Description |
|---------|-------------|
| `TyreAge` | Laps on current tyres |
| `GapToLeader` | Time delta to P1 |
| `Position` | Current race position |
| `Compound` | Current tyre type |
| `LapNumber` | Race progress |

## 🎯 Actions

| Action ID | Description |
|-----------|-------------|
| 0 | Stay Out |
| 1 | Push |
| 2 | Save Tyres |
| 3 | Pit - Soft |
| 4 | Pit - Medium |
| 5 | Pit - Hard |

## 📈 Training Results

Training uses Ray RLlib's PPO algorithm with:
- 2-layer FC network (64x64)
- Single rollout worker
- PyTorch backend

## 🛠️ Development

### Rebuild Backend After Changes

When you modify any Rust files in `backend/src/` (like `types.rs`, `sim.rs`, or `lib.rs`), you need to recompile the Python extension module. This command compiles the Rust code and installs it directly into your virtual environment for immediate use:

```bash
maturin develop
```

For release builds with optimizations (slower to compile but faster runtime):
```bash
maturin develop --release
```

**Common scenarios requiring rebuild:**
- Adding new fields to `DriverState` or `RaceConfig`
- Modifying simulation logic in `sim.rs`
- Changing PyO3 class exports in `lib.rs`

---

### Run Feature Store

Feast is used to manage and serve ML features for the RL agent. The feature definitions live in `feature_repo/features.py`. This command applies your feature definitions to the local registry:

```bash
feast -c feature_repo apply
```

**What this does:**
- Registers the `driver_stats` FeatureView
- Links the parquet data source (`data/laps.parquet`)
- Enables feature retrieval for training and inference

To materialize features for online serving:
```bash
feast -c feature_repo materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
```

---

### Verify Backend Import

After building, verify the Rust backend is correctly installed and all classes are accessible from Python:

```python
from f1_sim_backend import F1Env, RaceConfig, DriverState, TyreCompound, TyreConfig

# Quick sanity check
print(TyreCompound.Soft)        # TyreCompound.Soft
print(TyreCompound.Medium)      # TyreCompound.Medium

# Create a minimal config
config = RaceConfig(
    total_laps=57,
    track_length_km=5.412,
    base_lap_time=92.0,
    tyre_configs={}
)
print(f"Race: {config.total_laps} laps")
```

---

### Running Tests

Run the full test suite to verify all components work together:

```bash
# Verify data pipeline
python src/data_ingestion.py

# Verify environment loads correctly
python -c "from env import F1GymEnv; env = F1GymEnv(); print(env.reset())"

# Verify training pipeline (short run)
python src/train_agent.py
```

---

### Debugging Tips

**Backend not importing:**
```bash
# Check if module is installed
pip show f1-mlops

# Rebuild from scratch
rm -rf target/ && maturin develop
```

**Ray initialization issues:**
```python
import ray
ray.shutdown()  # Reset any existing instance
ray.init(ignore_reinit_error=True)
```

**FastF1 cache issues:**
```bash
# Clear FastF1 cache if data is corrupted
rm -rf data/cache/*
```

## 📝 License

MIT

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) - F1 telemetry data
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/) - Reinforcement learning
- [Marimo](https://marimo.io/) - Reactive notebooks
- [PyO3](https://pyo3.rs/) - Rust-Python bindings
