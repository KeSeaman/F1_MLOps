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
├──────────────┬──────────────────────────────────────────────────┤
│  Frontend    │ Marimo Dashboard (Plotly visualizations)         │
├──────────────┼──────────────────────────────────────────────────┤
│  ML/RL       │ Ray RLlib (PPO), PyTorch, Gymnasium              │
├──────────────┼──────────────────────────────────────────────────┤
│  Features    │ Feast Feature Store, Modin (parallel pandas)     │
├──────────────┼──────────────────────────────────────────────────┤
│  Backend     │ Rust Simulation Engine (PyO3, serde, rand)       │
├──────────────┼──────────────────────────────────────────────────┤
│  Data        │ FastF1 API → Parquet → Feature Store             │
└──────────────┴──────────────────────────────────────────────────┘
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
│   └── calibrate.py      # Model calibration from real data
├── notebooks/
│   └── dashboard.py      # Marimo interactive dashboard
├── data/                 # Cached data and parquet files
├── models/               # Trained model checkpoints
└── notes/
    └── Artefact.md       # Detailed results & implementation notes
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- Rust toolchain (for backend compilation)
- uv package manager

### 1. Install Dependencies
```bash
uv sync
```

### 2. Build Rust Backend
```bash
source .venv/bin/activate
maturin develop
```

### 3. Run Full Pipeline
```bash
# Ingest real F1 data (2024 Bahrain GP)
python src/data_ingestion.py

# Calibrate simulation from real data
python src/calibrate.py

# Train the RL agent (5 iterations PPO)
python src/train_agent.py
```

### 4. Launch Dashboard
```bash
marimo edit notebooks/dashboard.py
```

## 🔧 Tech Stack

| Layer | Technology |
|-------|------------|
| **Simulation** | Rust (PyO3, serde, rand) |
| **ML/RL** | Ray RLlib, PyTorch, Gymnasium |
| **Data** | FastF1, Modin, Parquet, Feast |
| **Dashboard** | Marimo, Plotly |
| **Package Mgmt** | uv, maturin |

## 📊 Results & Details

For detailed training results, performance analysis, implementation deep-dives, and development guides, see **[notes/Artefact.md](notes/Artefact.md)**.

## 📝 License

MIT

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) — F1 telemetry data
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/) — Reinforcement learning
- [Marimo](https://marimo.io/) — Reactive notebooks
- [PyO3](https://pyo3.rs/) — Rust-Python bindings
