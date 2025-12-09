import marimo

__generated_with = "0.8.0"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import numpy as np
    import sys
    import os
    
    # Add backend to path if not installed
    # sys.path.append("../backend") 
    # But we installed it via maturin develop
    
    from f1_sim_backend import F1Env, RaceConfig, DriverState, TyreCompound
    return F1Env, RaceConfig, DriverState, TyreCompound, mo, np, os, pd, px, sys

@app.cell
def __(mo):
    mo.md(r"""# 🏎️ The Undercut Oracle: Live Strategy Dashboard""")
    return

@app.cell
def __(DriverState, F1Env, RaceConfig, TyreCompound, mo):
    # Initialize Simulation State (cached in cell)
    config = RaceConfig(total_laps=66, track_length_km=4.675, base_lap_time=80.0)
    
    # Initial Drivers
    initial_drivers = [
        DriverState("HAM", 2, 0, TyreCompound.Medium, 0, 1.5, 0.0, 0, "OnTrack"),
        DriverState("VER", 1, 0, TyreCompound.Medium, 0, 0.0, 0.0, 0, "OnTrack"),
        DriverState("BOT", 3, 0, TyreCompound.Medium, 0, 5.0, 0.0, 0, "OnTrack"),
    ]
    
    # We use a state wrapper to persist the sim across re-runs if needed, 
    # but Marimo cells re-run only when inputs change.
    # To simulate a loop, we can use mo.ui.refresh or button.
    
    sim = F1Env(config, initial_drivers)
    return config, initial_drivers, sim

@app.cell
def __(mo):
    step_btn = mo.ui.button(label="🏁 Advance Lap")
    reset_btn = mo.ui.button(label="🔄 Reset")
    return reset_btn, step_btn

@app.cell
def __(DriverState, TyreCompound, config, reset_btn, sim, step_btn):
    # Handle Reset
    if reset_btn.value:
        # Re-init logic would go here, but sim is defined in previous cell.
        # We might need a class to hold state or use mo.state
        pass

    # Handle Step
    if step_btn.value:
        # Simple logic
        actions = {
            "HAM": "StayOut",
            "VER": "StayOut",
            "BOT": "StayOut"
        }
        sim.step(actions)
        
    drivers = sim.get_drivers()
    return actions, drivers

@app.cell
def __(drivers, mo, pd, px):
    # Visualization
    data = []
    for d_id, d in drivers.items():
        data.append({
            "Driver": d_id,
            "Gap": d.gap_to_leader,
            "TyreAge": d.tyre_age,
            "Position": d.position
        })
    
    df = pd.DataFrame(data)
    
    chart = px.bar(df, x="Driver", y="Gap", title="Gap to Leader", color="Driver",
                  color_discrete_map={"HAM": "#00D2BE", "VER": "#0600EF", "BOT": "#00D2BE"})
    
    mo.ui.table(df)
    return chart, data, df, d_id, d

@app.cell
def __(chart, mo):
    mo.vstack([chart])
    return
