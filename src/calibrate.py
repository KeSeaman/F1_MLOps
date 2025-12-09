import pandas as pd
import numpy as np
import json
import os

def calibrate_simulation():
    print("Calibrating simulation parameters...")
    
    # Load data
    try:
        df = pd.read_parquet("data/laps.parquet")
    except FileNotFoundError:
        print("Error: data/laps.parquet not found. Run data_ingestion.py first.")
        return

    # Filter for representative laps (clean laps, not in/out loops, reasonable times)
    # Assume generic cutoff for outliers (> 120s is probably SC or Slow)
    df = df[df['LapTimeSeconds'] < 120]
    
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    config = {
        "base_lap_time": float(df['LapTimeSeconds'].min()), # Fastest lap as baseline
        "compounds": {}
    }

    for compound in compounds:
        compound_data = df[df['Compound'] == compound]
        
        if len(compound_data) < 10:
            print(f"Warning: Not enough data for {compound}. Using defaults.")
            config["compounds"][compound] = {
                "degradation": 0.1,
                "pace_offset": 0.0
            }
            continue

        # Simple linear regression for degradation: LapTime ~ Base + Deg * TyreAge
        # We want the 'Deg' coefficient
        
        # Remove outlier TyreLife (sometimes bugs make it huge)
        compound_data = compound_data[compound_data['TyreLife'] < 50]
        
        x = compound_data['TyreLife'].values
        y = compound_data['LapTimeSeconds'].values
        
        if len(x) > 0:
            # Polyfit degree 1: returns [slope, intercept]
            slope, intercept = np.polyfit(x, y, 1)
            
            # Pace offset is intercept - base_lap_time
            pace_offset = intercept - config["base_lap_time"]
            
            # Ensure valid values
            degradation = max(0.01, float(slope))
            pace_offset = max(0.0, float(pace_offset))
            
            config["compounds"][compound] = {
                "degradation": degradation,
                "pace_offset": pace_offset
            }
            print(f"{compound}: Deg={degradation:.4f}, Offset={pace_offset:.4f}")
        else:
             config["compounds"][compound] = {"degradation": 0.1, "pace_offset": 0.0}

    # Save config
    os.makedirs('data', exist_ok=True)
    with open('data/sim_config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print("Calibration complete. Config saved to data/sim_config.json")

if __name__ == "__main__":
    calibrate_simulation()
