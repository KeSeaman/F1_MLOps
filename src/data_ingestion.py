import fastf1
import modin.pandas as pd
import ray
import os
import numpy as np

def fetch_and_process_data():
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print("Fetching data...")
    # Enable cache
    os.makedirs('data/cache', exist_ok=True)
    fastf1.Cache.enable_cache('data/cache')

    # Load session - 2024 Bahrain Grand Prix
    session = fastf1.get_session(2024, 'Bahrain', 'R')
    session.load()

    laps = session.laps
    
    # Convert to Modin DataFrame (converting from pandas since FastF1 returns pandas)
    # FastF1 returns a custom object that behaves like DataFrame, but let's convert explicitly
    laps_df = pd.DataFrame(laps)
    
    print("Processing data with Modin...")
    
    # Select relevant columns
    cols = ['Driver', 'LapNumber', 'LapTime', 'TyreLife', 'Compound', 'Position', 'PitInTime', 'PitOutTime']
    # Note: FastF1 column names might vary slightly, checking standard names
    
    # Filter out non-race laps if any
    laps_df = laps_df[laps_df['LapNumber'].notna()]
    
    # Convert LapTime to seconds
    laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()
    
    # Ensure TyreLife is float and handle NaNs
    laps_df['TyreLife'] = laps_df['TyreLife'].fillna(0).astype(float)
    
    # Calculate Gap to Leader
    # We need to calculate cumulative time for each driver
    # Group by Driver and cumsum LapTime
    laps_df = laps_df.sort_values(by=['Driver', 'LapNumber'])
    laps_df['TotalTime'] = laps_df.groupby('Driver')['LapTimeSeconds'].cumsum()
    
    # Find leader's time per lap
    leader_times = laps_df[laps_df['Position'] == 1][['LapNumber', 'TotalTime']].rename(columns={'TotalTime': 'LeaderTime'})
    
    # Merge leader time back
    laps_df = laps_df.merge(leader_times, on='LapNumber', how='left')
    
    # Calculate Gap
    laps_df['GapToLeader'] = laps_df['TotalTime'] - laps_df['LeaderTime']
    
    # Fill NA gaps
    laps_df['GapToLeader'] = laps_df['GapToLeader'].fillna(0)

    # Add EventTimestamp for Feast
    # Use session start time + TotalTime
    session_start = session.date
    # Ensure session_start is datetime
    laps_df['EventTimestamp'] = session_start + pd.to_timedelta(laps_df['TotalTime'], unit='s')

    # Save to parquet
    output_path = "data/laps.parquet"
    laps_df.to_parquet(output_path)
    
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    fetch_and_process_data()
