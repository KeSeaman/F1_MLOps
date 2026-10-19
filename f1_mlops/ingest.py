"""Download a race's lap data with FastF1 and save it as data/laps.parquet + data/race.json."""

import json
import logging
from pathlib import Path

import fastf1
import pandas as pd

LAP_COLUMNS = [
    "Driver", "LapNumber", "Stint", "Compound", "TyreLife", "FreshTyre", "Position",
    "TrackStatus", "IsAccurate", "Deleted", "PitInTime", "PitOutTime",
]


def fetch_race(year: int, event: str, data_dir: Path, cache_dir: Path) -> pd.DataFrame:
    """Save one race's laps and metadata under data_dir and return the laps."""
    # Plain pandas: a race is ~1,200 laps, far too small for Modin/Ray, whose start-up
    # made this step 5.7x slower (27.4 s vs 4.8 s) for identical output.
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    logging.getLogger("fastf1").setLevel(logging.WARNING)

    session = fastf1.get_session(year, event, "R")
    session.load(telemetry=False, weather=False, messages=True)  # messages: deleted laps

    laps = pd.DataFrame(session.laps)[LAP_COLUMNS + ["LapTime", "Time"]]
    laps = laps[laps["LapNumber"].notna()].copy()
    laps["LapTimeSeconds"] = laps.pop("LapTime").dt.total_seconds()
    # Session time when the lap ended; differences between drivers give the gaps.
    laps["SessionTimeSeconds"] = laps.pop("Time").dt.total_seconds()
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    results = session.results
    race = {
        "year": year,
        "event": session.event["EventName"],
        "total_laps": int(session.total_laps),
        "results": [
            {"driver": r.Abbreviation, "grid": int(r.GridPosition), "position": int(r.Position),
             "status": r.Status}
            for r in results.itertuples()
        ],
    }

    data_dir.mkdir(parents=True, exist_ok=True)
    laps.to_parquet(data_dir / "laps.parquet")
    (data_dir / "race.json").write_text(json.dumps(race, indent=2) + "\n")
    return laps
