import copy

import numpy as np
import pandas as pd
import pytest

from f1_mlops.env import Scenario

CALIBRATION = {
    "race": {"year": 2021, "event": "Test Grand Prix", "total_laps": 11},
    "first_lap": 2,
    "total_laps": 10,
    "base_lap_time": 80.0,
    "fuel_effect": 0.0,
    "pit_loss": 20.0,
    "lap_noise_sd": 0.0,
    "dirty_air_gap": 2.0,
    "dirty_air_loss": 0.5,
    "compounds": [
        {"name": "SOFT", "pace_offset": 0.0, "degradation": 0.2},
        {"name": "MEDIUM", "pace_offset": 0.5, "degradation": 0.1},
    ],
    "drivers": {
        "HAM": {"pace_offset": 0.0, "start_compound": "SOFT", "start_wear": 0.0, "start_gap": 5.0,
                "pit_stops": [{"lap": 6, "compound": "MEDIUM"}], "finish_gap": 0.0},
        "VER": {"pace_offset": 0.0, "start_compound": "SOFT", "start_wear": 0.0, "start_gap": 0.0,
                "pit_stops": [{"lap": 5, "compound": "MEDIUM"}], "finish_gap": 3.0},
    },
}


@pytest.fixture
def calibration() -> dict:
    return copy.deepcopy(CALIBRATION)


@pytest.fixture
def scenario(calibration: dict) -> Scenario:
    return Scenario(calibration)


# Parameters the calibration test should recover from synthetic_laps.
TRUE = {
    "fuel_effect": -0.05,
    "compounds": {"SOFT": (0.0, 0.08), "MEDIUM": (0.4, 0.04)},  # (pace offset, degradation)
    "pit_loss": 20.0,
    "dirty_air_loss": 0.5,
}


@pytest.fixture
def synthetic_laps() -> tuple[pd.DataFrame, dict]:
    """A 30-lap race generated from TRUE, with dirty air applied to close followers."""
    rng = np.random.default_rng(0)
    drivers = {  # pace, start gap, stops {lap: compound}
        "AAA": (-0.5, 0.0, {22: "MEDIUM"}),
        "TRL": (-1.0, 1.0, {14: "MEDIUM"}),  # faster, but loses 0.5 s following AAA
        "BBB": (0.3, 30.0, {15: "MEDIUM", 25: "SOFT"}),
        "CCC": (-0.2, 60.0, {12: "MEDIUM"}),
        "DDD": (0.0, 90.0, {18: "MEDIUM"}),
    }
    total = 30
    state = {d: {"time": gap, "compound": "SOFT", "life": 0, "stint": 1}
             for d, (_, gap, _) in drivers.items()}
    rows = []
    for lap in range(1, total + 1):
        ends = {}
        for d, (pace, _, stops) in drivers.items():
            s = state[d]
            s["life"] += 1
            offset, deg = TRUE["compounds"][s["compound"]]
            t = 80 + pace + offset + deg * s["life"] + TRUE["fuel_effect"] * lap + rng.normal(0, 0.05)
            pit_in = lap in stops
            pit_out = lap - 1 in stops
            t += 8.0 * pit_in + 12.0 * pit_out
            ends[d] = (s["time"] + t, t, pit_in, pit_out, s["compound"], s["life"], s["stint"])
        order = sorted(ends, key=lambda d: ends[d][0])
        for prev, d in zip([None] + order, order):
            end, t, pit_in, pit_out, compound, life, stint = ends[d]
            if prev is not None and end - ends[prev][0] <= 2.0:
                end += TRUE["dirty_air_loss"]
                t += TRUE["dirty_air_loss"]
            rows.append({
                "Driver": d, "LapNumber": float(lap), "Stint": float(stint), "Compound": compound,
                "TyreLife": float(life), "FreshTyre": True, "Position": 0.0, "TrackStatus": "1",
                "IsAccurate": True, "Deleted": False,
                "PitInTime": 1.0 if pit_in else np.nan,
                "PitOutTime": 1.0 if pit_out else np.nan,
                "LapTimeSeconds": t, "SessionTimeSeconds": end,
            })
            state[d]["time"] = end
        for d, (_, _, stops) in drivers.items():
            if lap in stops:
                state[d].update(compound=stops[lap], life=0, stint=state[d]["stint"] + 1)
    race = {"year": 2000, "event": "Synthetic Grand Prix", "total_laps": total}
    laps = pd.DataFrame(rows)
    for col in ("PitInTime", "PitOutTime"):
        laps[col] = pd.to_timedelta(laps[col], unit="s")
    return laps, race
