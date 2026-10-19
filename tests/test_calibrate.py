import pandas as pd
import pytest

from f1_mlops.calibrate import calibrate, clean_laps

from .conftest import TRUE


def test_recovers_the_generating_parameters(synthetic_laps: tuple[pd.DataFrame, dict]) -> None:
    laps, race = synthetic_laps
    config = calibrate(laps, race)
    compounds = {c["name"]: c for c in config["compounds"]}
    for name, (offset, deg) in TRUE["compounds"].items():
        assert compounds[name]["degradation"] == pytest.approx(deg, abs=0.01)
        assert compounds[name]["pace_offset"] == pytest.approx(offset, abs=0.1)  # SOFT fastest
    assert config["fuel_effect"] == pytest.approx(TRUE["fuel_effect"], abs=0.01)
    assert config["pit_loss"] == pytest.approx(TRUE["pit_loss"], abs=0.5)
    assert config["dirty_air_loss"] == pytest.approx(TRUE["dirty_air_loss"], abs=0.15)


def test_clean_laps_exclude_pit_and_dirty_air_laps(synthetic_laps) -> None:
    laps, _ = synthetic_laps
    clean = clean_laps(laps)
    assert clean["PitInTime"].isna().all() and clean["PitOutTime"].isna().all()
    trl = clean[clean["Driver"] == "TRL"]["LapNumber"]
    assert trl.min() > 14  # TRL follows AAA within 2 s until its stop


def test_strategies_and_start_state(synthetic_laps) -> None:
    laps, race = synthetic_laps
    drivers = calibrate(laps, race)["drivers"]
    assert drivers["BBB"]["pit_stops"] == [{"lap": 15, "compound": "MEDIUM"},
                                           {"lap": 25, "compound": "SOFT"}]
    assert drivers["AAA"]["start_gap"] == 0.0
    # 1 s behind at the start; 0.5 s faster but 0.5 s lost in dirty air on lap 1.
    assert drivers["TRL"]["start_gap"] == pytest.approx(1.0, abs=0.2)
