"""Fit the simulator's parameters to a race's laps.

Model, fitted by least squares on clean green-flag, free-air laps from every driver:

    lap_time = driver_pace[driver] + compound_offset[compound]
               + degradation[compound] * tyre_life + fuel_effect * lap_number

Cars get faster as fuel burns, which a tyre-only fit mistakes for negative degradation, so
lap number is its own term. (Replacing it with a fixed effect per lap, which also absorbs
track evolution, changes the compound estimates for Spain 2021 by < 0.07 s.)

Laps within DIRTY_AIR_GAP of the car in front are left out of the fit and instead give the
time lost per lap by following closely. Pit loss is measured on green-flag stops as the
in-lap plus out-lap time beyond what the model predicts for those two laps.

The simulation starts at the end of lap 1 (standing starts are not modelled), from the real
running order, gaps and tyre wear at that point.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
MIN_COMPOUND_LAPS = 30
MIN_DRIVER_LAPS = 10
MIN_PIT_STOPS = 3
# Laps slower than this multiple of the driver's median clean lap are traffic or incidents.
SLOW_LAP_FACTOR = 1.05
# Laps ending closer than this behind another car are in dirty air or a DRS train, where lap
# time reflects the car ahead rather than tyres. The model is fitted on free-air laps; the
# dirty-air laps give the time lost by following.
DIRTY_AIR_GAP = 2.0


def green_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Representative green-flag laps, with GapAhead: seconds behind the car in front."""
    order = laps.sort_values(["LapNumber", "SessionTimeSeconds"])
    gap_ahead = order.groupby("LapNumber")["SessionTimeSeconds"].diff().reindex(laps.index)
    ok = (
        (laps["TrackStatus"] == "1")
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & (laps["LapNumber"] > 1)
        & laps["IsAccurate"]
        & ~laps["Deleted"].fillna(False).astype(bool)
        & laps["LapTimeSeconds"].notna()
        & laps["Compound"].isin(DRY_COMPOUNDS)
    )
    green = laps[ok].assign(GapAhead=gap_ahead[ok])
    median = green.groupby("Driver")["LapTimeSeconds"].transform("median")
    return green[green["LapTimeSeconds"] < SLOW_LAP_FACTOR * median]


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Free-air green laps, for compounds and drivers with enough of them."""
    green = green_laps(laps)
    clean = green[~(green["GapAhead"] <= DIRTY_AIR_GAP)]  # the leader has no car ahead (NaN)
    counts = clean.groupby("Compound")["LapTimeSeconds"].transform("size")
    clean = clean[counts >= MIN_COMPOUND_LAPS]
    counts = clean.groupby("Driver")["LapTimeSeconds"].transform("size")
    return clean[counts >= MIN_DRIVER_LAPS]


class LapModel:
    """The least-squares fit described in the module docstring."""

    def __init__(self, clean: pd.DataFrame):
        self.drivers = sorted(clean["Driver"].unique())
        self.compounds = sorted(clean["Compound"].unique())  # the first is the reference
        x = self._design(clean)
        y = clean["LapTimeSeconds"].to_numpy()
        self.coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        self.residuals = y - x @ self.coef

    def _design(self, laps: pd.DataFrame) -> np.ndarray:
        n_d, n_c = len(self.drivers), len(self.compounds)
        x = np.zeros((len(laps), n_d + (n_c - 1) + n_c + 1))
        d = laps["Driver"].map({k: i for i, k in enumerate(self.drivers)}).to_numpy()
        c = laps["Compound"].map({k: i for i, k in enumerate(self.compounds)}).to_numpy()
        rows = np.arange(len(laps))
        x[rows, d] = 1.0
        has_offset = c > 0
        x[rows[has_offset], n_d + c[has_offset] - 1] = 1.0
        x[rows, n_d + n_c - 1 + c] = laps["TyreLife"].to_numpy()
        x[:, -1] = laps["LapNumber"].to_numpy()
        return x

    def predict(self, laps: pd.DataFrame) -> np.ndarray:
        return self._design(laps) @ self.coef

    @property
    def driver_pace(self) -> dict[str, float]:
        return dict(zip(self.drivers, self.coef[: len(self.drivers)]))

    @property
    def compound_offsets(self) -> dict[str, float]:
        """Relative to the reference compound (may be negative)."""
        n_d = len(self.drivers)
        return {c: (0.0 if i == 0 else float(self.coef[n_d + i - 1]))
                for i, c in enumerate(self.compounds)}

    @property
    def degradation(self) -> dict[str, float]:
        start = len(self.drivers) + len(self.compounds) - 1
        return dict(zip(self.compounds, map(float, self.coef[start: start + len(self.compounds)])))

    @property
    def fuel_effect(self) -> float:
        return float(self.coef[-1])

    @property
    def noise_sd(self) -> float:
        """Robust standard deviation of the residuals (scaled median absolute deviation)."""
        return float(1.4826 * np.median(np.abs(self.residuals - np.median(self.residuals))))


def pit_losses(laps: pd.DataFrame, model: LapModel) -> list[float]:
    """In-lap + out-lap time beyond the model's prediction, for green-flag stops."""
    losses = []
    known = laps[laps["Driver"].isin(model.drivers) & laps["Compound"].isin(model.compounds)]
    by_lap = known.set_index(["Driver", "LapNumber"])
    for row in known[known["PitInTime"].notna()].itertuples():
        out_key = (row.Driver, row.LapNumber + 1)
        if out_key not in by_lap.index:
            continue
        pair = by_lap.loc[[(row.Driver, row.LapNumber), out_key]].reset_index()
        if (pair["TrackStatus"] != "1").any() or pair["LapTimeSeconds"].isna().any():
            continue
        losses.append(float((pair["LapTimeSeconds"] - model.predict(pair)).sum()))
    return losses


def dirty_air_loss(laps: pd.DataFrame, model: LapModel) -> float:
    """Median time lost per lap within DIRTY_AIR_GAP of the car in front (0 if too few laps)."""
    green = green_laps(laps)
    dirty = green[(green["GapAhead"] <= DIRTY_AIR_GAP)
                  & green["Driver"].isin(model.drivers)
                  & green["Compound"].isin(model.compounds)]
    if len(dirty) < MIN_DRIVER_LAPS:
        return 0.0
    return float(np.median(dirty["LapTimeSeconds"] - model.predict(dirty)))


def driver_strategies(laps: pd.DataFrame, total_laps: int) -> dict[str, dict]:
    """Start state after lap 1, real pit stops and finishing gap for each driver."""
    lap1 = laps[laps["LapNumber"] == 1].set_index("Driver")
    leader_lap1 = lap1["SessionTimeSeconds"].min()
    finishers = laps[laps["LapNumber"] == total_laps].set_index("Driver")["SessionTimeSeconds"]
    out = {}
    for driver, dl in laps.groupby("Driver"):
        if driver not in lap1.index or pd.isna(lap1.at[driver, "SessionTimeSeconds"]):
            continue
        dl = dl.set_index("LapNumber")
        stops = [
            {"lap": int(lap), "compound": dl.at[lap + 1, "Compound"]}
            for lap in dl.index[dl["PitInTime"].notna()]
            if lap + 1 in dl.index and lap < total_laps
        ]
        out[driver] = {
            "start_compound": lap1.at[driver, "Compound"],
            "start_wear": float(lap1.at[driver, "TyreLife"]),
            "start_gap": float(lap1.at[driver, "SessionTimeSeconds"] - leader_lap1),
            "pit_stops": stops,
            "finish_gap": (float(finishers[driver] - finishers.min())
                           if driver in finishers.index else None),
        }
    return out


def calibrate(laps: pd.DataFrame, race: dict) -> dict:
    clean = clean_laps(laps)
    model = LapModel(clean)
    losses = pit_losses(laps, model)
    if len(losses) < MIN_PIT_STOPS:
        raise ValueError(f"only {len(losses)} green-flag pit stops; cannot estimate pit loss")

    first_lap = 2
    pace = model.driver_pace
    # Express pace relative to the field average on the fastest compound.
    offsets = model.compound_offsets
    fastest = min(offsets.values())
    field_pace = float(np.mean(list(pace.values()))) + fastest
    strategies = driver_strategies(laps, race["total_laps"])
    drivers = {
        d: {"pace_offset": float(pace[d] - field_pace), **strategies[d]}
        for d in model.drivers if d in strategies
    }
    excluded = sorted(set(laps["Compound"].dropna()) - set(model.compounds))
    return {
        "race": {k: race[k] for k in ("year", "event", "total_laps")},
        "first_lap": first_lap,
        "total_laps": race["total_laps"] - first_lap + 1,
        # Field-average pace on the fastest compound at the first simulated lap, new tyres.
        "base_lap_time": field_pace + model.fuel_effect * first_lap,
        "fuel_effect": model.fuel_effect,
        "pit_loss": float(np.median(losses)),
        "lap_noise_sd": model.noise_sd,
        "dirty_air_gap": DIRTY_AIR_GAP,
        "dirty_air_loss": dirty_air_loss(laps, model),
        "compounds": [
            {"name": c, "pace_offset": offsets[c] - fastest,
             "degradation": model.degradation[c],
             "clean_laps": int((clean["Compound"] == c).sum())}
            for c in model.compounds
        ],
        "drivers": drivers,
        "fit": {
            "clean_laps": len(clean),
            "drivers": len(model.drivers),
            "residual_sd": model.noise_sd,
            "pit_stops_measured": len(losses),
            "compounds_without_enough_data": excluded,
        },
    }


def calibrate_files(data_dir: Path) -> dict:
    laps = pd.read_parquet(data_dir / "laps.parquet")
    race = json.loads((data_dir / "race.json").read_text())
    config = calibrate(laps, race)
    (data_dir / "sim_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config
