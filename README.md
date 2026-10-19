# F1 MLOps: Pit-Strategy RL on a Calibrated Race Simulator

A reinforcement-learning agent chooses Lewis Hamilton's pit stops and tyre management for the
2021 Spanish Grand Prix, racing Max Verstappen, who repeats his real strategy. The race simulator
is written in Rust and calibrated from the real race's lap data.

- **Data**: FastF1 lap timing for the 2021 Spanish GP (any race can be ingested).
- **Simulator** (Rust, PyO3): tyre wear by compound, fuel burn, pit loss, push/save modes,
  dirty air and overtaking difficulty. Runs ~330k simulated laps per second from Python.
- **Calibration**: one least-squares model across all drivers' clean laps, fitted on free air,
  with fuel separated from tyre wear.
- **Training**: PPO in a single process, about 15 s on a laptop CPU.
- **Evaluation**: the trained policy against Hamilton's real strategy and against an exhaustive
  search over all one- and two-stop plans.
- **Dashboard**: marimo + Plotly, replaying any simulated race lap by lap.

## Results

200 simulated races per strategy, same random seeds for each (`f1-mlops evaluate`):

| Strategy | Race time (s) | Gap to VER (s) |
|---|---|---|
| Hamilton's real strategy (Medium at laps 28 and 42) | 5378.7 ± 3.7 | −24.5 |
| Best fixed pit plan (Medium at laps 21 and 44) | 5373.5 ± 4.0 | −30.8 |
| Best fixed pit plan + computed push/save | 5363.6 ± 3.8 | −39.8 |
| **Trained PPO policy** | **5363.4 ± 3.7** | **−40.1** |

- The policy learns a two-stop strategy: Medium at about lap 17 and lap 42, saving tyres early in
  each stint and pushing in its last few laps. It matches the best plan found by search
  (5363.4 s vs 5363.6 s), without being told how tyres work, but does not beat it. With the
  rival on a fixed script, there is little to react to.
- Most of the gain over the real strategy comes from push/save tyre management, whose costs
  are assumptions rather than calibrated values (see [Limitations](#limitations)).
- **Simulator accuracy**: replaying both drivers' real strategies gives Hamilton a 24.5 s win;
  the real margin was 15.8 s. Same winner, but the simulator flatters Hamilton by ~9 s, which
  is within the uncertainty of his pace estimate (he spent most of the race in Verstappen's
  dirty air, leaving few free-air laps to fit).

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and a Rust toolchain. `uv sync` builds the Rust
extension and installs CPU-only PyTorch.

```bash
uv sync
uv run f1-mlops ingest          # 2021 Spanish GP by default: --year, --event
uv run f1-mlops calibrate       # -> data/sim_config.json
uv run f1-mlops train           # -> models/policy.pt (~15 s)
uv run f1-mlops evaluate        # -> models/evaluation.json and the table above
uv run marimo run notebooks/dashboard.py
uv run pytest
```

The calibrated config, trained policy and evaluation are committed, so the dashboard runs
without the first four steps.

## Project structure

```
backend/src/sim.rs     Race simulator (pure Rust)
backend/src/lib.rs     Python bindings: f1_mlops._sim.Race
f1_mlops/ingest.py     FastF1 -> data/laps.parquet, data/race.json
f1_mlops/calibrate.py  Lap-time model fit -> data/sim_config.json
f1_mlops/env.py        Scenario, Gymnasium environment, batched environment
f1_mlops/train.py      Single-process PPO
f1_mlops/evaluate.py   Strategy comparison and exhaustive plan search
f1_mlops/cli.py        f1-mlops command
notebooks/dashboard.py marimo dashboard
tests/                 Simulator, environment, calibration and pipeline tests
```

## Limitations

- **One rival, one race.** The other 18 cars are not simulated, and the rival's strategy is
  fixed, so the agent can't react to a rival's stops or provoke them.
- **Not calibrated:** push/save pace and wear (±0.3 s for 50% more / 40% less wear), and the
  1.0 s pace advantage needed to overtake. These are in `ASSUMPTIONS` in `f1_mlops/env.py`.
- **Simplified race:** no safety cars, weather, or starts (the simulation begins after lap 1).
  Tyre wear is linear, with no cliff.
- **Unusual tyre data:** this race's data shows Mediums 0.4 s/lap quicker than Softs when new,
  but wearing faster. This holds with per-lap fixed effects too, so it reflects the race, not
  a fitting artefact.

See [notes/Artefact.md](notes/Artefact.md) for the model, calibration results and design
decisions.
