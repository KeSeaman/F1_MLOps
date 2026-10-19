# Implementation Notes

## Simulator (`backend/src/sim.rs`)

Each lap, for every driver:

```
lap_time = base_lap_time + driver_pace + compound_offset
         + degradation[compound] * wear + fuel_effect * laps_completed
         + mode_pace_delta + noise(sd = lap_noise_sd)
```

- **Wear** increases by 1 per normal lap, 1.5 when pushing and 0.6 when saving. The
  simulator also tracks tyre age in laps.
- **A pit stop** (`PIT_<COMPOUND>`) adds `pit_loss` to that lap and fits new tyres for the
  next lap.
- **Traffic.** Cars are processed in running order.
  - A car that starts the lap within `dirty_air_gap` of the car ahead loses `dirty_air_loss`.
  - A car that would finish the lap ahead of the car in front passes only if it was at least
    `overtake_delta` faster over the lap. Otherwise it finishes `follow_gap` behind.
- **Positions and gaps** come from total race time.
- **Noise** comes from a seeded `StdRng`, so a seed reproduces a race exactly.

The Python API is `Race(config_json, seed)` with `step({driver: action})`, `driver(id)`,
`standings()`, `lap` and `finished`. Unknown actions, compounds or drivers raise errors.

## Calibration (`f1_mlops/calibrate.py`)

**Clean laps**:
- green flag, not in- or out-laps, not lap 1, accurate, not deleted;
- within 5% of the driver's median;
- more than 2 s behind the car in front (free air).

**Model**: one least-squares fit across all drivers. The terms are a pace for each driver, a
pace offset and degradation rate for each compound, and a linear lap-number term for fuel.

**2021 Spanish GP results** (`data/sim_config.json`):

| Parameter | Value |
|---|---|
| Clean laps / drivers | 653 / 19 |
| Medium: offset, degradation | 0.00 s, 0.085 s/lap |
| Soft: offset, degradation | +0.42 s, 0.054 s/lap |
| Fuel effect | −0.044 s per lap completed |
| Pit loss (median of 34 green-flag stops) | 23.4 s |
| Dirty-air loss (median, within 2 s) | 0.53 s/lap |
| Lap noise (robust residual SD) | 0.51 s |
| HAM / VER pace vs field | −1.86 / −1.62 s/lap |

**Why the design changed:**

| Earlier issue | Effect | Fix |
|---|---|---|
| Fitted each compound alone, without a fuel term | Fuel burn made softs show negative wear | Separate fuel term |
| Fitted on all laps | Laps in DRS trains made tyres look slower | Fit on free-air laps |
| No dirty-air term | Replaying the real strategies, the simulator had HAM winning by 29 s (real: 15.8 s) | Dirty-air term: now 24.5 s |

The compound estimates were cross-checked with a fixed effect per lap in place of the linear
fuel term, which also absorbs track evolution. They change by less than 0.07 s.

## Environment (`f1_mlops/env.py`)

- **Scenario**: the hero (HAM by default) against a rival (VER) who repeats their real pit stops.
  Both start from their real lap-1 order, gap and tyre wear.
- **Actions**: `STAY`, `PUSH`, `SAVE`, plus `PIT_<compound>` for each compound with enough data.
- **Observation**:
  - fraction of the race left;
  - own tyre wear, pit stops, and whether two compounds have been used yet;
  - gap to the rival;
  - the rival's wear and pit stops;
  - own compound, one-hot.
- **Reward**: (base lap time − lap time) × 0.1, so the return is minus the race time. Finishing
  without using two dry compounds costs 60 s.
- **Two front ends**:
  - `StrategyEnv` is the standard Gymnasium environment and passes `check_env`.
  - `BatchEnv` steps many races together for training.

## Training (`f1_mlops/train.py`)

- **Algorithm**: PPO with separate 64×64 tanh actor and critic networks.
- **Batches**: 64 races per batch. Every race lasts 65 laps, so each rollout is exactly 64
  complete races, with no bootstrapping across episode boundaries.
- **Settings**: undiscounted GAE (λ 0.95), 4 epochs × 4 minibatches, learning rate 3e-4,
  entropy coefficient 0.01.
- **Default run**: 300 iterations, 1.25M simulated laps, about 15 s on 4 CPU threads.
  Convergence comes after about 150 iterations.

**Why not Ray RLlib.** Measured on this environment, for 416k simulated laps:

| Setup | Start-up | Training time | Throughput |
|---|---|---|---|
| Single-process PPO (this repo) | ~0 s | 4.9 s | ~85k laps/s |
| RLlib PPO, 4 env runners | 5.7 s | 34.3 s | 12k laps/s |
| RLlib PPO, no env runners | 7.1 s | 226.5 s | 1.8k laps/s |

The simulator is cheap and the network is tiny, so distributed sampling only adds overhead.
For the same reason, a GPU wouldn't help. Dropping Ray also cut the environment from 8 GB to
1.6 GB, together with CPU-only PyTorch.

## Evaluation (`f1_mlops/evaluate.py`)

Strategies are compared on the same 200 seeds:

- **real**: the hero's real pit stops.
- **best_fixed**: the best of every one- and two-stop plan that uses two compounds (6,112
  plans). Plans are ranked on a noise-free race, then the top 20 are re-scored on 50 noisy
  races.
- **best_fixed_managed**: the same search, with push/save chosen per lap by their total cost
  over the rest of the stint (ignoring traffic).
- **ppo**: the trained policy, always taking its most likely action.

## Dashboard (`notebooks/dashboard.py`)

`uv run marimo run notebooks/dashboard.py` shows:

- the evaluation table;
- a seed slider;
- for that race, under each strategy: the gap to the rival, lap times, and the decision taken
  each lap.
