import json
from pathlib import Path

import pytest

from f1_mlops import cli
from f1_mlops.env import Scenario
from f1_mlops.evaluate import fixed_plans, run_race, plan_chooser, with_tyre_management
from f1_mlops.train import TrainConfig, load_policy, save_policy, train


def test_fixed_plans_all_use_two_compounds(scenario: Scenario) -> None:
    plans = list(fixed_plans(scenario, max_stops=2))
    # 1 stop: 9 laps x MEDIUM. 2 stops: C(9, 2) laps x 3 compound pairs that include MEDIUM.
    assert len(plans) == 9 + 36 * 3
    assert all(any(a == "PIT_MEDIUM" for a in p.values()) for p in plans)


def test_tyre_management_saves_early_and_pushes_late(scenario: Scenario) -> None:
    plan = with_tyre_management(scenario, {})
    # SOFT degradation 0.2: saving pays with > 3.75 laps left, pushing with < 3 left.
    assert plan[0] == "SAVE" and plan[9] == "PUSH"


def test_trace_has_one_row_per_lap(scenario: Scenario) -> None:
    result = run_race(scenario, plan_chooser(scenario, {3: "PIT_MEDIUM"}), 0, trace=True)
    assert [row["lap"] for row in result["laps"]] == list(range(2, 12))
    assert result["laps"][3]["action"] == "PIT_MEDIUM"


def test_policy_round_trip(scenario: Scenario, calibration: dict, tmp_path: Path) -> None:
    policy, history = train(scenario, TrainConfig(iterations=2, n_envs=4), log=lambda s: None)
    assert len(history) == 2
    save_policy(tmp_path / "policy.pt", policy, scenario, TrainConfig())
    loaded = load_policy(tmp_path / "policy.pt", scenario)
    obs = run_race(scenario, plan_chooser(scenario, {}), 0)  # smoke: scenario still runs
    assert obs["race_time"] > 0 and loaded.act is not None
    with pytest.raises(ValueError, match="different scenario"):
        load_policy(tmp_path / "policy.pt", Scenario(calibration, hero="VER", rival="HAM"))


def test_cli_calibrate_train_evaluate(synthetic_laps, tmp_path: Path) -> None:
    laps, race = synthetic_laps
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    laps.to_parquet(data / "laps.parquet")
    (data / "race.json").write_text(json.dumps(race))
    dirs = ["--data-dir", str(data), "--models-dir", str(models)]

    cli.main(dirs + ["calibrate"])
    cli.main(dirs + ["train", "--hero", "BBB", "--rival", "AAA", "--iterations", "2"])
    cli.main(dirs + ["evaluate", "--hero", "BBB", "--rival", "AAA", "--races", "3"])

    report = json.loads((models / "evaluation.json").read_text())
    assert set(report["strategies"]) == {"real", "best_fixed", "best_fixed_managed", "ppo"}
    assert isinstance(report["real_gap_to_rival"], float)
