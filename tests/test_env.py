import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from f1_mlops.env import REWARD_SCALE, RULE_PENALTY, BatchEnv, Episode, Scenario, StrategyEnv


def test_passes_gymnasium_checks(scenario: Scenario) -> None:
    check_env(StrategyEnv(scenario), skip_render_check=True)


def test_actions_come_from_calibrated_compounds(scenario: Scenario) -> None:
    assert scenario.actions == ["STAY", "PUSH", "SAVE", "PIT_SOFT", "PIT_MEDIUM"]


def test_real_plan_uses_simulated_lap_indices(scenario: Scenario) -> None:
    # Race lap 6 is the 5th simulated lap (index 4): the simulation starts on lap 2.
    assert scenario.real_plan("HAM") == {4: "PIT_MEDIUM"}


def test_return_is_minus_race_time(scenario: Scenario) -> None:
    env = StrategyEnv(scenario)
    env.reset(seed=0)
    total, steps, done = 0.0, 0, False
    while not done:
        action = 4 if steps == 3 else 0  # one stop for mediums
        _, reward, done, _, info = env.step(action)
        total += reward
        steps += 1
    assert steps == scenario.total_laps
    start_gap = scenario.calibration["drivers"]["HAM"]["start_gap"]
    race_time = info["race_time"] - start_gap
    assert total == pytest.approx((10 * 80.0 - race_time) * REWARD_SCALE)


def test_one_compound_race_is_penalised(scenario: Scenario) -> None:
    def last_reward(first_action: int) -> float:
        ep = Episode(scenario, 0)
        ep.step(first_action)
        for _ in range(scenario.total_laps - 2):
            ep.step(0)
        reward, done = ep.step(0)
        assert done
        return reward / REWARD_SCALE - (80.0 - ep.race.driver("HAM").last_lap_time)

    assert last_reward(0) == pytest.approx(-RULE_PENALTY)  # softs only
    assert last_reward(3) == pytest.approx(-RULE_PENALTY)  # softs -> softs
    assert last_reward(4) == pytest.approx(0.0)  # softs -> mediums


def test_batch_env_restarts_finished_races(scenario: Scenario) -> None:
    env = BatchEnv(scenario, n=3, seed=0)
    obs = env.reset()
    assert obs.shape == (3, scenario.obs_size)
    for lap in range(scenario.total_laps):
        obs, rewards, done = env.step(np.zeros(3, dtype=int))
        assert done == (lap == scenario.total_laps - 1)
    assert len(env.final_info) == 3
    assert all(ep.race.lap == 0 for ep in env.episodes)


def test_missing_calibration_is_a_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="f1-mlops calibrate"):
        Scenario.load(tmp_path / "sim_config.json")
