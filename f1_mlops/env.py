"""Race-strategy environments: one driver's decisions each lap against a rival who repeats
their real strategy from the calibrated race.

Reward is minus the driver's lap time (relative to the base lap, scaled by REWARD_SCALE), so
the return is minus the race time: pit stops, traffic and tyre wear all cost time directly.
Finishing without using two dry compounds, as the rules require, costs RULE_PENALTY seconds.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from f1_mlops._sim import Race

# Not calibrated from data: drive modes and the traffic model are assumptions.
ASSUMPTIONS = {
    # Push: 0.3 s faster for 50% more wear. Save: 0.3 s slower for 40% less wear.
    "push": {"pace_delta": -0.3, "wear": 1.5},
    "save": {"pace_delta": 0.3, "wear": 0.6},
    # A car must be 1.0 s faster over the lap to pass; otherwise it follows 0.5 s behind.
    "overtake_delta": 1.0,
    "follow_gap": 0.5,
}
RULE_PENALTY = 60.0
REWARD_SCALE = 0.1
BASE_ACTIONS = ["STAY", "PUSH", "SAVE"]


@dataclass(frozen=True)
class Scenario:
    """A calibrated race (sim_config.json) and the two drivers being simulated."""

    calibration: dict
    hero: str = "HAM"
    rival: str = "VER"

    def __post_init__(self) -> None:
        for d in (self.hero, self.rival):
            if d not in self.calibration["drivers"]:
                raise ValueError(f"no calibration for driver {d!r}")

    @classmethod
    def load(cls, path: Path, hero: str = "HAM", rival: str = "VER") -> "Scenario":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `f1-mlops calibrate` first")
        return cls(json.loads(path.read_text()), hero, rival)

    @property
    def total_laps(self) -> int:
        return self.calibration["total_laps"]

    @property
    def first_lap(self) -> int:
        """Race lap number of the first simulated lap."""
        return self.calibration["first_lap"]

    @cached_property
    def compounds(self) -> list[str]:
        return [c["name"] for c in self.calibration["compounds"]]

    @cached_property
    def actions(self) -> list[str]:
        return BASE_ACTIONS + [f"PIT_{c}" for c in self.compounds]

    @cached_property
    def sim_json(self) -> str:
        cal = self.calibration
        start = {d: cal["drivers"][d] for d in (self.hero, self.rival)}
        leader_gap = min(s["start_gap"] for s in start.values())
        config = {
            **{k: cal[k] for k in ("total_laps", "base_lap_time", "fuel_effect", "pit_loss",
                                   "lap_noise_sd", "dirty_air_gap", "dirty_air_loss",
                                   "compounds")},
            **ASSUMPTIONS,
            "drivers": [
                {"id": d, "pace_offset": s["pace_offset"], "start_compound": s["start_compound"],
                 "start_wear": s["start_wear"], "start_gap": s["start_gap"] - leader_gap}
                for d, s in start.items()
            ],
        }
        return json.dumps(config)

    def real_plan(self, driver: str) -> dict[int, str]:
        """The driver's real pit stops, keyed by simulated lap index."""
        return {
            stop["lap"] - self.first_lap: f"PIT_{stop['compound']}"
            for stop in self.calibration["drivers"][driver]["pit_stops"]
            if stop["lap"] >= self.first_lap and stop["compound"] in self.compounds
        }

    @property
    def obs_size(self) -> int:
        return 7 + len(self.compounds)


class Episode:
    """One race, stepped by action index; shared by both environments."""

    def __init__(self, scenario: Scenario, seed: int | None):
        self.scenario = scenario
        self.race = Race(scenario.sim_json, seed)
        self._rival_plan = scenario.real_plan(scenario.rival)
        self._compound_index = {c: i for i, c in enumerate(scenario.compounds)}
        self._base = scenario.calibration["base_lap_time"]

    def step(self, action: int) -> tuple[float, bool]:
        s = self.scenario
        lap = self.race.lap
        self.race.step({s.hero: s.actions[action], s.rival: self._rival_plan.get(lap, "STAY")})
        hero = self.race.driver(s.hero)
        reward = self._base - hero.last_lap_time
        done = self.race.finished
        if done and hero.compounds_used < 2:
            reward -= RULE_PENALTY
        return reward * REWARD_SCALE, done

    def obs(self, out: np.ndarray | None = None) -> np.ndarray:
        s = self.scenario
        hero, rival = self.race.driver(s.hero), self.race.driver(s.rival)
        out = np.zeros(s.obs_size, dtype=np.float32) if out is None else out
        out[0] = 1.0 - self.race.lap / s.total_laps
        out[1] = hero.wear / 30.0
        out[2] = float(hero.compounds_used >= 2)
        out[3] = hero.pit_stops / 3.0
        out[4] = np.tanh((hero.total_time - rival.total_time) / 10.0)  # > 0: behind the rival
        out[5] = rival.wear / 30.0
        out[6] = rival.pit_stops / 3.0
        out[7:] = 0.0
        out[7 + self._compound_index[hero.compound]] = 1.0
        return out

    def info(self) -> dict:
        s = self.scenario
        hero, rival = self.race.driver(s.hero), self.race.driver(s.rival)
        return {"race_time": hero.total_time, "gap_to_rival": hero.total_time - rival.total_time,
                "position": hero.position, "compounds_used": hero.compounds_used}


class StrategyEnv(gym.Env):
    """Gymnasium environment for one race."""

    def __init__(self, scenario: Scenario):
        super().__init__()
        self.scenario = scenario
        self.action_space = spaces.Discrete(len(scenario.actions))
        self.observation_space = spaces.Box(-1.0, 10.0, (scenario.obs_size,), np.float32)
        self.episode: Episode | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.episode = Episode(self.scenario, int(self.np_random.integers(2**63)))
        return self.episode.obs(), self.episode.info()

    def step(self, action):
        if self.episode is None:
            raise RuntimeError("call reset() first")
        reward, done = self.episode.step(int(action))
        return self.episode.obs(), reward, done, False, self.episode.info()


class BatchEnv:
    """`n` races stepped together, for training; finished races restart automatically.

    All races last the same number of laps, so they finish together.
    """

    def __init__(self, scenario: Scenario, n: int, seed: int = 0):
        self.scenario = scenario
        self.n = n
        self._rng = np.random.default_rng(seed)
        self._obs = np.zeros((n, scenario.obs_size), dtype=np.float32)
        self.episodes: list[Episode] = []
        self.final_info: list[dict] = []  # info of each race when the last batch finished

    def reset(self) -> np.ndarray:
        seeds = self._rng.integers(2**63, size=self.n)
        self.episodes = [Episode(self.scenario, int(s)) for s in seeds]
        for i, ep in enumerate(self.episodes):
            ep.obs(self._obs[i])
        return self._obs.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Returns (observations, rewards, done); after done the observations are new races'."""
        rewards = np.empty(self.n, dtype=np.float32)
        done = False
        for i, (ep, a) in enumerate(zip(self.episodes, actions)):
            rewards[i], done = ep.step(int(a))
            ep.obs(self._obs[i])
        if done:
            self.final_info = [ep.info() for ep in self.episodes]
            return self.reset(), rewards, True
        return self._obs.copy(), rewards, False
