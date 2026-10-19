"""Compare strategies on the same simulated races (same seeds): the driver's real strategy,
the best fixed pit plans from an exhaustive search (with and without push/save tyre
management), and a trained policy."""

from collections.abc import Callable, Iterable
from itertools import combinations, product

import numpy as np

from f1_mlops.env import Episode, Scenario
from f1_mlops.train import Policy

Chooser = Callable[[np.ndarray, Episode], int]
Plan = dict[int, str]  # simulated lap index -> action; other laps STAY


def plan_chooser(scenario: Scenario, plan: Plan) -> Chooser:
    index = {a: i for i, a in enumerate(scenario.actions)}
    return lambda obs, ep: index[plan.get(ep.race.lap, "STAY")]


def policy_chooser(policy: Policy) -> Chooser:
    return lambda obs, ep: policy.act(obs)


def run_race(scenario: Scenario, choose: Chooser, seed: int, trace: bool = False) -> dict:
    """Race to the finish; with trace, also return per-lap states under "laps"."""
    ep = Episode(scenario, seed)
    obs = ep.obs()
    laps = []
    while not ep.race.finished:
        action = choose(obs, ep)
        ep.step(action)
        obs = ep.obs()
        if trace:
            hero, rival = ep.race.driver(scenario.hero), ep.race.driver(scenario.rival)
            laps.append({
                "lap": ep.race.lap - 1 + scenario.first_lap,
                "action": scenario.actions[action],
                "lap_time": hero.last_lap_time,
                "rival_lap_time": rival.last_lap_time,
                "gap_to_rival": hero.total_time - rival.total_time,
                "compound": hero.compound,
                "rival_compound": rival.compound,
                "wear": hero.wear,
            })
    return {**ep.info(), "laps": laps} if trace else ep.info()


def summarize(results: list[dict]) -> dict:
    times = np.array([r["race_time"] for r in results])
    gaps = np.array([r["gap_to_rival"] for r in results])
    return {
        "mean_race_time": float(times.mean()),
        "sd_race_time": float(times.std()),
        "mean_gap_to_rival": float(gaps.mean()),
        "win_rate": float((gaps < 0).mean()),
        "rule_violations": int(sum(r["compounds_used"] < 2 for r in results)),
    }


def describe_plan(scenario: Scenario, plan: Plan) -> str:
    stops = [f"lap {k + scenario.first_lap} -> {a.removeprefix('PIT_')}"
             for k, a in sorted(plan.items()) if a.startswith("PIT_")]
    managed = any(a in ("PUSH", "SAVE") for a in plan.values())
    return (", ".join(stops) or "no stops") + (" (+ push/save)" if managed else "")


def with_tyre_management(scenario: Scenario, plan: Plan) -> Plan:
    """Add the drive mode with the lowest total cost on each lap of a pit plan.

    One unit of wear on a lap costs `degradation` on that lap and every later lap of the
    stint, so a mode's cost is pace_delta + degradation * wear * laps_left_in_stint.
    Traffic is ignored.
    """
    from f1_mlops.env import ASSUMPTIONS

    deg = {c["name"]: c["degradation"] for c in scenario.calibration["compounds"]}
    modes = {"STAY": (0.0, 1.0)} | {
        m.upper(): (ASSUMPTIONS[m]["pace_delta"], ASSUMPTIONS[m]["wear"]) for m in ("push", "save")
    }
    compound = scenario.calibration["drivers"][scenario.hero]["start_compound"]
    stint_ends = sorted(plan) + [scenario.total_laps - 1]
    full: Plan = {}
    for lap in range(scenario.total_laps):
        if lap in plan:
            full[lap] = plan[lap]
            compound = plan[lap].removeprefix("PIT_")
            continue
        left = next(e for e in stint_ends if e >= lap) - lap + 1
        full[lap] = min(modes, key=lambda m: modes[m][0] + deg[compound] * modes[m][1] * left)
    return full


def fixed_plans(scenario: Scenario, max_stops: int = 2) -> Iterable[Plan]:
    """Every plan of up to max_stops stops that uses two compounds."""
    start = scenario.calibration["drivers"][scenario.hero]["start_compound"]
    laps = range(scenario.total_laps - 1)  # a stop on the last lap only loses time
    for stops in range(1, max_stops + 1):
        for when in combinations(laps, stops):
            for compounds in product(scenario.compounds, repeat=stops):
                if set(compounds) != {start}:
                    yield {k: f"PIT_{c}" for k, c in zip(when, compounds)}


def search_fixed_plans(scenario: Scenario, seeds: Iterable[int], manage_tyres: bool = False,
                       max_stops: int = 2, shortlist: int = 20) -> list[tuple[Plan, float]]:
    """Rank all fixed plans on a noise-free race, then the shortlist on noisy races."""
    calm = Scenario({**scenario.calibration, "lap_noise_sd": 0.0}, scenario.hero, scenario.rival)
    plans = fixed_plans(scenario, max_stops)
    if manage_tyres:
        plans = (with_tyre_management(scenario, plan) for plan in plans)
    ranked = sorted(plans,
                    key=lambda plan: run_race(calm, plan_chooser(calm, plan), 0)["race_time"])
    seeds = list(seeds)
    scored = [
        (plan, float(np.mean([run_race(scenario, plan_chooser(scenario, plan), s)["race_time"]
                              for s in seeds])))
        for plan in ranked[:shortlist]
    ]
    return sorted(scored, key=lambda x: x[1])


def evaluate(scenario: Scenario, policy: Policy | None, seeds: Iterable[int],
             search_seeds: Iterable[int]) -> dict:
    seeds = list(seeds)
    search_seeds = list(search_seeds)
    best = search_fixed_plans(scenario, search_seeds)[0][0]
    best_managed = search_fixed_plans(scenario, search_seeds, manage_tyres=True)[0][0]
    real = scenario.real_plan(scenario.hero)
    strategies: dict[str, tuple[str, Chooser]] = {
        "real": (describe_plan(scenario, real), plan_chooser(scenario, real)),
        "best_fixed": (describe_plan(scenario, best), plan_chooser(scenario, best)),
        "best_fixed_managed": (describe_plan(scenario, best_managed),
                               plan_chooser(scenario, best_managed)),
    }
    if policy is not None:
        strategies["ppo"] = ("trained policy", policy_chooser(policy))
    return {
        name: {"strategy": desc, **summarize([run_race(scenario, choose, s) for s in seeds])}
        for name, (desc, choose) in strategies.items()
    }
