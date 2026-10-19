"""f1-mlops: ingest a race, calibrate the simulator, train a strategy policy, evaluate it."""

import argparse
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="f1-mlops", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="download a race's laps with FastF1")
    p.add_argument("--year", type=int, default=2021)
    p.add_argument("--event", default="Spain")

    sub.add_parser("calibrate", help="fit the simulator to the ingested race")

    for name, help_text in [("train", "train a PPO policy"), ("evaluate", "compare strategies")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--hero", default="HAM")
        p.add_argument("--rival", default="VER")
        if name == "train":
            p.add_argument("--iterations", type=int, default=300)
            p.add_argument("--seed", type=int, default=0)
        else:
            p.add_argument("--races", type=int, default=200, help="simulated races per strategy")

    args = parser.parse_args(argv)
    if args.command == "ingest":
        from f1_mlops.ingest import fetch_race

        laps = fetch_race(args.year, args.event, args.data_dir, args.data_dir / "cache")
        print(f"Saved {len(laps)} laps to {args.data_dir / 'laps.parquet'}")
    elif args.command == "calibrate":
        from f1_mlops.calibrate import calibrate_files

        config = calibrate_files(args.data_dir)
        print(json.dumps({k: v for k, v in config.items() if k != "drivers"}, indent=2))
    elif args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)


def _scenario(args: argparse.Namespace):
    from f1_mlops.env import Scenario

    return Scenario.load(args.data_dir / "sim_config.json", args.hero, args.rival)


def _train(args: argparse.Namespace) -> None:
    import torch

    from f1_mlops.train import TrainConfig, save_history, save_policy, train

    # More threads than this only add synchronisation for a 64-unit network.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    scenario = _scenario(args)
    cfg = TrainConfig(iterations=args.iterations, seed=args.seed)
    policy, history = train(scenario, cfg)
    save_policy(args.models_dir / "policy.pt", policy, scenario, cfg)
    save_history(args.models_dir / "training_history.json", history, cfg)
    print(f"Saved {args.models_dir / 'policy.pt'}")


def _evaluate(args: argparse.Namespace) -> None:
    from f1_mlops.evaluate import evaluate
    from f1_mlops.train import load_policy

    scenario = _scenario(args)
    policy_path = args.models_dir / "policy.pt"
    policy = load_policy(policy_path, scenario) if policy_path.exists() else None
    if policy is None:
        print(f"No {policy_path}; comparing fixed strategies only.")
    # Evaluation seeds are disjoint from the plan-search seeds.
    results = evaluate(scenario, policy, seeds=range(10_000, 10_000 + args.races),
                       search_seeds=range(50))
    drivers = scenario.calibration["drivers"]
    real_gap = drivers[scenario.hero]["finish_gap"] - drivers[scenario.rival]["finish_gap"]
    report = {"race": scenario.calibration["race"], "hero": scenario.hero,
              "rival": scenario.rival, "races": args.races,
              "real_gap_to_rival": real_gap, "strategies": results}
    args.models_dir.mkdir(parents=True, exist_ok=True)
    (args.models_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"{scenario.hero} vs {scenario.rival}, {args.races} simulated races "
          f"(gap < 0: {scenario.hero} ahead)\n")
    print("| Strategy | Race time (s) | Gap to rival (s) | Plan |\n|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['mean_race_time']:.1f} ± {r['sd_race_time']:.1f} | "
              f"{r['mean_gap_to_rival']:+.1f} | {r['strategy']} |")
    print(f"\nReal race: gap {real_gap:+.1f} s. The 'real' row replays both drivers' real "
          "strategies, so the difference is the simulator's error.")


if __name__ == "__main__":
    main()
