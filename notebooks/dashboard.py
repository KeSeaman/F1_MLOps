import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import marimo as mo
    import pandas as pd
    import plotly.express as px

    from f1_mlops.env import Scenario
    from f1_mlops.evaluate import plan_chooser, policy_chooser, run_race, search_fixed_plans
    from f1_mlops.train import load_policy
    return (Scenario, json, load_policy, mo, pd, plan_chooser, policy_chooser, px, run_race,
            search_fixed_plans)


@app.cell
def _(Scenario, json, load_policy, mo):
    root = mo.notebook_dir().parent
    scenario = Scenario.load(root / "data" / "sim_config.json")
    policy = load_policy(root / "models" / "policy.pt", scenario)
    evaluation_path = root / "models" / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
    race = scenario.calibration["race"]
    mo.md(
        f"""
        # Pit strategy: {scenario.hero} vs {scenario.rival}, {race['year']} {race['event']}

        The simulator is calibrated from the real race's laps; {scenario.rival} repeats their real
        strategy. Compare the trained policy with {scenario.hero}'s real strategy and the best fixed
        pit plan (with push/save tyre management) found by exhaustive search.
        """
    )
    return evaluation, policy, scenario


@app.cell
def _(evaluation, mo, pd):
    if evaluation is None:
        table = mo.md("Run `f1-mlops evaluate` for averages over many races.")
    else:
        table = mo.vstack([
            mo.md(f"### Averages over {evaluation['races']} simulated races "
                  f"(real gap: {evaluation['real_gap_to_rival']:+.1f} s)"),
            mo.ui.table(pd.DataFrame([
                {"strategy": name, "plan": r["strategy"],
                 "race time (s)": round(r["mean_race_time"], 1),
                 "gap to rival (s)": round(r["mean_gap_to_rival"], 1),
                 "win rate": r["win_rate"]}
                for name, r in evaluation["strategies"].items()
            ]), selection=None),
        ])
    table
    return


@app.cell
def _(mo, scenario, search_fixed_plans):
    with mo.status.spinner("Searching fixed pit plans..."):
        best_plan = search_fixed_plans(scenario, range(50), manage_tyres=True)[0][0]
    seed = mo.ui.slider(0, 999, value=0, label="Simulated race (seed)", full_width=True)
    seed
    return best_plan, seed


@app.cell
def _(best_plan, pd, plan_chooser, policy, policy_chooser, run_race, scenario, seed):
    strategies = {
        "trained policy": policy_chooser(policy),
        "real strategy": plan_chooser(scenario, scenario.real_plan(scenario.hero)),
        "best fixed plan": plan_chooser(scenario, best_plan),
    }
    runs = {name: run_race(scenario, choose, seed.value, trace=True)
            for name, choose in strategies.items()}
    laps = pd.concat(
        [pd.DataFrame(r["laps"]).assign(strategy=name) for name, r in runs.items()],
        ignore_index=True,
    )
    return laps, runs


@app.cell
def _(mo, runs, scenario):
    mo.hstack([
        mo.stat(f"{r['race_time'] / 60:.0f} min {r['race_time'] % 60:.1f} s", label=name,
                caption=f"P{r['position']}, {r['gap_to_rival']:+.1f} s to {scenario.rival}")
        for name, r in runs.items()
    ])
    return


@app.cell
def _(laps, px, scenario):
    gap = px.line(laps, x="lap", y="gap_to_rival", color="strategy",
                  labels={"gap_to_rival": f"Gap to {scenario.rival} (s, below 0 = ahead)"},
                  title=f"{scenario.hero}'s gap to {scenario.rival}")
    gap.add_hline(y=0, line_dash="dot", line_color="grey")
    gap
    return


@app.cell
def _(laps, px):
    # Pit laps are ~23 s slower; clip them so tyre wear stays visible.
    clipped = laps.assign(lap_time=laps["lap_time"].clip(upper=laps["lap_time"].median() + 4))
    times = px.line(clipped, x="lap", y="lap_time", color="strategy",
                    labels={"lap_time": "Lap time (s, pit laps clipped)"}, title="Lap times")
    times
    return


@app.cell
def _(laps, px):
    decisions = px.scatter(
        laps, x="lap", y="strategy", color="action", symbol="compound",
        category_orders={"action": ["STAY", "SAVE", "PUSH"]},
        title="Decisions each lap (colour) and tyre fitted after the lap (symbol)",
    )
    decisions.update_traces(marker_size=9)
    decisions
    return


if __name__ == "__main__":
    app.run()
