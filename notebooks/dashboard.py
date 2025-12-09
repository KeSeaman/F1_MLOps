import marimo

__generated_with = "0.8.0"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import numpy as np
    import sys
    import os
    
    # Add backend to path if not installed
    # sys.path.append("../backend") 
    # But we installed it via maturin develop
    
    from f1_sim_backend import F1Env, RaceConfig, DriverState, TyreCompound
    return F1Env, RaceConfig, DriverState, TyreCompound, mo, np, os, pd, px, sys

@app.cell
def __(mo):
    mo.md(r"""# 🏎️ The Undercut Oracle: Live Strategy Dashboard""")
    return

@app.cell
def __(mo):
    # Load Environment and Agent
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from env import F1GymEnv
    import os
    
    # Init Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level='ERROR', include_dashboard=False)
        
    # Recreate Config to restore agent
    # Note: In real app, load from checkpoint config
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env=F1GymEnv)
        .framework("torch")
        .env_runners(num_env_runners=1)
        .training(model={"fcnet_hiddens": [64, 64]})
    )
    
    algo = config.build_algo()
    
    # Try to restore checkpoint
    checkpoint_path = "models/ppo_f1"
    model_status = "❌ Model not found (using random policy)"
    if os.path.exists(checkpoint_path):
        try:
            algo.restore(checkpoint_path)
            model_status = "✅ PPO Agent Loaded"
        except Exception as e:
            model_status = f"⚠️ Load failed: {e}"
            
    env = F1GymEnv()
    obs, _ = env.reset()
    
    return F1GymEnv, PPOConfig, algo, config, env, model_status, obs, os, ray

@app.cell
def __(model_status, mo):
    mo.md(f"**Status:** {model_status}")
    return

@app.cell
def __(mo):
    step_btn = mo.ui.button(label="🏁 Race Step")
    reset_btn = mo.ui.button(label="🔄 Reset Race")
    return reset_btn, step_btn

@app.cell
def __(algo, env, mo, obs, reset_btn, step_btn):
    # State mapping
    # Marimo cells are reactive. We need to manage state carefully.
    # For a simple dashboard, we'll just run one step per click in this cell context 
    # but Marimo statelessness makes "looping" hard without `mo.state`.
    # Let's use static globals for this simple demo or re-instantiate.
    # Better: Use mo.get_state for persistent simulation.
    
    get_state, set_state = mo.state(None)
    
    current_obs = get_state() if get_state() is not None else obs
    
    if reset_btn.value:
        current_obs, _ = env.reset()
        set_state(current_obs)
        message = "Race Reset."
    elif step_btn.value:
        # Compute Action
        action = algo.compute_single_action(current_obs)
        
        # Step Env
        current_obs, reward, done, _, info = env.step(action)
        set_state(current_obs)
        
        action_map = {0: "StayOut", 1: "PUSH", 2: "SAVE", 3: "PIT_SOFT", 4: "PIT_MEDIUM", 5: "PIT_HARD"}
        message = f"Action: **{action_map[action]}** | Reward: {reward:.2f}"
            
        if done:
            message += " | 🏁 RACE FINISHED!"
    else:
        message = "Ready to race."

    return action_map, step_btn, current_obs, get_state, message, set_state

@app.cell
def __(env, message, mo):
    mo.md(message)
    
    # Visualize State
    # Fetch internal state from env (cheating a bit for dashboard)
    drivers = env.sim.get_drivers()
    hero = drivers["HAM"]
    rival = drivers["VER"]
    
    stat_list = [
        {"Metric": "Lap", "Value": hero.lap_number},
        {"Metric": "Position", "Value": hero.position},
        {"Metric": "Gap to Leader", "Value": f"{hero.gap_to_leader:.2f}s"},
        {"Metric": "Tyre Age", "Value": hero.tyre_age},
        {"Metric": "Tyre Compound", "Value": str(hero.tyre_compound).split('.')[-1]},
        {"Metric": "Last Lap", "Value": f"{hero.last_lap_time:.2f}s"},
    ]
    
    import pandas as pd
    df = pd.DataFrame(stat_list)
    
    return df, drivers, hero, rival

@app.cell
def __(df, mo):
    mo.ui.table(df)
    return

