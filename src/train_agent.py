import os
import warnings

# Suppress deprecation warnings
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'
# Disable Ray metrics collection to prevent exporter timeouts
os.environ['RAY_DISABLE_METRICS_COLLECTION'] = '1'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Force CPU-only mode to prevent NVML warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from env import F1GymEnv
from ray.tune.registry import register_env

def train():
    if ray.is_initialized():
        ray.shutdown()
    # Initialize Ray with dashboard/metrics disabled to prevent connection errors
    # Exclude dashboard/metrics and ensure 0 GPUs
    ray.init(
        ignore_reinit_error=True, 
        logging_level='ERROR',
        include_dashboard=False,
        num_gpus=0,
        _system_config={
            "metrics_report_interval_ms": 1000000,
            "enable_autoscaler_v2": False
        }
    )
    
    register_env("f1_env", lambda config: F1GymEnv())
    
    # Use legacy API stack for compatibility (avoids new API deprecation warnings)
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment("f1_env")
        .framework("torch")
        .env_runners(
            num_env_runners=3,  # Enable multithreading with 3 workers
            num_cpus_per_env_runner=1
        )
        .training(
            model={"fcnet_hiddens": [64, 64]},
            lr=0.0001,  # Lower learning rate to stabilize training
            train_batch_size=2000,
            vf_clip_param=100.0, # Relax value function clipping
            grad_clip=0.5 # Add gradient clipping
        )
    )
    
    print("Building PPO algorithm with 3 parallel workers...")
    algo = config.build_algo()
    
    print("Starting training (5 iterations)...")
    results = []
    for i in range(5):
        result = algo.train()
        # Extract reward from result structure
        reward = result.get('env_runners', {}).get('episode_reward_mean', 
                 result.get('episode_reward_mean', 0))
        results.append(reward)
        print(f"Iteration {i+1}/5 | Mean Reward: {reward:.2f}")
        
    # Save model
    os.makedirs("models", exist_ok=True)
    checkpoint_dir = algo.save("models/ppo_f1")
    print(f"\nModel saved to: {checkpoint_dir}")
    
    # Save results in both text and JSON formats
    import json
    from datetime import datetime
    
    avg_reward = sum(results) / len(results) if results else 0
    
    # Structured JSON for Artefact.md
    training_data = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": "PPO",
        "iterations": 5,
        "network": "FC 64x64",
        "learning_rate": 0.0001,
        "train_batch_size": 2000,
        "num_workers": 3,
        "framework": "PyTorch",
        "per_iteration": [
            {"iteration": i+1, "mean_reward": round(r, 2)}
            for i, r in enumerate(results)
        ],
        "average_reward": round(avg_reward, 2),
    }
    
    with open("data/training_results.json", "w") as f:
        json.dump(training_data, f, indent=2)
    print("Results saved to data/training_results.json")
    
    # Also save human-readable text
    with open("data/training_results.txt", "w") as f:
        f.write("=== F1 PPO Training Results ===\n")
        for i, r in enumerate(results):
            f.write(f"Iteration {i+1}: Reward = {r:.2f}\n")
        f.write(f"\nAverage Reward: {avg_reward:.2f}\n")
    print("Results saved to data/training_results.txt")
    
    ray.shutdown()

if __name__ == "__main__":
    train()


