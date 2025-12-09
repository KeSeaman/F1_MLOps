import ray
from ray.rllib.algorithms.ppo import PPOConfig
from env import F1GymEnv
from ray.tune.registry import register_env
import os
import shutil

def train():
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    
    register_env("f1_env", lambda config: F1GymEnv())
    
    config = (
        PPOConfig()
        .environment("f1_env")
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .training(model={"fcnet_hiddens": [64, 64]})
    )
    
    print("Building algorithm...")
    algo = config.build()
    
    print("Starting training...")
    for i in range(5): # Short training for demo
        result = algo.train()
        print(f"Iter: {i}, Reward: {result['env_runners']['episode_reward_mean']}")
        
    # Save model
    os.makedirs("models", exist_ok=True)
    checkpoint_dir = algo.save("models/ppo_f1")
    print(f"Model saved to {checkpoint_dir}")

if __name__ == "__main__":
    train()
