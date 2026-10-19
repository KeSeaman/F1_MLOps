"""Train a pit-strategy policy with PPO in a single process.

The simulator runs ~300k laps/s from Python, so one process stepping a batch of races keeps
PPO fed without distributed sampling: no worker start-up and no inter-process copies. Every
race lasts the same number of laps, so each rollout is exactly one batch of complete races.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from f1_mlops.env import BatchEnv, Scenario


@dataclass
class TrainConfig:
    iterations: int = 300
    n_envs: int = 64
    hidden: int = 64
    lr: float = 3e-4
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0


class Policy(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 64):
        super().__init__()

        def mlp(out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(obs_size, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))

        self.actor, self.critic = mlp(n_actions), mlp(1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        """Most likely action, for evaluation."""
        return int(self.actor(torch.from_numpy(obs)).argmax())


def train(scenario: Scenario, cfg: TrainConfig,
          log: Callable[[str], None] = print) -> tuple[Policy, list[dict]]:
    torch.manual_seed(cfg.seed)
    env = BatchEnv(scenario, cfg.n_envs, cfg.seed)
    policy = Policy(scenario.obs_size, len(scenario.actions), cfg.hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    steps, n = scenario.total_laps, cfg.n_envs
    obs_buf = torch.zeros(steps, n, scenario.obs_size)
    act_buf = torch.zeros(steps, n, dtype=torch.long)
    logp_buf, rew_buf, val_buf = (torch.zeros(steps, n) for _ in range(3))

    history = []
    obs = env.reset()
    start = time.perf_counter()
    for iteration in range(1, cfg.iterations + 1):
        for t in range(steps):
            obs_t = torch.from_numpy(obs)
            with torch.no_grad():
                logits, value = policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs_t, action, dist.log_prob(action), value
            obs, reward, done = env.step(action.numpy())
            rew_buf[t] = torch.from_numpy(reward)
        assert done, "a rollout should end exactly when the races do"

        # GAE, undiscounted: the objective is total race time. Races end at the last step.
        adv = torch.zeros(steps, n)
        running = torch.zeros(n)
        for t in reversed(range(steps)):
            next_value = val_buf[t + 1] if t + 1 < steps else 0.0
            delta = rew_buf[t] + next_value - val_buf[t]
            running = delta + cfg.gae_lambda * running
            adv[t] = running
        returns = adv + val_buf

        b_obs, b_act, b_logp = obs_buf.flatten(0, 1), act_buf.flatten(), logp_buf.flatten()
        b_adv, b_ret = adv.flatten(), returns.flatten()
        batch = steps * n
        size = batch // cfg.minibatches
        for _ in range(cfg.epochs):
            for idx in torch.randperm(batch).split(size):
                logits, value = policy(b_obs[idx])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                value_loss = (value - b_ret[idx]).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

        race_times = [info["race_time"] for info in env.final_info]
        record = {
            "iteration": iteration,
            "steps": iteration * batch,
            "seconds": round(time.perf_counter() - start, 2),
            "mean_race_time": float(np.mean(race_times)),
            "rule_violations": sum(info["compounds_used"] < 2 for info in env.final_info),
            "entropy": float(dist.entropy().mean().detach()),
        }
        history.append(record)
        if iteration == 1 or iteration % 25 == 0 or iteration == cfg.iterations:
            log(f"iter {iteration:4d}  {record['steps']:>8,} laps  {record['seconds']:7.1f}s  "
                f"race time {record['mean_race_time']:8.1f}s  "
                f"rule violations {record['rule_violations']:2d}/{n}  "
                f"entropy {record['entropy']:.2f}")
    return policy, history


def save_policy(path: Path, policy: Policy, scenario: Scenario, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "obs_size": scenario.obs_size,
        "actions": scenario.actions,
        "hero": scenario.hero,
        "rival": scenario.rival,
        "race": scenario.calibration["race"],
        "config": asdict(cfg),
    }, path)


def load_policy(path: Path, scenario: Scenario) -> Policy:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `f1-mlops train` first")
    saved = torch.load(path, weights_only=True)
    expected = (scenario.obs_size, scenario.actions, scenario.hero, scenario.rival)
    if (saved["obs_size"], saved["actions"], saved["hero"], saved["rival"]) != expected:
        raise ValueError(f"{path} was trained for a different scenario; retrain it")
    policy = Policy(saved["obs_size"], len(saved["actions"]), saved["config"]["hidden"])
    policy.load_state_dict(saved["state_dict"])
    return policy.eval()


def save_history(path: Path, history: list[dict], cfg: TrainConfig) -> None:
    path.write_text(json.dumps({"config": asdict(cfg), "history": history}, indent=1) + "\n")
