"""
dqn_agent.py
------------
DQN agent (PyTorch) for IV infusion flow control, using the same
IVInfusionEnv/reward as the tabular Q-learning agent and PID baseline, but
operating on a continuous state representation instead of the env's
discretized (error, error-rate) bucket index:

    state = (error, error-rate, k_eff_estimate)

`error` and `error-rate` come straight from the env's per-step history
(target - measured flow, and its step-to-step change). `k_eff_estimate` is
NOT the plant's true occlusion factor -- the controller never observes that
directly (see environment.py) -- it's a causal online estimate built only
from the command the agent itself sent and the flow it measured back:

    k_hat = clip(measured_flow / command, 0, 1.2), EMA-smoothed

which tracks the same information a real infusion pump's fault-detection
logic would have (commanded vs. delivered flow ratio), giving the network a
feature that reflects "the line seems to be delivering less than commanded"
without cheating by reading env internals.

Action space is the same 7 discrete pump-command deltas used by
QLearningAgent, reusing IVInfusionEnv.ACTIONS -- only the state
representation and the function approximator differ.
"""

import argparse
import random
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from src.environment import IVInfusionEnv


class KEffEstimator:
    """Causal, online estimate of effective line gain from command vs. measured flow.

    Uses only signals the controller actually has access to (its own command
    and the measured flow), never the plant's true k_eff.
    """

    def __init__(self, alpha=0.1, u_threshold=5.0):
        self.alpha = alpha
        self.u_threshold = u_threshold
        self.estimate = 1.0

    def reset(self):
        self.estimate = 1.0

    def update(self, command, measured_flow):
        if command > self.u_threshold:
            ratio = float(np.clip(measured_flow / command, 0.0, 1.2))
            self.estimate = (1 - self.alpha) * self.estimate + self.alpha * ratio
        return self.estimate


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000, seed=None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(s_next, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


STATE_DIM = 3  # (error, error_rate, k_eff_estimate)


def _normalize_state(err, derr, k_hat, err_clip=60.0, derr_clip=15.0):
    return np.array([
        np.clip(err / err_clip, -3.0, 3.0),
        np.clip(derr / derr_clip, -3.0, 3.0),
        k_hat,
    ], dtype=np.float32)


class DQNAgent:
    def __init__(self, n_actions, state_dim=STATE_DIM, hidden=64, lr=1e-3,
                 gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=0.995,
                 buffer_capacity=50_000, batch_size=64, target_sync_every=500,
                 seed=None, device="cpu"):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.policy_net = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, seed=seed)
        self._train_steps = 0

    def select_action(self, state, greedy=False):
        if (not greedy) and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(torch.argmax(q, dim=1).item())

    def store(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s_next, done = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, device=self.device)
        s_next = torch.as_tensor(s_next, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        q_sa = self.policy_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s_next).max(dim=1)[0]
            target = r + self.gamma * q_next * (1.0 - done)
        loss = nn.functional.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)


def run_dqn_episode(env, agent, k_eff_estimator=None, greedy=True):
    """Run one IVInfusionEnv episode with a DQN agent, using the continuous
    (error, error-rate, k_eff-estimate) state instead of env's discretized
    state index. Returns env.history (same schema PID/Q-learning produce).
    """
    if k_eff_estimator is None:
        k_eff_estimator = KEffEstimator()
    k_eff_estimator.reset()

    env.reset()
    err = env.target - env.plant.Q
    derr = 0.0
    k_hat = k_eff_estimator.estimate
    state = _normalize_state(err, derr, k_hat)

    done = False
    total_r = 0.0
    while not done:
        a = agent.select_action(state, greedy=greedy)
        _, r, done, _ = env.step(a)
        h = env.history[-1]
        new_err = h["error"]
        derr = new_err - err
        err = new_err
        k_hat = k_eff_estimator.update(h["command"], h["measured"])
        next_state = _normalize_state(err, derr, k_hat)

        if not greedy:
            agent.store(state, a, r, next_state, done)
            agent.train_step()

        state = next_state
        total_r += r

    return env.history, total_r


def train_dqn(n_episodes=3000, seed=0, out_prefix="results/dqn"):
    env = IVInfusionEnv(seed=seed, mode="train")
    agent = DQNAgent(env.n_actions, seed=seed)
    k_eff_estimator = KEffEstimator()

    ep_rewards = []
    for ep in range(n_episodes):
        _, total_r = run_dqn_episode(env, agent, k_eff_estimator, greedy=False)
        agent.decay_epsilon()
        ep_rewards.append(total_r)

        if (ep + 1) % 200 == 0:
            recent = np.mean(ep_rewards[-200:])
            print(f"episode {ep+1}/{n_episodes}  avg_reward(last200)={recent:.2f}  eps={agent.eps:.3f}")

    agent.save(f"{out_prefix}_model.pt")

    plt.figure(figsize=(8, 4))
    window = 50
    smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
    plt.plot(ep_rewards, alpha=0.3, label="episode reward")
    plt.plot(range(window - 1, len(ep_rewards)), smoothed, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("DQN training convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_training_curve.png", dpi=150)
    print(f"Saved model to {out_prefix}_model.pt and curve to {out_prefix}_training_curve.png")
    return agent, ep_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_dqn(n_episodes=args.episodes, seed=args.seed)
