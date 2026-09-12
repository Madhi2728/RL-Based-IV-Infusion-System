"""
train.py
--------
Trains the tabular Q-learning agent on IVInfusionEnv over many randomized
episodes (random target flow rates, random occlusion/pressure disturbances),
then saves the learned Q-table and a training-reward curve.

Usage:
    python -m src.train --episodes 3000 --seed 0
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment import IVInfusionEnv
from src.q_learning_agent import QLearningAgent


def train(n_episodes=3000, seed=0, out_prefix="results/qlearning"):
    env = IVInfusionEnv(seed=seed, mode="train")
    agent = QLearningAgent(env.n_states, env.n_actions, seed=seed)

    ep_rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        total_r = 0.0
        while not done:
            a = agent.select_action(s)
            s_next, r, done, _ = env.step(a)
            agent.update(s, a, r, s_next, done)
            s = s_next
            total_r += r
        agent.decay_epsilon()
        ep_rewards.append(total_r)

        if (ep + 1) % 200 == 0:
            recent = np.mean(ep_rewards[-200:])
            print(f"episode {ep+1}/{n_episodes}  avg_reward(last200)={recent:.2f}  eps={agent.eps:.3f}")

    agent.save(f"{out_prefix}_qtable.pkl")

    plt.figure(figsize=(8, 4))
    window = 50
    smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
    plt.plot(ep_rewards, alpha=0.3, label="episode reward")
    plt.plot(range(window - 1, len(ep_rewards)), smoothed, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-learning training convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_training_curve.png", dpi=150)
    print(f"Saved Q-table to {out_prefix}_qtable.pkl and curve to {out_prefix}_training_curve.png")
    return agent, ep_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(n_episodes=args.episodes, seed=args.seed)
