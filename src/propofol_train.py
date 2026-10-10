"""
propofol_train.py
------------------

Trains RL agents on PropofolEnv (src/propofol_environment.py), mode="train".
  --agent qlearning : QLearningAgent(840 states, 8 actions), defaults
                      alpha=0.15 gamma=0.95 eps 1.0 -> 0.05 (x0.995/episode)
                      default 3000 episodes
  --agent dqn       : DQNAgent(8 actions, state_dim=PROPOFOL_STATE_DIM=7) on
                      env.continuous_state(); same QNetwork/ReplayBuffer/DQN
                      defaults as src/dqn_agent.py; default 600 episodes
  episode = 720 steps (60 min at dt=5 s)
Both agents are reused unchanged from src/q_learning_agent.py / src/dqn_agent.py.
Bootstrapping: the TD target is cut only on a TRUE termination (BIS < 20,
info["terminated"]); hitting the 60-min time limit is a truncation, so the
update still bootstraps from s' (the IV-env scripts cut on both).
--seed seeds the env (patient / PD / stimulus / target / noise streams) AND the
agent, so a run is fully reproducible.
Saves: results/propofol_qlearning_qtable.pkl + results/propofol_qlearning_training_curve.png
       results/propofol_dqn_model.pt         + results/propofol_dqn_training_curve.png

Usage:
    python -m src.propofol_train --agent qlearning --episodes 3000 --seed 0
    python -m src.propofol_train --agent dqn --episodes 600 --seed 0
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.propofol_environment import PropofolEnv, PROPOFOL_STATE_DIM
from src.q_learning_agent import QLearningAgent
from src.dqn_agent import DQNAgent

DEFAULT_EPISODES = {"qlearning": 3000, "dqn": 600}


def run_qlearning_episode(env, agent, greedy=True):
    """One PropofolEnv episode with tabular Q-learning (learns unless greedy)."""
    s = env.reset()
    done = False
    total_r = 0.0
    while not done:
        a = agent.select_action(s, greedy=greedy)
        s_next, r, done, info = env.step(a)
        if not greedy:
            agent.update(s, a, r, s_next, info["terminated"])
        s = s_next
        total_r += r
    return env.history, total_r


def run_dqn_episode(env, agent, greedy=True):
    """One PropofolEnv episode with DQN on env.continuous_state() (learns unless greedy)."""
    env.reset()
    state = env.continuous_state()
    done = False
    total_r = 0.0
    while not done:
        a = agent.select_action(state, greedy=greedy)
        _, r, done, info = env.step(a)
        next_state = env.continuous_state()
        if not greedy:
            agent.store(state, a, r, next_state, info["terminated"])
            agent.train_step()
        state = next_state
        total_r += r
    return env.history, total_r


def make_dqn_agent(n_actions, seed=None):
    return DQNAgent(n_actions, state_dim=PROPOFOL_STATE_DIM, seed=seed)


def _save_curve(ep_rewards, title, path):
    plt.figure(figsize=(8, 4))
    window = min(50, max(1, len(ep_rewards)))
    smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
    plt.plot(ep_rewards, alpha=0.3, label="episode reward")
    plt.plot(range(window - 1, len(ep_rewards)), smoothed, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train(agent_kind="qlearning", n_episodes=None, seed=0, out_dir="results", log_every=100):
    n_episodes = n_episodes or DEFAULT_EPISODES[agent_kind]
    env = PropofolEnv(seed=seed, mode="train")
    if agent_kind == "qlearning":
        agent = QLearningAgent(env.n_states, env.n_actions, seed=seed)
        run = run_qlearning_episode
    elif agent_kind == "dqn":
        agent = make_dqn_agent(env.n_actions, seed=seed)
        run = run_dqn_episode
    else:
        raise ValueError(f"unknown agent {agent_kind!r}")

    ep_rewards = []
    for ep in range(n_episodes):
        _, total_r = run(env, agent, greedy=False)
        agent.decay_epsilon()
        ep_rewards.append(total_r)
        if (ep + 1) % log_every == 0:
            recent = np.mean(ep_rewards[-log_every:])
            print(f"episode {ep+1}/{n_episodes}  avg_reward(last{log_every})={recent:.2f}  eps={agent.eps:.3f}")

    prefix = f"{out_dir}/propofol_{agent_kind}"
    model_path = f"{prefix}_qtable.pkl" if agent_kind == "qlearning" else f"{prefix}_model.pt"
    agent.save(model_path)
    label = "Q-learning" if agent_kind == "qlearning" else "DQN"
    _save_curve(ep_rewards, f"Propofol {label} training convergence", f"{prefix}_training_curve.png")
    print(f"Saved {model_path} and curve to {prefix}_training_curve.png")
    return agent, ep_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["qlearning", "dqn"], default="qlearning")
    parser.add_argument("--episodes", type=int, default=None,
                        help="default: 3000 for qlearning, 600 for dqn")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(agent_kind=args.agent, n_episodes=args.episodes, seed=args.seed)
