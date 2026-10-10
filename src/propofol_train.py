"""
propofol_train.py
------------------

Trains RL agents on PropofolEnv (src/propofol_environment.py), mode="train":
random patient every episode (age 20-80, 50-110 kg, 150-195 cm, M/F), Ce50
+/-15 %, gamma +/-20 %, 2-4 surgical stimuli, 60->50 target profile.
  --agent qlearning : QLearningAgent(9720 states, 9 actions), alpha=0.15
                      gamma=0.95, one decision every 6 env steps (30 s,
                      DECISION_EVERY["qlearning"]); default 20000 episodes
  --agent dqn       : DQNAgent(9 actions, state_dim=PROPOFOL_STATE_DIM=6) on
                      env.continuous_state(); same QNetwork/ReplayBuffer/DQN
                      defaults as src/dqn_agent.py; one decision every 6 env
                      steps (30 s) and one gradient step per decision;
                      default 1500 episodes
  epsilon (both)    : 1.0 -> 0.05, exponential per-episode decay chosen so the
                      floor is reached at 70 % of the run
                      (eps_decay = 0.05 ** (1 / (0.7 * episodes)))
Both agents are reused unchanged from src/q_learning_agent.py / src/dqn_agent.py.

Decision interval ("action hold"): a chosen action is held for k env steps and
the k rewards are summed into one transition (standard frame-skip). Tabular
Q-learning at k=1 learned nothing (Q-values flat across actions to within
noise -- one 5-s step of infusion barely moves the next state), and DQN at k=1
had a gamma=0.95 horizon of ~100 s, shorter than the drug's onset, so it
learned to overdose (92 % of a nominal 30-min run below BIS 40). k=6 (30 s)
makes each decision's effect visible and stretches the horizon to ~10 min. A bolus decision delivers the bolus in
its first step and the previous infusion level for the rest of the hold.
Bootstrapping: the TD target is cut only on TRUE termination (BIS < 20,
info["terminated"]); the 60-min time limit is a truncation.
--seed seeds the env (patient / PD / stimulus / target / noise streams) AND the agent.
Saves (v2 filenames; the v1 models trained on the old 0-12 mg/kg/h, 8-action
env are incompatible and are left untouched):
       results/propofol_v2_qlearning_qtable.pkl + results/propofol_v2_qlearning_training_curve.png
       results/propofol_v2_dqn_model.pt         + results/propofol_v2_dqn_training_curve.png
       (+ *_rewards.npy with the per-episode returns behind each curve)

--obs ce|bis and --dist default|robust select the observation set and the
training distribution (robustness study: --dist robust, see
src/propofol_robustness.py). Non-default combinations save to
results/propofol_<dist>_<obs>obs_<agent>_*.

Usage:
    python -m src.propofol_train --agent qlearning --episodes 20000 --seed 0
    python -m src.propofol_train --agent dqn --episodes 1500 --seed 0
    python -m src.propofol_train --agent qlearning --episodes 20000 --seed 0 --obs bis --dist robust
    python -m src.propofol_train --agent dqn --episodes 1500 --seed 0 --obs bis --dist robust
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.propofol_environment import PropofolEnv, PROPOFOL_STATE_DIM
from src.q_learning_agent import QLearningAgent
from src.dqn_agent import DQNAgent

DEFAULT_EPISODES = {"qlearning": 20000, "dqn": 1500}
# 30-s decision interval for both agents, picked on validation seeds 30000+
# (time-in-target: Q k=12 71.6 % vs k=6 77.9 %; DQN k=1 63.5 % vs k=6 82.2 %)
DECISION_EVERY = {"qlearning": 6, "dqn": 6}
# One DQN gradient step per 4 env steps (standard DQN "update every 4"): 5-s
# propofol transitions are highly correlated, and a 720-step episode with a
# gradient step per env step is ~4x slower for little gain.
DQN_TRAIN_EVERY = 4
MODEL_PATHS = {"qlearning": "results/propofol_v2_qlearning_qtable.pkl",
               "dqn": "results/propofol_v2_dqn_model.pt"}

# Training distributions (--dist). "default" = the v2 env defaults (Ce50 +/-15 %,
# BIS noise 1.5). "robust" = the robustness study's distribution (Ce50 +/-30 %,
# BIS noise 2.0); stimuli are the train-mode 2-4 events, +15-30 %, 2-5 min in both.
DISTRIBUTIONS = {"default": dict(),
                 "robust": dict(ce50_jitter=0.30, bis_noise_std=2.0)}


def model_prefix(agent_kind, obs="ce", dist="default", out_dir="results"):
    tag = "v2" if (obs, dist) == ("ce", "default") else f"{dist}_{obs}obs"
    return f"{out_dir}/propofol_{tag}_{agent_kind}"


def model_path(agent_kind, obs="ce", dist="default", out_dir="results"):
    ext = "_qtable.pkl" if agent_kind == "qlearning" else "_model.pt"
    return model_prefix(agent_kind, obs, dist, out_dir) + ext


class ActionHold:
    """Turns one agent decision into k env steps. A bolus decision gives the
    bolus on its first step and the last infusion level on the rest."""

    def __init__(self, k):
        self.k = k
        self.level = 0  # last non-bolus action

    def run(self, env, a):
        total_r, s, done, info = 0.0, None, False, {"terminated": False}
        for j in range(self.k):
            sub = a if (j == 0 or a != env.BOLUS_ACTION) else self.level
            s, r, done, info = env.step(sub)
            total_r += r
            if done:
                break
        if a != env.BOLUS_ACTION:
            self.level = a
        return s, total_r, done, info


def run_qlearning_episode(env, agent, greedy=True, decision_every=DECISION_EVERY["qlearning"]):
    """One PropofolEnv episode with tabular Q-learning (learns unless greedy)."""
    s = env.reset()
    hold = ActionHold(decision_every)
    done = False
    total_r = 0.0
    while not done:
        a = agent.select_action(s, greedy=greedy)
        s_next, r, done, info = hold.run(env, a)
        if not greedy:
            agent.update(s, a, r, s_next, info["terminated"])
        s = s_next
        total_r += r
    return env.history, total_r


def run_dqn_episode(env, agent, greedy=True, decision_every=DECISION_EVERY["dqn"],
                    train_every=DQN_TRAIN_EVERY):
    """One PropofolEnv episode with DQN on env.continuous_state() (learns unless
    greedy). Every decision is stored; one gradient step every `train_every`
    env steps."""
    env.reset()
    hold = ActionHold(decision_every)
    state = env.continuous_state()
    done = False
    total_r = 0.0
    while not done:
        a = agent.select_action(state, greedy=greedy)
        _, r, done, info = hold.run(env, a)
        next_state = env.continuous_state()
        if not greedy:
            agent.store(state, a, r, next_state, info["terminated"])
            if env.step_idx % train_every < decision_every:
                agent.train_step()
        state = next_state
        total_r += r
    return env.history, total_r


def make_dqn_agent(n_actions, seed=None, state_dim=PROPOFOL_STATE_DIM, **kw):
    return DQNAgent(n_actions, state_dim=state_dim, seed=seed, **kw)


def eps_decay_for(n_episodes, eps_end=0.05, frac=0.7):
    return eps_end ** (1.0 / max(1.0, frac * n_episodes))


def _save_curve(ep_rewards, title, path):
    plt.figure(figsize=(8, 4))
    window = min(100, max(1, len(ep_rewards) // 10))
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


def train(agent_kind="qlearning", n_episodes=None, seed=0, out_dir="results", log_every=None,
          decision_every=None, obs="ce", dist="default"):
    n_episodes = n_episodes or DEFAULT_EPISODES[agent_kind]
    k = decision_every or DECISION_EVERY[agent_kind]
    log_every = log_every or max(1, n_episodes // 20)
    env = PropofolEnv(seed=seed, mode="train", obs_mode=obs, **DISTRIBUTIONS[dist])
    decay = eps_decay_for(n_episodes)
    if agent_kind == "qlearning":
        agent = QLearningAgent(env.n_states, env.n_actions, eps_decay=decay, seed=seed)
        run = run_qlearning_episode
    elif agent_kind == "dqn":
        # the tiny MLP is faster single-threaded (thread overhead dominates at
        # batch 64); speed only, results are unaffected
        torch.set_num_threads(1)
        agent = make_dqn_agent(env.n_actions, seed=seed, state_dim=env.state_dim, eps_decay=decay)
        run = run_dqn_episode
    else:
        raise ValueError(f"unknown agent {agent_kind!r}")

    ep_rewards = []
    for ep in range(n_episodes):
        _, total_r = run(env, agent, greedy=False, decision_every=k)
        agent.decay_epsilon()
        ep_rewards.append(total_r)
        if (ep + 1) % log_every == 0:
            recent = np.mean(ep_rewards[-log_every:])
            print(f"episode {ep+1}/{n_episodes}  avg_reward(last{log_every})={recent:.2f}  eps={agent.eps:.3f}",
                  flush=True)

    prefix = model_prefix(agent_kind, obs, dist, out_dir)
    path = model_path(agent_kind, obs, dist, out_dir)
    agent.save(path)
    np.save(f"{prefix}_rewards.npy", np.array(ep_rewards))
    label = "Q-learning" if agent_kind == "qlearning" else "DQN"
    _save_curve(ep_rewards, f"Propofol {label} training convergence (obs={obs}, dist={dist})",
                f"{prefix}_training_curve.png")
    print(f"Saved {path} and curve to {prefix}_training_curve.png")
    return agent, ep_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["qlearning", "dqn"], default="qlearning")
    parser.add_argument("--episodes", type=int, default=None,
                        help="default: 20000 for qlearning, 1500 for dqn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results",
                        help="where the model + training curve are written (default results/)")
    parser.add_argument("--decision_every", type=int, default=None,
                        help="env steps per agent decision (default: DECISION_EVERY[agent])")
    parser.add_argument("--obs", choices=["ce", "bis"], default="ce",
                        help="ce: v2 observations incl. TCI Ce; bis: BIS-derived features only")
    parser.add_argument("--dist", choices=list(DISTRIBUTIONS), default="default",
                        help="default: v2 env (Ce50 +/-15 %%, noise 1.5); robust: Ce50 +/-30 %%, noise 2")
    args = parser.parse_args()
    train(agent_kind=args.agent, n_episodes=args.episodes, seed=args.seed, out_dir=args.out_dir,
          decision_every=args.decision_every, obs=args.obs, dist=args.dist)
