"""
evaluate.py
-----------
Evaluates the trained Q-learning policy AND a PID baseline on the same set of
evaluation episodes. Evaluation episodes use `mode="eval"` in the
DisturbanceScheduler, which draws occlusion severity/timing and pressure-bump
amplitude from a shifted (harder / partly unseen) distribution relative to
training -- this tests generalization, not just memorization of the training
disturbance regime.

Usage:
    python -m src.evaluate --n_eval 20 --qtable results/qlearning_qtable.pkl
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment import IVInfusionEnv
from src.q_learning_agent import QLearningAgent
from src.pid_controller import PIDController
from src.metrics import compute_metrics


def run_rl_episode(seed, qtable_path):
    env = IVInfusionEnv(seed=seed, mode="eval")
    agent = QLearningAgent(env.n_states, env.n_actions, seed=seed)
    agent.load(qtable_path)
    s = env.reset()
    done = False
    while not done:
        a = agent.select_action(s, greedy=True)
        s, r, done, _ = env.step(a)
    return env.history


def run_pid_episode(seed, kp=2.0, ki=0.4, kd=0.5):
    env = IVInfusionEnv(seed=seed, mode="eval")
    pid = PIDController(kp=kp, ki=ki, kd=kd, dt=env.dt, u_max=env.plant.u_max)
    env.reset()
    pid.reset(u0=env.plant.u)

    # Re-implement the step loop directly against the plant so the PID
    # sees a continuous command rather than the RL's discrete deltas.
    done = False
    err = env.target - env.plant.Q
    while not done:
        u = pid.compute(err)
        env.plant.set_command(u)
        if env._target_ptr < len(env.change_points) and env.step_idx >= env.change_points[env._target_ptr]:
            env._target_ptr += 1
            env.target = env.targets[env._target_ptr]
        env.plant.apply_occlusion(env.k_eff_schedule[env.step_idx])
        env.plant.apply_pressure_disturbance(env.d_p_schedule[env.step_idx])
        measured = env.plant.step()
        err = env.target - measured
        env.history.append(dict(t=env.step_idx * env.dt, target=env.target, measured=measured,
                                 command=env.plant.u, error=err,
                                 k_eff=env.plant.k_eff, d_p=env.plant.d_p))
        env.step_idx += 1
        done = env.step_idx >= env.n_steps
    return env.history


def evaluate(n_eval=20, qtable_path="results/qlearning_qtable.pkl", out_prefix="results/eval"):
    rl_metrics, pid_metrics = [], []
    example_rl, example_pid = None, None

    for i in range(n_eval):
        seed = 10_000 + i
        h_rl = run_rl_episode(seed, qtable_path)
        h_pid = run_pid_episode(seed)
        rl_metrics.append(compute_metrics(h_rl))
        pid_metrics.append(compute_metrics(h_pid))
        if i == 0:
            example_rl, example_pid = h_rl, h_pid

    def summarize(metric_list, name):
        keys = metric_list[0].keys()
        print(f"\n--- {name} (mean over {len(metric_list)} eval episodes) ---")
        summary = {}
        for k in keys:
            vals = [m[k] for m in metric_list]
            summary[k] = (float(np.mean(vals)), float(np.std(vals)))
            print(f"  {k:22s}: {summary[k][0]:.3f}  (+/- {summary[k][1]:.3f})")
        return summary

    rl_summary = summarize(rl_metrics, "Q-learning agent")
    pid_summary = summarize(pid_metrics, "PID baseline")

    # example-episode comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, hist, title in zip(axes, [example_rl, example_pid], ["Q-learning agent", "PID baseline"]):
        t = [h["t"] for h in hist]
        ax.plot(t, [h["target"] for h in hist], "k--", label="target")
        ax.plot(t, [h["measured"] for h in hist], label="measured flow")
        ax2 = ax.twinx()
        ax2.plot(t, [h["k_eff"] for h in hist], color="red", alpha=0.3, label="occlusion factor (k_eff)")
        ax2.set_ylabel("k_eff")
        ax.set_ylabel("Flow (mL/hr)")
        ax.set_title(f"{title} - example eval episode")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_comparison.png", dpi=150)
    print(f"\nSaved comparison plot to {out_prefix}_comparison.png")

    return rl_summary, pid_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--qtable", type=str, default="results/qlearning_qtable.pkl")
    args = parser.parse_args()
    evaluate(n_eval=args.n_eval, qtable_path=args.qtable)
