"""
propofol_evaluate.py
---------------------

Eval episodes: --n_eval, default 50 (eval seeds 20000 + i), PropofolEnv mode="eval"
               (random patient age 20-80 / 50-110 kg / 150-195 cm / M-F, Ce50
               +/-25 %, gamma +/-20 %, 3-4 stimuli of +20-30 % Ce50, 3-5 min).
Controllers  : PID (TUNED_PID_GAINS from src/propofol_pid_tune.py, tuned on the
               nominal patient only), Q-learning (--qtable), DQN (--dqn_model);
               an RL controller whose file is missing is skipped with a warning.
Fairness     : each controller gets its own PropofolEnv(seed=20000+i, "eval");
               the per-episode seed streams make the patient, PD jitter, stimulus
               schedule, target schedule AND BIS-noise sequence identical across
               controllers -- this is asserted for every episode below.
Metrics (src/propofol_metrics.py): IAE ISE ITAE ITSE RMSE, time-in-target
               (|BIS-target| <= 5) %, % BIS < 40, % BIS > 60, induction time,
               settling time after target change, total propofol (mg), mean
               rate, early-termination rate.
Writes: results/propofol_eval_episodes.csv   (one row per controller x episode)
        results/propofol_eval_summary.csv    (mean / std per controller)
        results/propofol_eval_summary.json   (same schema as eval_summary.json,
                                              read by app.py)
        results/propofol_eval_bis.png        (example-episode BIS traces)
        results/propofol_eval_ce.png         (example-episode Ce traces)
        results/propofol_eval_bars.png       (headline-metric bar chart)

Usage:
    python -m src.propofol_evaluate --n_eval 50
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.propofol_environment import PropofolEnv
from src.propofol_metrics import compute_propofol_metrics
from src.propofol_pid_tune import TUNED_PID_GAINS, run_pid_episode
from src.propofol_train import make_dqn_agent, run_dqn_episode, run_qlearning_episode
from src.q_learning_agent import QLearningAgent

EVAL_SEED_BASE = 20_000
COLORS = {"PID": "tab:blue", "Q-learning": "tab:orange", "DQN": "tab:green"}
HEADLINE = [
    ("time_in_target_pct", "Time in target (+/-5) %", True),
    ("pct_bis_below_40", "% time BIS < 40", False),
    ("pct_bis_above_60", "% time BIS > 60", False),
    ("IAE", "IAE (BIS*s)", False),
    ("settling_time_s", "Settling after target change (s)", False),
    ("total_propofol_mg", "Total propofol (mg)", None),
]


def _signature(env):
    return (tuple(sorted(env.patient.items())), env.bis_model.ce50, env.bis_model.gamma,
            env.stim_shift.tobytes(), env.target_schedule.tobytes())


def build_controllers(qtable_path, dqn_model_path):
    """name -> fn(env) -> history. Missing RL models are skipped."""
    controllers = {"PID": lambda env: run_pid_episode(env, **TUNED_PID_GAINS)}

    if qtable_path and os.path.exists(qtable_path):
        probe = PropofolEnv()
        ql = QLearningAgent(probe.n_states, probe.n_actions)
        ql.load(qtable_path)
        controllers["Q-learning"] = lambda env: run_qlearning_episode(env, ql, greedy=True)[0]
    else:
        print(f"[skip] no Q-table at {qtable_path} -- run `python -m src.propofol_train --agent qlearning`")

    if dqn_model_path and os.path.exists(dqn_model_path):
        probe = PropofolEnv()
        dqn = make_dqn_agent(probe.n_actions)
        dqn.load(dqn_model_path)
        controllers["DQN"] = lambda env: run_dqn_episode(env, dqn, greedy=True)[0]
    else:
        print(f"[skip] no DQN model at {dqn_model_path} -- run `python -m src.propofol_train --agent dqn`")
    return controllers


def evaluate(n_eval=50, qtable_path="results/propofol_qlearning_qtable.pkl",
             dqn_model_path="results/propofol_dqn_model.pt", out_prefix="results/propofol_eval"):
    controllers = build_controllers(qtable_path, dqn_model_path)
    rows, examples, example_env = [], {}, None

    for i in range(n_eval):
        seed = EVAL_SEED_BASE + i
        sig = None
        for name, run in controllers.items():
            env = PropofolEnv(seed=seed, mode="eval")
            hist = run(env)
            s = _signature(env)
            assert sig is None or s == sig, f"seed {seed}: {name} saw a different patient/schedule"
            sig = s
            m = compute_propofol_metrics(hist, env.target_change_steps, env.n_steps, env.dt)
            rows.append(dict(controller=name, episode=i, seed=seed, **env.patient,
                             ce50=env.bis_model.ce50, gamma=env.bis_model.gamma, **m))
            if i == 0:
                examples[name] = hist
                example_env = env

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_prefix}_episodes.csv", index=False)
    metric_cols = [c for c in df.columns if c not in
                   ("controller", "episode", "seed", "age", "weight", "height", "sex", "ce50", "gamma")]
    summary = df.groupby("controller", sort=False)[metric_cols].agg(["mean", "std"])
    summary.to_csv(f"{out_prefix}_summary.csv")

    summary_json = {"n_eval": n_eval, "pid_gains": TUNED_PID_GAINS, "controllers": {}}
    for name in controllers:
        sub = df[df.controller == name]
        print(f"\n--- {name} (mean over {len(sub)} eval patients) ---")
        summary_json["controllers"][name] = {}
        for k in metric_cols:
            mean, std = float(sub[k].mean()), float(sub[k].std(ddof=0))
            summary_json["controllers"][name][k] = {"mean": mean, "std": std}
            print(f"  {k:22s}: {mean:10.2f}  (+/- {std:.2f})")
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSaved {out_prefix}_episodes.csv, {out_prefix}_summary.csv, {out_prefix}_summary.json")

    _plot_traces(examples, example_env, out_prefix)
    _plot_bars(df, list(controllers), out_prefix)
    return summary_json


def _shade_stimuli(ax, env):
    on = env.stim_shift > 0
    t = np.arange(len(on)) * env.dt / 60.0
    ax.fill_between(t, 0, 1, where=on, transform=ax.get_xaxis_transform(),
                    color="red", alpha=0.08, label="surgical stimulus")


def _plot_traces(examples, env, out_prefix):
    p = env.patient
    patient_str = (f"{p['sex']}, {p['age']:.0f} y, {p['weight']:.0f} kg, {p['height']:.0f} cm, "
                   f"Ce50={env.bis_model.ce50:.2f}, gamma={env.bis_model.gamma:.2f}")

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 3.2 * len(examples)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, hist) in zip(axes, examples.items()):
        t = np.array([h["t"] for h in hist]) / 60.0
        ax.axhspan(40, 60, color="grey", alpha=0.12, label="BIS 40-60")
        ax.plot(t, [h["target"] for h in hist], "k--", label="target")
        ax.plot(t, [h["measured"] for h in hist], color=COLORS.get(name), lw=1, label="measured BIS")
        _shade_stimuli(ax, env)
        ax2 = ax.twinx()
        ax2.step(t, [h["rate_mg_kg_h"] for h in hist], where="post", color="grey", alpha=0.6, lw=0.8)
        ax2.set_ylabel("rate (mg/kg/h)")
        ax2.set_ylim(0, 13)
        ax.set_ylim(0, 100)
        ax.set_ylabel("BIS")
        ax.set_title(f"{name} - example eval episode")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (min)")
    fig.suptitle(patient_str, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bis.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, hist in examples.items():
        t = np.array([h["t"] for h in hist]) / 60.0
        ax.plot(t, [h["ce"] for h in hist], color=COLORS.get(name), label=f"{name} Ce")
    h0 = next(iter(examples.values()))
    t0 = np.array([h["t"] for h in h0]) / 60.0
    ax.plot(t0, [h["ce50_eff"] for h in h0], "r:", label="effective Ce50 (stimulus-shifted)")
    _shade_stimuli(ax, env)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Effect-site Ce (ug/mL)")
    ax.set_title("Effect-site concentration - example eval episode\n" + patient_str, fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_ce.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_prefix}_bis.png and {out_prefix}_ce.png")


def _plot_bars(df, names, out_prefix):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, (key, label, higher_better) in zip(axes.flat, HEADLINE):
        means = [df[df.controller == n][key].mean() for n in names]
        stds = [df[df.controller == n][key].std(ddof=0) for n in names]
        ax.bar(names, means, yerr=stds, capsize=4, color=[COLORS.get(n) for n in names])
        hint = "" if higher_better is None else (" (higher better)" if higher_better else " (lower better)")
        ax.set_title(label + hint, fontsize=9)
    fig.suptitle(f"Propofol BIS control - mean +/- std over {df.episode.nunique()} eval patients")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bars.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_prefix}_bars.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--qtable", type=str, default="results/propofol_qlearning_qtable.pkl")
    parser.add_argument("--dqn_model", type=str, default="results/propofol_dqn_model.pt")
    parser.add_argument("--out_prefix", type=str, default="results/propofol_eval")
    args = parser.parse_args()
    evaluate(n_eval=args.n_eval, qtable_path=args.qtable, dqn_model_path=args.dqn_model,
             out_prefix=args.out_prefix)
