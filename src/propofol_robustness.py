"""
propofol_robustness.py
-----------------------

Robustness study: all controllers on the SAME 100 random patients.
  patients : age U(20, 80), weight U(50, 110) kg, height U(150, 195) cm, sex M/F
  PD       : Ce50 x U(0.70, 1.30) per episode (+/-30 %, unknown to every
             controller), gamma x U(0.80, 1.20)
  sensor   : BIS noise sigma = 2.0
  stimuli  : 2-4 events per episode, +15-30 % effective Ce50, 2-5 min each
             (StimulusScheduler mode="train" -- exactly this envelope)
  target   : standard profile (60 for 10 min, ramp to 50 by 15 min, 50 after,
             p=0.5 extra step to 40-45 / 55-60 between 25 and 50 min)
  seeds    : 50000 + i, i = 0..n-1 (disjoint from training 0, validation
             30000+, eval 20000+). Every controller gets its own
             PropofolEnv(seed=50000+i), so patient, PD, stimuli, target and
             the BIS-noise draws are identical across controllers (asserted).

Controllers
  HEADLINE (same information -- BIS-derived features only, no Ce):
    PID         : BISPID (bolus + PID on BIS), TUNED_PID_GAINS, nominal-tuned
    Q-learning  : obs_mode="bis", trained with --obs bis --dist robust
    DQN         : obs_mode="bis", trained with --obs bis --dist robust
  SUPPLEMENTARY (asymmetric information, for reference -- RL also sees TCI Ce):
    Q-learning / DQN (Ce obs, v2): the app's v2 models, trained on the narrower
    default distribution (Ce50 +/-15 %, noise 1.5) -- included only if present
Metrics (src/propofol_metrics.py): time-in-target (|BIS - target| <= 5) %,
  % BIS < 40, % BIS > 60, IAE (BIS*s), early termination; mean +/- std over
  patients, plus each controller's WORST patient by IAE.
Writes: results/propofol_robustness.json, results/propofol_robustness_episodes.csv,
        results/propofol_robustness.png

Usage:
    python -m src.propofol_robustness --n 100
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.propofol_environment import PropofolEnv
from src.propofol_metrics import compute_propofol_metrics
from src.propofol_pid_tune import TUNED_PID_GAINS, run_pid_episode
from src.propofol_train import (DISTRIBUTIONS, MODEL_PATHS, make_dqn_agent, model_path,
                                run_dqn_episode, run_qlearning_episode)
from src.q_learning_agent import QLearningAgent

SEED_BASE = 50_000
ROBUST_ENV = dict(mode="train", **DISTRIBUTIONS["robust"])   # Ce50 +/-30 %, noise 2.0
METRICS = [("time_in_target_pct", "Time in target (+/-5) %", True),
           ("pct_bis_below_40", "% time BIS < 40", False),
           ("pct_bis_above_60", "% time BIS > 60", False),
           ("IAE", "IAE (BIS*s)", False)]


def _signature(env):
    return (tuple(sorted(env.patient.items())), env.bis_model.ce50, env.bis_model.gamma,
            env.stim_shift.tobytes(), env.target_schedule.tobytes())


def _rl(kind, obs, path):
    probe = PropofolEnv(obs_mode=obs)
    if kind == "qlearning":
        ag = QLearningAgent(probe.n_states, probe.n_actions)
        ag.load(path)
        return lambda env: run_qlearning_episode(env, ag, greedy=True)[0]
    ag = make_dqn_agent(probe.n_actions, state_dim=probe.state_dim)
    ag.load(path)
    return lambda env: run_dqn_episode(env, ag, greedy=True)[0]


def build_controllers():
    """name -> (obs_mode, run_fn, group)."""
    ctrls = {"PID": ("bis", lambda env: run_pid_episode(env, **TUNED_PID_GAINS), "headline")}
    for kind, label in (("qlearning", "Q-learning"), ("dqn", "DQN")):
        p = model_path(kind, obs="bis", dist="robust")
        if os.path.exists(p):
            ctrls[label] = ("bis", _rl(kind, "bis", p), "headline")
        else:
            print(f"[skip] {p} missing -- run `python -m src.propofol_train --agent {kind} --obs bis --dist robust`")
    for kind, label in (("qlearning", "Q-learning (Ce obs, v2)"), ("dqn", "DQN (Ce obs, v2)")):
        if os.path.exists(MODEL_PATHS[kind]):
            ctrls[label] = ("ce", _rl(kind, "ce", MODEL_PATHS[kind]), "supplementary")
    return ctrls


def run(n=100, out_prefix="results/propofol_robustness"):
    torch.set_num_threads(1)
    ctrls = build_controllers()
    rows = []
    for i in range(n):
        seed = SEED_BASE + i
        sig = None
        for name, (obs, fn, group) in ctrls.items():
            env = PropofolEnv(seed=seed, obs_mode=obs, **ROBUST_ENV)
            hist = fn(env)
            s = _signature(env)
            assert sig is None or s == sig, f"seed {seed}: {name} saw a different patient/schedule"
            sig = s
            m = compute_propofol_metrics(hist, env.target_change_steps, env.n_steps, env.dt)
            rows.append(dict(controller=name, group=group, seed=seed, **env.patient,
                             ce50=env.bis_model.ce50, gamma=env.bis_model.gamma,
                             n_stimuli_steps=int((env.stim_shift > 0).sum()), **m))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n} patients done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_prefix}_episodes.csv", index=False)

    keys = [k for k, _, _ in METRICS] + ["terminated_early"]
    out = {"n_patients": n, "seeds": f"{SEED_BASE}..{SEED_BASE + n - 1}",
           "distribution": {"age": [20, 80], "weight_kg": [50, 110], "height_cm": [150, 195],
                            "sex": "M/F uniform", "ce50_jitter": "+/-30 %", "gamma_jitter": "+/-20 %",
                            "bis_noise_std": 2.0, "stimuli": "2-4 events, +15-30 % Ce50, 2-5 min"},
           "pid_gains": TUNED_PID_GAINS, "controllers": {}}
    print(f"\n=== Robustness: {n} random patients, Ce50 +/-30 %, noise 2, 2-4 stimuli ===")
    print(f"{'controller':26s} {'info':12s} " + " ".join(f"{lbl:>24s}" for _, lbl, _ in METRICS))
    for name, (obs, _, group) in ctrls.items():
        sub = df[df.controller == name]
        stats = {k: {"mean": float(sub[k].mean()), "std": float(sub[k].std(ddof=0))} for k in keys}
        w = sub.loc[sub.IAE.idxmax()]
        worst = {"criterion": "max IAE", "seed": int(w.seed),
                 "patient": {k: (w[k] if k == "sex" else round(float(w[k]), 1))
                             for k in ("age", "weight", "height", "sex")},
                 "ce50": round(float(w.ce50), 3), "gamma": round(float(w.gamma), 3),
                 **{k: round(float(w[k]), 2) for k in keys}}
        out["controllers"][name] = {"group": group, "observations": obs, "stats": stats, "worst_case": worst}
        info = "BIS only" if obs == "bis" else "BIS + TCI Ce"
        print(f"{name:26s} {info:12s} " + " ".join(
            f"{stats[k]['mean']:>14.1f} +/- {stats[k]['std']:<7.1f}" for k, _, _ in METRICS))
    print("\nWorst-case patient per controller (by max IAE):")
    for name, c in out["controllers"].items():
        w = c["worst_case"]
        print(f"  {name:26s} seed {w['seed']}  {w['patient']}  Ce50={w['ce50']}  gamma={w['gamma']}  "
              f"IAE={w['IAE']:.0f}  in-target={w['time_in_target_pct']:.1f}%  "
              f"<40={w['pct_bis_below_40']:.1f}%  >60={w['pct_bis_above_60']:.1f}%")

    head = [nm for nm, (_, _, g) in ctrls.items() if g == "headline"]
    ranking = sorted(head, key=lambda nm: out["controllers"][nm]["stats"]["IAE"]["mean"])
    out["ranking_by_mean_IAE_headline"] = ranking
    print("\nHeadline ranking by mean IAE (lower = better):", " < ".join(ranking))
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(out, f, indent=2)

    names = list(ctrls)
    colors = ["tab:blue", "tab:orange", "tab:green", "navajowhite", "lightgreen"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (k, lbl, hb) in zip(axes, METRICS):
        means = [out["controllers"][nm]["stats"][k]["mean"] for nm in names]
        stds = [out["controllers"][nm]["stats"][k]["std"] for nm in names]
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors[:len(names)])
        for b, nm in zip(bars, names):
            if ctrls[nm][2] == "supplementary":
                b.set_hatch("//")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([nm.replace(" (", "\n(") for nm in names], fontsize=7)
        ax.set_title(lbl + (" (higher better)" if hb else " (lower better)"), fontsize=9)
    fig.suptitle(f"Propofol robustness: {n} random patients, Ce50 +/-30 %, BIS noise 2, 2-4 stimuli "
                 "(hatched = RL also sees TCI Ce)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_prefix}.json, {out_prefix}_episodes.csv, {out_prefix}.png")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out_prefix", type=str, default="results/propofol_robustness")
    args = parser.parse_args()
    run(n=args.n, out_prefix=args.out_prefix)
