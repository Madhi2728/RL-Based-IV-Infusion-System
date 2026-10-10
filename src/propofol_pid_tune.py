"""
propofol_pid_tune.py
---------------------

PID baseline for PropofolEnv -- BIS-ONLY feedback (v3, the default).
  observes : measured BIS only (+ its own integral/derivative state). No Ce,
             no Cp, no plant state -- exactly what a BIS monitor provides.
  t = 0    : one 1 mg/kg induction bolus (env action BOLUS_ACTION; an action,
             not an observation), then every 5 s:
  e_pid    = measured_BIS - target_BIS     (positive = too light -> more drug)
  rate     = PID(e_pid), clipped to [0, 40] mg/kg/h, via env.step_continuous
  anti-windup: |Ki * integral| <= 30 mg/kg/h (integral_limit = 30 / Ki), so the
             integrator can hold any maintenance rate in the patient ranges
             (up to ~28 mg/kg/h for a young, resistant patient under
             stimulation) but cannot wind past it
  D filter : derivative_filter = 0.2 (EMA on de/dt, ~25 s) -- raw de/dt of
             BIS with sigma=2 noise is otherwise pure noise
  dt = 5 s

FAIRNESS NOTE: the v2 baseline (CeCascadePID below, kept for reference only)
closed its inner loop on the TCI-model effect-site concentration Ce. In this
simulator there is no PK model error, so that Ce equals the true plant Ce --
information no real BIS monitor gives. v3 removes it: the PID now sees only BIS.

Tuning (scipy.optimize.minimize, Nelder-Mead over log10(Kp, Ki, Kd), 4 starts):
  plant     = NOMINAL Schnider patient only (age 53, 77 kg, 177 cm, male),
              nominal PD (Ce50=4.7, gamma=2.0), NO surgical stimuli, base target
              profile (60 -> 50 ramp, no extra step), BIS noise sigma=2.0 (the
              robustness study's sensor noise; fixed noise seed)
  objective = J = dt * sum( |e| + 10 * du_penalty * |d level| ), i.e. IAE plus
              the RL reward's own smoothness term at the same relative weight
Tuned gains (python -m src.propofol_pid_tune --seed 0), hardcoded as TUNED_PID_GAINS:
  Kp = 0.4011 (mg/kg/h per BIS)   Ki = 0.003560 (/s)   Kd = 0 (tuned to 1.7e-13)
  nominal IAE = 11,298 BIS*s (J = 12,245) -- effectively a PI controller
Saves: results/propofol_pid_gains.json

Usage:
    python -m src.propofol_pid_tune --seed 0
"""

import argparse
import json

import numpy as np
from scipy.optimize import minimize

from src.pid_controller import PIDController
from src.propofol_environment import BISModel, PropofolEnv, NOMINAL_PATIENT
from src.propofol_metrics import pad_history

# BIS-only PID gains from `python -m src.propofol_pid_tune --seed 0`: 3 of 4
# Nelder-Mead starts converge to Kp=0.4011, Ki=0.003560, Kd=1.7e-13 (J=12,245,
# nominal IAE=11,298 BIS*s). The optimizer switches the D term off: with BIS
# noise sigma=2 even the filtered derivative costs more (jitter) than it buys,
# so the tuned controller is effectively PI. Kd is hardcoded as exactly 0.
TUNED_PID_GAINS = dict(kp=0.4011, ki=0.003560, kd=0.0)

PID_INTEGRAL_TERM_CAP = 30.0     # mg/kg/h, anti-windup on the integral term
PID_DERIVATIVE_FILTER = 0.2      # EMA weight on de/dt (~25 s time constant at dt=5 s)
TUNING_NOISE_STD = 2.0


class BISPID:
    """Induction bolus + PID on measured BIS only (the default PID baseline)."""

    def __init__(self, env, kp, ki, kd, bolus=True):
        self.env = env
        self.bolus = bolus
        self.pid = PIDController(kp=kp, ki=ki, kd=kd, dt=env.dt, u_min=0.0,
                                 u_max=env.MAX_RATE_MG_KG_H,
                                 integral_limit=PID_INTEGRAL_TERM_CAP / ki if ki > 0 else 1e12,
                                 derivative_filter=PID_DERIVATIVE_FILTER)
        self.reset()

    def reset(self):
        self.pid.reset()

    def step(self):
        """Advance the env one step under PID control; returns env.step's tuple."""
        env = self.env
        if self.bolus and env.step_idx == 0:
            return env.step(env.BOLUS_ACTION)
        rate = self.pid.compute(env.bis - env.target)
        return env.step_continuous(rate)


# app.py (frozen in the round that introduced BISPID) imports the name
# `CascadePID`; it now resolves to the BIS-only controller so the dashboard
# uses the fair baseline too.
CascadePID = BISPID


class CeCascadePID:
    """v2 baseline, kept for reference only -- NOT used by default.
    Bolus + outer BIS->Ce_ref I-trim + inner PID on (Ce_ref - Ce), where Ce is
    the TCI-model effect-site concentration (== true plant Ce in this
    simulator, i.e. information a BIS monitor does not provide).
    v2 tuned gains: kp=17.0715, ki=0.0, kd=789.5324; outer_ki=0.0008, trim +/-3."""

    V2_GAINS = dict(kp=17.0715, ki=0.0, kd=789.5324)

    def __init__(self, env, kp, ki, kd, outer_ki=0.0008, trim_cap=3.0):
        self.env = env
        self.inner = PIDController(kp=kp, ki=ki, kd=kd, dt=env.dt, u_min=0.0,
                                   u_max=env.MAX_RATE_MG_KG_H,
                                   integral_limit=15.0 / ki if ki > 0 else 1e12,
                                   derivative_filter=PID_DERIVATIVE_FILTER)
        self.outer = PIDController(kp=0.0, ki=outer_ki, kd=0.0, dt=env.dt,
                                   u_min=-trim_cap, u_max=trim_cap, integral_limit=trim_cap / outer_ki)

    def step(self):
        env = self.env
        if env.step_idx == 0:
            return env.step(env.BOLUS_ACTION)
        ce_ref = BISModel.nominal_ce_for_bis(env.target) + self.outer.compute(env.bis - env.target)
        return env.step_continuous(self.inner.compute(ce_ref - env.plant.Ce))


def run_pid_episode(env, kp=None, ki=None, kd=None):
    """One full PropofolEnv episode under BIS-only PID control. Returns env.history."""
    g = dict(TUNED_PID_GAINS)
    g.update({k: v for k, v in dict(kp=kp, ki=ki, kd=kd).items() if v is not None})
    env.reset()
    pid = BISPID(env, **g)
    done = False
    while not done:
        _, _, done, _ = pid.step()
    return env.history


def nominal_tuning_env(seed=0):
    return PropofolEnv(seed=seed, mode="eval", patient=NOMINAL_PATIENT, randomize_pd=False,
                       stimuli=False, extra_target_steps=False, bis_noise_std=TUNING_NOISE_STD)


def nominal_cost(gains, seed=0):
    """(objective, IAE) on the nominal tuning episode.
    objective = IAE + smoothness, weighted exactly like the RL reward's
    tracking vs smoothness terms (|e|/10 vs du_penalty*|d level| per step):
        J = dt * sum( |e| + 10 * du_penalty * |d level_frac| )"""
    env = nominal_tuning_env(seed)
    raw = run_pid_episode(env, **gains)
    hist = pad_history(raw, env.n_steps, env.dt)
    iae = float(sum(abs(h["error"]) for h in hist) * env.dt)
    rates = [h["command"] for h in raw[1:]]          # skip the bolus step
    levels = np.interp(rates, env.RATE_LEVELS_MG_KG_H, np.arange(env.n_levels))
    tv = float(np.abs(np.diff(levels)).sum())
    return iae + 10.0 * env.du_penalty * tv * env.dt, iae


def nominal_iae(gains, seed=0):
    return nominal_cost(gains, seed)[1]


def tune(seed=0, maxiter=300):
    def objective(log_g):
        kp, ki, kd = 10.0 ** np.asarray(log_g)
        return nominal_cost(dict(kp=kp, ki=ki, kd=kd), seed=seed)[0]

    starts = [np.log10([0.5, 0.003, 5.0]), np.log10([1.5, 0.01, 1.0]),
              np.log10([0.3, 0.001, 20.0]), np.log10([3.0, 0.005, 0.1])]
    best = None
    for x0 in starts:
        res = minimize(objective, x0, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, xatol=1e-3, fatol=1.0))
        kp, ki, kd = 10.0 ** res.x
        iae = nominal_iae(dict(kp=kp, ki=ki, kd=kd), seed)
        print(f"  start {np.round(10.0 ** x0, 4)} -> Kp={kp:.4f} Ki={ki:.6f} Kd={kd:.4f}  "
              f"J={res.fun:.1f}  IAE={iae:.1f}", flush=True)
        if best is None or res.fun < best.fun:
            best = res
    kp, ki, kd = 10.0 ** best.x
    gains = dict(kp=float(kp), ki=float(ki), kd=float(kd))
    return gains, nominal_iae(gains, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="BIS-noise seed of the nominal tuning episode")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--out", type=str, default="results/propofol_pid_gains.json")
    args = parser.parse_args()

    print(f"Tuning BIS-only PID on the nominal Schnider patient {NOMINAL_PATIENT} "
          f"(no stimuli, noise sigma {TUNING_NOISE_STD}, seed {args.seed})")
    gains, iae = tune(seed=args.seed, maxiter=args.maxiter)
    print(f"\nBest: Kp={gains['kp']:.4f}  Ki={gains['ki']:.6f}  Kd={gains['kd']:.4f}  nominal IAE={iae:.1f}")
    print(f"Hardcoded default TUNED_PID_GAINS = {TUNED_PID_GAINS}  "
          f"(nominal IAE {nominal_iae(TUNED_PID_GAINS, seed=args.seed):.1f})")
    with open(args.out, "w") as f:
        json.dump(dict(structure="BIS-only PID: 1 mg/kg bolus at t=0, then PID on (BIS - target)",
                       gains=gains, integral_term_cap=PID_INTEGRAL_TERM_CAP,
                       derivative_filter=PID_DERIVATIVE_FILTER, tuning_noise_std=TUNING_NOISE_STD,
                       nominal_iae=iae, seed=args.seed, patient=NOMINAL_PATIENT,
                       superseded_v2_ce_cascade=CeCascadePID.V2_GAINS), f, indent=2)
    print(f"Saved tuned gains to {args.out}")
