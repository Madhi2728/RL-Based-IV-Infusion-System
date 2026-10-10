"""
propofol_pid_tune.py
---------------------

PID baseline for PropofolEnv, reusing src/pid_controller.PIDController.
  e_pid  = measured_BIS - target_BIS      (positive = too light -> more drug;
                                           the env's own error is target - BIS)
  output = continuous infusion rate, clipped to [0, 30] mg/kg/h
           (driven through env.step_continuous, like evaluate.py's IV PID that
            commands a continuous pump rate instead of the RL's discrete actions)
  dt = 5 s
  anti-windup: integral_limit = 15 / Ki -> |Ki * integral| <= 15 mg/kg/h
  D filter   : derivative_filter = 0.2 (EMA on de/dt; BIS noise sigma=2 per 5 s
               sample otherwise makes the raw D term pure noise)

Tuning (scipy.optimize.minimize, Nelder-Mead over log10(Kp, Ki, Kd)):
  plant     = NOMINAL Schnider patient only (age 53, 77 kg, 177 cm, male),
              nominal PD (Ce50=4.7, gamma=2.0), NO surgical stimuli, base target
              profile (60 -> 50 ramp, no extra step), BIS noise ON (sigma=2,
              fixed noise seed -> deterministic objective)
  objective = IAE over the 60-min episode (padded if it terminates early)
  starts    = 4 initial guesses, best result kept
Tuned gains (python -m src.propofol_pid_tune --seed 0), hardcoded below as
TUNED_PID_GAINS and used by propofol_evaluate.py and app.py:
  Kp = 1.4005 (mg/kg/h per BIS)   Ki = 0.005901 (/s)   Kd = 0.1112 (s)
  nominal IAE = 12,205 BIS*s (90 % time-in-target on the nominal patient).
  All 4 starts converge to Kp ~1.4, Ki ~0.006; Kd is nearly unidentifiable
  (IAE moves < 1 % for Kd anywhere in 0.0006 - 1.0) because the loop spends
  much of the episode saturated at 12 mg/kg/h and BIS noise swamps the D term.
Saves: results/propofol_pid_gains.json

Usage:
    python -m src.propofol_pid_tune --seed 0
"""

import argparse
import json

import numpy as np
from scipy.optimize import minimize

from src.pid_controller import PIDController
from src.propofol_environment import PropofolEnv, NOMINAL_PATIENT
from src.propofol_metrics import pad_history

TUNED_PID_GAINS = dict(kp=1.4005, ki=0.005901, kd=0.1112)


# Anti-windup: the integral TERM (Ki * integral) is clamped to +/-15 mg/kg/h --
# enough to hold any maintenance rate in the patient ranges (nominal ~8-10,
# resistant patient under stimulation ~14) but well under the 30 mg/kg/h
# output ceiling, so induction (output pinned at 30 for minutes) cannot wind
# the integrator up to the full output range. The old setting
# (integral_limit = u_max / Ki) technically clamped but let the integral alone
# reach u_max, i.e. no effective anti-windup.
PID_INTEGRAL_TERM_CAP = 15.0     # mg/kg/h
PID_DERIVATIVE_FILTER = 0.2      # EMA weight on de/dt (~25 s time constant at dt=5 s)


def make_pid(env, kp, ki, kd):
    return PIDController(kp=kp, ki=ki, kd=kd, dt=env.dt, u_min=0.0, u_max=env.MAX_RATE_MG_KG_H,
                         integral_limit=PID_INTEGRAL_TERM_CAP / ki if ki > 0 else 1e12,
                         derivative_filter=PID_DERIVATIVE_FILTER)


def run_pid_episode(env, kp=None, ki=None, kd=None):
    """One full PropofolEnv episode under PID control. Returns env.history."""
    g = dict(TUNED_PID_GAINS)
    g.update({k: v for k, v in dict(kp=kp, ki=ki, kd=kd).items() if v is not None})
    pid = make_pid(env, **g)
    env.reset()
    pid.reset()
    done = False
    e_pid = env.bis - env.target
    while not done:
        rate = pid.compute(e_pid)
        _, _, done, _ = env.step_continuous(rate)
        e_pid = env.bis - env.target
    return env.history


def nominal_tuning_env(seed=0):
    return PropofolEnv(seed=seed, mode="eval", patient=NOMINAL_PATIENT, randomize_pd=False,
                       stimuli=False, extra_target_steps=False)


def nominal_cost(gains, seed=0):
    """(objective, IAE) on the nominal tuning episode.
    objective = IAE + smoothness, weighted exactly like the RL reward's
    tracking vs smoothness terms (|e|/10 vs du_penalty*|d level| per step):
        J = dt * sum( |e| + 10 * du_penalty * |d level_frac| )"""
    env = nominal_tuning_env(seed)
    raw = run_pid_episode(env, **gains)
    hist = pad_history(raw, env.n_steps, env.dt)
    iae = float(sum(abs(h["error"]) for h in hist) * env.dt)
    levels = np.interp([h["rate_mg_kg_h"] for h in raw], env.RATE_LEVELS_MG_KG_H, np.arange(env.n_levels))
    tv = float(np.abs(np.diff(np.concatenate([[0.0], levels]))).sum())
    return iae + 10.0 * env.du_penalty * tv * env.dt, iae


def nominal_iae(gains, seed=0):
    return nominal_cost(gains, seed)[1]


def tune(seed=0, maxiter=300):
    def objective(log_g):
        kp, ki, kd = 10.0 ** np.asarray(log_g)
        return nominal_cost(dict(kp=kp, ki=ki, kd=kd), seed=seed)[0]

    starts = [np.log10([0.3, 0.005, 1.0]), np.log10([1.0, 0.01, 0.1]),
              np.log10([0.1, 0.001, 5.0]), np.log10([0.5, 0.002, 0.01])]
    best = None
    for x0 in starts:
        res = minimize(objective, x0, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, xatol=1e-3, fatol=1.0))
        kp, ki, kd = 10.0 ** res.x
        iae = nominal_iae(dict(kp=kp, ki=ki, kd=kd), seed)
        print(f"  start {np.round(10.0 ** x0, 4)} -> Kp={kp:.4f} Ki={ki:.6f} Kd={kd:.4f}  "
              f"J={res.fun:.1f}  IAE={iae:.1f}")
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

    print(f"Tuning PID on the nominal Schnider patient {NOMINAL_PATIENT} (no stimuli, seed {args.seed})")
    gains, iae = tune(seed=args.seed, maxiter=args.maxiter)
    print(f"\nBest: Kp={gains['kp']:.4f}  Ki={gains['ki']:.6f}  Kd={gains['kd']:.4f}  nominal IAE={iae:.1f}")
    print(f"Hardcoded default TUNED_PID_GAINS = {TUNED_PID_GAINS}  "
          f"(nominal IAE {nominal_iae(TUNED_PID_GAINS, seed=args.seed):.1f})")
    with open(args.out, "w") as f:
        json.dump(dict(gains=gains, nominal_iae=iae, seed=args.seed, patient=NOMINAL_PATIENT), f, indent=2)
    print(f"Saved tuned gains to {args.out}")
