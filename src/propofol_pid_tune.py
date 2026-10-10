"""
propofol_pid_tune.py
---------------------

PID baseline for PropofolEnv: a CASCADE built from two src/pid_controller.PIDController
instances, plus a fixed induction bolus.
  t = 0      : one 1 mg/kg induction bolus (env action BOLUS_ACTION), then:
  outer loop : Ce_ref = Ce_nominal(target_BIS) + I-only trim on (BIS - target)
               Ce_nominal(B) = inverse of the NOMINAL BIS curve (Ce50 4.7, gamma 2)
               trim = PIDController(kp=0, ki=OUTER_KI, kd=0), clamped to
               +/- OUTER_TRIM_CAP ug/mL (anti-windup via integral_limit)
               -> slowly corrects for the unknown patient sensitivity / stimuli
  inner loop : rate = PID(Ce_ref - Ce), Ce = TCI-model effect-site estimate
               output clipped to [0, 40] mg/kg/h, driven through
               env.step_continuous (continuous pump rate, like the IV-env PID)
               anti-windup: |Ki * integral| <= 15 mg/kg/h (integral_limit = 15/Ki)
               D filter: derivative_filter = 0.2 (EMA on de/dt)
  dt = 5 s.  Fixed by hand (not tuned): OUTER_KI = 0.0008 ug/mL per BIS*s
  (10 BIS of error for 1 min moves Ce_ref by ~0.5 ug/mL), OUTER_TRIM_CAP = 3.

Why cascade: BIS responds to infusion through a slow 3-compartment PK chain
plus the ke0 lag, so a BIS-only PID either overshoots after induction or is
sluggish. Closing the fast loop on Ce (which the TCI model provides) and only
trimming the Ce setpoint from BIS is the standard closed-loop-anesthesia
structure.

Tuning (scipy.optimize.minimize, Nelder-Mead over log10(Kp, Ki, Kd) of the inner loop):
  plant     = NOMINAL Schnider patient only (age 53, 77 kg, 177 cm, male),
              nominal PD (Ce50=4.7, gamma=2.0), NO surgical stimuli, base target
              profile (60 -> 50 ramp, no extra step), BIS noise ON (sigma=1.5,
              fixed noise seed -> deterministic objective)
  objective = J = dt * sum( |e| + 10 * du_penalty * |d level| ), i.e. IAE plus
              the RL reward's own smoothness term at the same relative weight
  starts    = 4 initial guesses, best kept
Tuned gains (python -m src.propofol_pid_tune --seed 0), hardcoded as TUNED_PID_GAINS:
  Kp = 17.07 (mg/kg/h per ug/mL)   Ki = 0 (tuned to ~1e-111)   Kd = 789.5 (s)
  -> effectively a PD inner loop; integral action lives in the outer BIS loop.
  Kd is large because the inner loop differentiates the noise-free TCI Ce
  (dCe/dt ~ 1e-3 ug/mL/s), not noisy BIS.
  Nominal patient, nominal tuning episode: IAE = 9,585 BIS*s.
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

# Inner-loop gains from `python -m src.propofol_pid_tune --seed 0` (Nelder-Mead,
# best of 4 starts, nominal J=9915 / IAE=9585). Ki is driven to ~1e-111 by
# every start, i.e. the optimizer removes inner integral action: the outer
# BIS loop is already an integrator (it trims Ce_ref until BIS error is zero),
# so a second integrator on Ce only adds overshoot. Hardcoded as exactly 0.
TUNED_PID_GAINS = dict(kp=17.0715, ki=0.0, kd=789.5324)

PID_INTEGRAL_TERM_CAP = 15.0     # mg/kg/h, inner-loop anti-windup
PID_DERIVATIVE_FILTER = 0.2      # EMA weight on de/dt (~25 s time constant at dt=5 s)
OUTER_KI = 0.0008                # ug/mL per (BIS * s)
OUTER_TRIM_CAP = 3.0             # ug/mL


class CascadePID:
    """Induction bolus + outer BIS->Ce_ref trim + inner Ce->rate PID."""

    def __init__(self, env, kp, ki, kd, outer_ki=OUTER_KI, bolus=True):
        self.env = env
        self.bolus = bolus
        self.inner = PIDController(kp=kp, ki=ki, kd=kd, dt=env.dt, u_min=0.0,
                                   u_max=env.MAX_RATE_MG_KG_H,
                                   integral_limit=PID_INTEGRAL_TERM_CAP / ki if ki > 0 else 1e12,
                                   derivative_filter=PID_DERIVATIVE_FILTER)
        self.outer = PIDController(kp=0.0, ki=outer_ki, kd=0.0, dt=env.dt,
                                   u_min=-OUTER_TRIM_CAP, u_max=OUTER_TRIM_CAP,
                                   integral_limit=OUTER_TRIM_CAP / outer_ki)
        self.reset()

    def reset(self):
        self.inner.reset()
        self.outer.reset()
        self.ce_ref = 0.0

    def step(self):
        """Advance the env one step under PID control; returns env.step's tuple."""
        env = self.env
        if self.bolus and env.step_idx == 0:
            return env.step(env.BOLUS_ACTION)
        trim = self.outer.compute(env.bis - env.target)
        self.ce_ref = BISModel.nominal_ce_for_bis(env.target) + trim
        rate = self.inner.compute(self.ce_ref - env.plant.Ce)
        return env.step_continuous(rate)


def run_pid_episode(env, kp=None, ki=None, kd=None):
    """One full PropofolEnv episode under cascade-PID control. Returns env.history."""
    g = dict(TUNED_PID_GAINS)
    g.update({k: v for k, v in dict(kp=kp, ki=ki, kd=kd).items() if v is not None})
    env.reset()
    pid = CascadePID(env, **g)
    done = False
    while not done:
        _, _, done, _ = pid.step()
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

    starts = [np.log10([2.0, 0.01, 10.0]), np.log10([5.0, 0.05, 1.0]),
              np.log10([1.0, 0.002, 30.0]), np.log10([10.0, 0.02, 0.1])]
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

    print(f"Tuning cascade PID on the nominal Schnider patient {NOMINAL_PATIENT} (no stimuli, seed {args.seed})")
    gains, iae = tune(seed=args.seed, maxiter=args.maxiter)
    print(f"\nBest: Kp={gains['kp']:.4f}  Ki={gains['ki']:.6f}  Kd={gains['kd']:.4f}  nominal IAE={iae:.1f}")
    print(f"Hardcoded default TUNED_PID_GAINS = {TUNED_PID_GAINS}  "
          f"(nominal IAE {nominal_iae(TUNED_PID_GAINS, seed=args.seed):.1f})")
    with open(args.out, "w") as f:
        json.dump(dict(structure="cascade: bolus + BIS->Ce_ref (I-only) + Ce->rate PID",
                       gains=gains, outer_ki=OUTER_KI, outer_trim_cap=OUTER_TRIM_CAP,
                       integral_term_cap=PID_INTEGRAL_TERM_CAP, derivative_filter=PID_DERIVATIVE_FILTER,
                       nominal_iae=iae, seed=args.seed, patient=NOMINAL_PATIENT), f, indent=2)
    print(f"Saved tuned gains to {args.out}")
