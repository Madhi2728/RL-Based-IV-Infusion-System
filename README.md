# RL-Based Adaptive Control of a Simulated IV Infusion System

Reinforcement-learning (tabular Q-learning) control of a simulated IV infusion
pump's flow rate, benchmarked against a conventional PID controller, under
dynamic setpoints and physically-motivated disturbances (line occlusion,
hydrostatic pressure changes, sensor noise).

## Objectives

1. Build a mathematical simulation of an IV infusion flow process.
2. Formulate flow-rate control as an RL problem (state / action / reward).
3. Implement a tabular Q-learning agent.
4. Train the agent under changing targets and injected disturbances.
5. Evaluate the learned policy on a disturbance distribution *different* from
   training (generalization test).
6. Compare against a conventional PID controller on identical scenarios.
7. Analyze the strengths and limitations of RL vs. classical control here.

## Plant model

A first-order lag models line compliance/inertia between the pump's commanded
rate `u(t)` and the actually delivered flow `Q(t)`:

```
tau * dQ/dt = k_eff(t) * u(t) - Q(t) + d_p(t)
```

- `k_eff(t)` (0-1): effective line gain, reduced by partial/near-full
  occlusion (kinked line, positional occlusion against a vein wall).
- `d_p(t)`: additive hydrostatic disturbance (bag height changes, patient
  arm movement, venous back-pressure).
- Measured flow is `Q(t)` plus sensor noise.

Full derivation and disturbance-generation logic: `src/environment.py`.

## RL formulation

| | |
|---|---|
| **State** | discretized `(error, error-rate)` — 21 x 11 = 231 states |
| **Action** | 7 discrete pump-command deltas: `{-8,-4,-1,0,+1,+4,+8}` mL/hr |
| **Reward** | `-|e|/20 - 0.02|du| - 0.15*max(0,|e|-5) + 0.5*(bonus if |e|<1.5)` |
| **Episode** | 180 s, random target 50-200 mL/hr with 0-1 mid-episode step change, 0-2 random occlusion windows + 1-2 pressure bumps |

The reward balances tracking accuracy, control smoothness (clinically-relevant
— abrupt flow-rate jumps are undesirable), and a safety-band penalty that
models the clinical risk of straying from the prescribed rate.

## PID baseline

Standard positional PID (`u = Kp*e + Ki*integral(e) + Kd*de/dt`) with
anti-windup clamping, gains `Kp=2.0, Ki=0.4, Kd=0.5` (`src/pid_controller.py`).
Operates on the identical plant/disturbance instances as the RL agent for a
fair comparison.

## Generalization test

Training disturbances (`mode="train"`) use milder occlusion severity
(k_eff in [0.35, 0.85]) and smaller pressure bumps. Evaluation
(`mode="eval"`) draws from a harder, partly-unseen regime (k_eff in
[0.15, 0.75], larger pressure amplitude, more occlusion events per episode)
— see `DisturbanceScheduler` in `src/environment.py`. This is what "evaluate
under conditions different from training" means concretely here.

## Results (30 held-out evaluation episodes, seed-matched RL vs. PID)

| Metric | Q-learning | PID |
|---|---|---|
| IAE | 3453.7 | **1297.3** |
| ISE | 160620.3 | **37585.3** |
| RMSE (mL/hr) | 26.6 | **12.9** |
| Settling time (s) | 178.5 | **142.3** |
| Overshoot (%) | 54.0 | **29.6** |
| % time in +-5 mL/hr safe band | 21.2% | **67.1%** |

Reproduce with `python -m src.train --episodes 5000` then
`python -m src.evaluate --n_eval 30`. Plots: `results/qlearning_training_curve.png`,
`results/eval_comparison.png`.

## Analysis: strengths and limitations

**Where PID wins here, and why:** the plant is close to linear and
time-invariant around any operating point (a first-order lag with a
multiplicative gain disturbance), which is exactly the regime PID was
designed for. Its integral term drives steady-state error to zero once
occlusion clears, and its continuous-valued output avoids the limit-cycle
oscillation visible in the RL agent's trace. Tuning took one fix (an
anti-windup clamp that was too tight) rather than thousands of episodes.

**Where the tabular Q-learning agent is structurally disadvantaged:**
- **Discretization loses information.** 231 states and 7 actions can't
  represent the fine command adjustments a continuous PID makes near
  setpoint, producing visible oscillation/limit-cycling instead of smooth
  convergence (see the top panel of `eval_comparison.png`).
- **Sample inefficiency.** 5,000 episodes (~15 hours of simulated time) is a
  lot of interaction to learn what one line of calculus (PID) encodes
  directly from the error signal.
- **Table doesn't generalize within-episode disturbance combinations well** —
  a state seen alone during training (e.g. "large positive error") may pair
  with an occlusion context that was rare in training, and the flat table
  has no way to interpolate between nearby states/disturbance contexts the
  way a function approximator (DQN, or the TD3 approach used in the
  companion glucose-regulation project) would.

**Where RL-style control could still win, in principle, on a system like
this:** if the plant were meaningfully nonlinear or its parameters drifted
in ways a fixed-gain PID isn't tuned for (e.g. large viscosity swings,
patient-specific venous resistance), a learned policy could in principle
adapt without manual re-tuning — but a *tabular* agent isn't the right tool
to prove that; the honest conclusion from this study is that Q-learning's
value here is pedagogical (it cleanly demonstrates the RL formulation and
its failure modes), while a well-tuned PID remains the safer, better-
performing controller for this near-linear plant. A function-approximation
method (DQN/DDPG/TD3, as used for the blood-glucose project) would be the
natural next step to test whether RL can close this gap.

## Repo layout

```
src/
  environment.py       # plant ODE, disturbance scheduler, RL env
  q_learning_agent.py  # tabular Q-learning
  pid_controller.py    # PID baseline
  metrics.py           # IAE/ISE/RMSE/settling-time/overshoot
  train.py             # trains + saves Q-table + training curve
  evaluate.py          # RL vs PID on held-out disturbance regime
results/                # training curve + comparison plots (generated)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --episodes 5000 --seed 0
python -m src.evaluate --n_eval 30
```
