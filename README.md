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

**Transport delay.** The plant also has a pure `delay_steps=10`-step (10 s at
`dt=1.0`) dead time between a command being issued and it reaching the ODE
(`IVInfusionPlant.pending_correction()` / `_cmd_queue` in `src/environment.py`)
— modeling line length / pump mechanism / sensor-placement lag on a real
infusion set. This delay is exposed to RL agents as an extra state feature
(`pending_correction` — how much commanded change hasn't landed yet) but
**not** to PID, which only ever sees the error signal, matching how a real
deployed pump's PID loop would be built.

Full derivation and disturbance-generation logic: `src/environment.py`.

## RL formulation

| | |
|---|---|
| **State** | discretized `(error, error-rate, pending-correction)` — 21 x 11 x 7 = 1,617 states |
| **Action** | 7 discrete pump-command deltas: `{-8,-4,-1,0,+1,+4,+8}` mL/hr |
| **Reward** | `-|e|/20 - 0.02|du| - 0.15*max(0,|e|-5) + 0.5*(bonus if |e|<1.5)` |
| **Episode** | 180 s, random target 50-200 mL/hr with 0-1 mid-episode step change, 0-2 random occlusion windows + 1-2 pressure bumps, 10-step transport delay |

The reward balances tracking accuracy, control smoothness (clinically-relevant
— abrupt flow-rate jumps are undesirable), and a safety-band penalty that
models the clinical risk of straying from the prescribed rate. The reward
formula itself is unchanged by the transport-delay addition — only the state
representation and PID tuning changed.

## PID baseline

Standard positional PID (`u = Kp*e + Ki*integral(e) + Kd*de/dt`) with
anti-windup clamping (`src/pid_controller.py`). Gains are `Kp=0.6, Ki=0.08,
Kd=0.2` — retuned down from the original delay-free `2.0/0.4/0.5` because
those gains go unstable once the 10-step transport delay is introduced (a
fixed-gain PID reacts to error that's already 10 s stale, so aggressive gains
overcorrect on landed error while more correction is still in flight and
ring/oscillate). The detuned gains are stable but slower and less precise —
that tradeoff is the whole point of adding the delay. Operates on the
identical plant/disturbance instances as the RL agents for a fair comparison.

## Generalization test

Training disturbances (`mode="train"`) use milder occlusion severity
(k_eff in [0.35, 0.85]) and smaller pressure bumps. Evaluation
(`mode="eval"`) draws from a harder, partly-unseen regime (k_eff in
[0.15, 0.75], larger pressure amplitude, more occlusion events per episode)
— see `DisturbanceScheduler` in `src/environment.py`. This is what "evaluate
under conditions different from training" means concretely here.

## Results (30 held-out evaluation episodes, seed-matched Q-learning vs. PID vs. DQN, with the 10-step transport delay)

| Metric | Q-learning | PID | DQN |
|---|---|---|---|
| IAE | 6032.9 | 4866.5 | **4215.8** |
| ISE | 461023.2 | 255069.7 | **196390.3** |
| RMSE (mL/hr) | 41.6 | 35.4 | **29.9** |
| Settling time (s) | 178.4 | **169.8** | 178.8 |
| Overshoot (%) | 54.2% | **24.4%** | 28.4% |
| % time in +-5 mL/hr safe band | 16.2% | **21.9%** | 17.1% |

Reproduce with `python -m src.train --episodes 15000 --seed 0` then
`python -m src.dqn_agent --episodes 2000 --seed 0` then
`python -m src.evaluate --n_eval 30 --dqn_model results/dqn_model.pt`.
Plots: `results/qlearning_training_curve.png`, `results/dqn_training_curve.png`,
`results/eval_comparison.png`.

**This is a genuinely mixed result, not a clean win for either side.** DQN
has the lowest IAE, ISE, and RMSE of all three controllers — it tracks the
target more accurately on average than PID does. But PID still wins on
settling time, peak overshoot, and % of time spent inside the tight +-5
mL/hr safe band. See the DQN and analysis sections below for why those two
kinds of metrics disagree here, and don't read this table as "DQN beat PID"
or "PID beat DQN" without that caveat.

(For reference, before the transport delay was added, both Q-learning and
DQN lost to PID cleanly on every metric — IAE 3453.7 / 2172.3 vs PID's
1297.3. The delay changes that story, but not into an unambiguous RL win.)

## DQN agent

`src/dqn_agent.py` adds a function-approximation baseline against the same
`IVInfusionEnv` reward and disturbance regime, to test whether tabular
discretization (not RL itself) was Q-learning's bottleneck. It uses the same
7 discrete pump-command deltas as `QLearningAgent`, but a small MLP (2 hidden
layers, 64 units, PyTorch) over a **continuous** 4-feature state instead of a
1,617-bucket table:

- `error` = target - measured flow
- `error-rate` = step-to-step change in error
- `pending_correction` — read directly from `env.plant.pending_correction()`,
  the same ground-truth "how much commanded change is still in the delay
  line" feature the tabular agent gets through its discretized state. This
  one is not an estimate: the environment is explicitly designed to expose it
  to RL agents (not PID), so DQN reads it the same way Q-learning does.
- `k_eff-estimate` — a causal, online estimate of the line's effective gain,
  built only from the agent's own command and the measured flow it gets back
  (`measured / command`, EMA-smoothed). This one **is** an estimate — no
  controller, RL or PID, ever observes the true occlusion factor directly.

Trained for 2,000 episodes (vs. Q-learning's 15,000 — a function approximator
needs less experience to cover the same continuous state space than a flat
table does) with epsilon-greedy exploration, a target network synced every
500 gradient steps, and a 50k-transition replay buffer.

**Why DQN's IAE/ISE/RMSE beat PID but its settling-time/overshoot/safe-band
numbers don't:** IAE/ISE/RMSE are *averages over the whole 180 s episode*,
dominated by how well a controller tracks through the many occlusion/target-
change transients. DQN's continuous `pending_correction` feature lets it
partially compensate for the delay during those transients, pulling its
time-averaged error below PID's detuned response. Settling time and % time
in the +-5 mL/hr band instead measure whether a controller *parks and stays*
inside a narrow deadband — and there, DQN's 7 fixed-size discrete actions
(same `{-8,-4,-1,0,+1,+4,+8}` set as Q-learning) cause small persistent
chatter around the setpoint that a continuous-valued PID output doesn't have,
so DQN crosses in and out of the tight band more than PID does even while its
average error is lower. In short: DQN is more accurate on average, PID is
more precisely settled once it gets there.

## Live comparison app

`app.py` (Streamlit) runs PID, Q-learning and DQN side by side on three
independent instances of the same plant, stepped forward interactively:
pick a target flow rate, trigger an occlusion event or a pressure bump
mid-episode, and watch each controller's flow trace and command signal
update on the same chart.

```bash
streamlit run app.py
```

Requires `results/qlearning_qtable.pkl` and `results/dqn_model.pt` to exist
(train them first, see above) — the app will still run PID-only with a
warning if either is missing.

## Analysis: strengths and limitations

**Where PID still wins, and why:** the plant is close to linear and
time-invariant around any operating point, which is the regime PID was
designed for, and its continuous-valued output avoids the persistent
discrete-action chatter visible in both RL agents' traces. Adding the
10-step transport delay didn't change that structural advantage — it forced
PID to detune (`2.0/0.4/0.5` -> `0.6/0.08/0.2`), which cost it settling speed
and IAE/ISE/RMSE headroom, but PID is still the tightest-settling controller
of the three (best settling time, lowest overshoot, most time in the safe
band). A model-based dead-time compensator (Smith predictor) bolted onto
PID would likely recover more of that headroom without needing RL at all —
we did not build one here, since the point of this experiment was to see
what a delay-aware *state feature* buys a learned policy, not to build the
best possible classical controller.

**Where the transport delay changes the RL picture:** delay is exactly the
kind of thing tabular discretization *and* a memoryless feature set both
handle badly — PID has no internal notion of "how much correction is already
in flight," and neither did the original 2-feature RL state. Giving RL
agents a third/fourth feature (`pending_correction`, the actual queued
correction) that PID structurally cannot use is what lets DQN pull ahead of
PID on time-averaged tracking error (IAE/ISE/RMSE) for the first time in
this project. That is a genuine, measured result, not an artifact of retuning
reward or plant in RL's favor — the reward and plant ODE are unchanged from
the pre-delay version; only the delay, the PID gains it forces, and the new
state feature changed.

**Why that isn't a clean RL win, though:** the same **discrete** 7-action
space that hurt tabular Q-learning before still caps DQN's settling
precision now — a function approximator over a richer continuous state fixes
the *discretization* problem (hence lower average error) but not the
*discrete-action* problem (hence still-worse settling time/overshoot/safe-
band-time than PID). A next step that could plausibly close *that* gap is a
continuous-action method (DDPG/TD3, as used in the companion glucose-
regulation project) rather than another discrete-action approximator.

**Where tabular Q-learning stands now:** it lost to PID before the delay and
still loses to PID on every metric after it (IAE 6032.9 vs 4866.5, etc.) even
with the same `pending_correction` feature available (through its
discretized state) and 3x the training episodes (15,000 vs. the original
5,000) needed just to fill a table that's now 7x larger (1,617 states vs.
231). Its role in this project remains pedagogical: it demonstrates the RL
formulation and where a flat table structurally can't keep up, not a
competitive controller for this plant.

## Repo layout

```
src/
  environment.py       # plant ODE, disturbance scheduler, RL env
  q_learning_agent.py  # tabular Q-learning
  dqn_agent.py          # DQN (PyTorch) over continuous state, same env/reward
  pid_controller.py    # PID baseline
  metrics.py           # IAE/ISE/RMSE/settling-time/overshoot
  train.py             # trains + saves Q-table + training curve
  evaluate.py          # RL vs PID (vs DQN) on held-out disturbance regime
app.py                  # Streamlit: live PID vs Q-learning vs DQN comparison
results/                # training curves + comparison plots (generated)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --episodes 15000 --seed 0
python -m src.dqn_agent --episodes 2000 --seed 0
python -m src.evaluate --n_eval 30 --dqn_model results/dqn_model.pt
streamlit run app.py
```
