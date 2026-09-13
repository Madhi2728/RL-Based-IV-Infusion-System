"""
environment.py
---------------
Mathematical simulation of a simulated IV infusion flow process, wrapped as an
RL environment (Gym-style API: reset() / step()) for a tabular Q-learning agent.

PLANT MODEL
===========
A volumetric infusion pump commands a flow rate via a control signal u(t)
(effectively "desired pump duty" in mL/hr). The *actual* delivered flow Q(t)
lags the command because of line compliance, tubing resistance and fluid
inertia, and is corrupted by physical disturbances that a real IV line
experiences:

    tau * dQ/dt = K_eff(t) * u(t) - Q(t) + d_p(t)

  - tau      : first-order lag time constant (compliance/inertia of the line)
  - K_eff(t) : effective gain, nominally 1.0, reduced by partial occlusions
               (kinked line, positional occlusion) -> K_eff < 1
  - d_p(t)   : additive disturbance from hydrostatic pressure changes
               (bag height changes, patient arm movement, venous back-pressure)

Disturbances modeled:
  1. Partial occlusion: K_eff drops (e.g. to 0.4-0.8) for a random window,
     representing a kink or positional occlusion.
  2. Full/near-full occlusion spike: K_eff drops sharply and briefly (alarm-
     worthy event), tests robustness.
  3. Hydrostatic bias: d_p(t) shifts (bag raised/lowered, patient moves),
     applied as a slowly varying additive term.
  4. Sensor/measurement noise on the observed flow.

The controller (RL agent or PID) only manipulates u(t); it does not know
K_eff(t) or d_p(t) directly -- it only observes the resulting flow error,
exactly like a real infusion pump's closed-loop drop-rate control.

Discretized with Euler integration at dt seconds per control step.

TRANSPORT DELAY
===============
Real IV lines have a meaningful dead time between a pump command changing and
that change reaching the patient end of the tubing (line length, pump
mechanism response, sensor placement). This is modeled as a pure transport
delay of `delay_steps * dt` seconds: the command issued at time t only starts
affecting the plant ODE at time t + delay.

This delay is what makes the control problem genuinely hard for PID: a
fixed-gain PID has no internal model of "how much correction is already in
flight but hasn't landed yet," so with a large delay-to-time-constant ratio it
either has to be detuned (slow, sluggish) or it overcorrects on stale error
and oscillates/rings. Classical process control handles this with a
model-based dead-time compensator (a Smith predictor) bolted onto the PID --
but that is a structural redesign, not a tuning fix. An RL agent, by
contrast, can be given a state feature that exposes exactly this "control
already in the pipe" quantity, and learn to act on it without any explicit
delay model -- so this environment gives the RL state that extra feature
while the PID baseline remains the standard, undesigned-for-delay controller
a real deployment would actually use.
"""

import numpy as np


class IVInfusionPlant:
    """Continuous-time plant dynamics, Euler-integrated, with transport delay."""

    def __init__(self, tau=8.0, dt=1.0, u_max=400.0, q_max=500.0, noise_std=0.5,
                 delay_steps=6, seed=None):
        self.tau = tau
        self.dt = dt
        self.u_max = u_max      # max pump command, mL/hr
        self.q_max = q_max      # max physically deliverable flow, mL/hr
        self.noise_std = noise_std
        self.delay_steps = delay_steps
        self.rng = np.random.default_rng(seed)

        self.Q = 0.0            # actual delivered flow (mL/hr)
        self.u = 0.0            # current (just-issued) pump command (mL/hr)
        self.u_applied = 0.0    # command actually driving the ODE right now (delayed)
        self.k_eff = 1.0        # effective line gain (occlusion factor)
        self.d_p = 0.0          # hydrostatic/back-pressure disturbance
        self._cmd_queue = [0.0] * delay_steps

    def reset(self, start_flow=0.0):
        self.Q = start_flow
        self.u = start_flow
        self.u_applied = start_flow
        self.k_eff = 1.0
        self.d_p = 0.0
        self._cmd_queue = [start_flow] * self.delay_steps
        return self.Q

    def set_command(self, u):
        self.u = float(np.clip(u, 0.0, self.u_max))

    def apply_occlusion(self, factor):
        """factor in (0,1]; 1.0 = clear line, <1 = partial/full occlusion."""
        self.k_eff = float(np.clip(factor, 0.0, 1.0))

    def apply_pressure_disturbance(self, d_p):
        self.d_p = float(d_p)

    def pending_correction(self):
        """How much commanded change hasn't reached the plant yet (u - u_applied).
        Exposed to the RL state only -- PID never sees this."""
        return self.u - self.u_applied

    def step(self):
        # push latest command into the delay line, pop the one that lands now
        self._cmd_queue.append(self.u)
        self.u_applied = self._cmd_queue.pop(0)

        dQ = (self.k_eff * self.u_applied - self.Q + self.d_p) / self.tau
        self.Q = self.Q + self.dt * dQ
        self.Q = float(np.clip(self.Q, 0.0, self.q_max))
        measured_Q = self.Q + self.rng.normal(0.0, self.noise_std)
        return float(np.clip(measured_Q, 0.0, self.q_max))


class DisturbanceScheduler:
    """Generates randomized occlusion + pressure-disturbance timelines per episode."""

    def __init__(self, episode_len, dt, rng, mode="train"):
        self.episode_len = episode_len
        self.dt = dt
        self.rng = rng
        self.mode = mode  # "train" -> disturbance distribution seen in training
                           # "eval"  -> shifted distribution, unseen severity/timing

    def generate(self):
        n_steps = int(self.episode_len / self.dt)
        k_eff = np.ones(n_steps)
        d_p = np.zeros(n_steps)

        if self.mode == "train":
            n_occlusions = self.rng.integers(0, 3)
            occ_range = (0.35, 0.85)
            pressure_amp = 15.0
        else:  # eval: harder / novel disturbance regime
            n_occlusions = self.rng.integers(1, 4)
            occ_range = (0.15, 0.75)   # includes more severe occlusions
            pressure_amp = 25.0

        for _ in range(n_occlusions):
            start = self.rng.integers(0, max(n_steps - 5, 1))
            dur = self.rng.integers(5, max(int(n_steps * 0.25), 6))
            end = min(start + dur, n_steps)
            severity = self.rng.uniform(*occ_range)
            k_eff[start:end] = severity

        # slow-varying hydrostatic bias (bag height / arm movement)
        n_bumps = self.rng.integers(1, 3)
        for _ in range(n_bumps):
            center = self.rng.integers(0, n_steps)
            width = self.rng.integers(20, max(int(n_steps * 0.4), 21))
            amp = self.rng.uniform(-pressure_amp, pressure_amp)
            idx = np.arange(n_steps)
            d_p += amp * np.exp(-0.5 * ((idx - center) / max(width, 1)) ** 2)

        return k_eff, d_p


class IVInfusionEnv:
    """
    RL environment around IVInfusionPlant.

    STATE (discretized for tabular Q-learning):
        e      = target_flow - measured_flow             (error, mL/hr)
        de     = e - e_prev                               (error rate)
        p      = u - u_applied                            (pending correction
                                                             still "in flight"
                                                             through the delay
                                                             line -- PID never
                                                             sees this feature)
      -> each binned into discrete buckets -> single integer state index

    ACTION (discrete pump-command adjustments, mL/hr):
        [-8, -4, -1, 0, +1, +4, +8]   applied to the current command u

    REWARD:
        r = -|e|/scale                        (tracking accuracy)
            - w_du * |du|                     (penalize jerky commands -> smooth,
                                                 clinically safer flow changes)
            - w_over * max(0, |e| - safe_band) (extra penalty once error exceeds
                                                 a clinically-acceptable band,
                                                 modeling overdose/underdose risk)
            + bonus if |e| < tight_band        (settled / in-spec bonus)
    """

    ACTIONS = np.array([-8, -4, -1, 0, 1, 4, 8], dtype=float)

    def __init__(self, dt=1.0, episode_len=180.0, seed=None,
                 err_bins=21, derr_bins=11, pending_bins=7,
                 err_clip=60.0, derr_clip=15.0, pending_clip=40.0,
                 delay_steps=10, mode="train"):
        self.dt = dt
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)
        self.plant = IVInfusionPlant(dt=dt, delay_steps=delay_steps, seed=seed)
        self.mode = mode
        self.scheduler = DisturbanceScheduler(episode_len, dt, self.rng, mode=mode)

        self.err_bins = err_bins
        self.derr_bins = derr_bins
        self.pending_bins = pending_bins
        self.err_clip = err_clip
        self.derr_clip = derr_clip
        self.pending_clip = pending_clip

        self.safe_band = 5.0     # mL/hr - clinically acceptable error band
        self.tight_band = 1.5    # mL/hr - "settled" band for bonus

        self.n_states = err_bins * derr_bins * pending_bins
        self.n_actions = len(self.ACTIONS)

        self.target = 100.0
        self.k_eff_schedule = None
        self.d_p_schedule = None
        self.step_idx = 0
        self.n_steps = int(episode_len / dt)
        self.prev_err = 0.0

        self.history = []  # list of dicts per step, for plotting/eval

    # ---------- target profile ----------
    def _sample_target_profile(self):
        """Piecewise-constant target flow rate, changes 1-2 times per episode."""
        n_changes = self.rng.integers(0, 2)
        targets = [self.rng.uniform(50, 200)]
        change_points = sorted(self.rng.choice(range(20, self.n_steps), size=n_changes, replace=False)) \
            if n_changes > 0 else []
        for _ in range(n_changes):
            targets.append(self.rng.uniform(50, 200))
        return targets, change_points

    def reset(self):
        self.plant.reset(start_flow=self.rng.uniform(40, 160))
        self.k_eff_schedule, self.d_p_schedule = self.scheduler.generate()
        self.targets, self.change_points = self._sample_target_profile()
        self.target = self.targets[0]
        self._target_ptr = 0
        self.step_idx = 0
        self.history = []

        measured = self.plant.Q
        err = self.target - measured
        self.prev_err = err
        state = self._discretize(err, 0.0, self.plant.pending_correction())
        return state

    def _discretize(self, err, derr, pending):
        e_idx = int(np.clip((err + self.err_clip) / (2 * self.err_clip) * self.err_bins,
                             0, self.err_bins - 1))
        de_idx = int(np.clip((derr + self.derr_clip) / (2 * self.derr_clip) * self.derr_bins,
                              0, self.derr_bins - 1))
        p_idx = int(np.clip((pending + self.pending_clip) / (2 * self.pending_clip) * self.pending_bins,
                             0, self.pending_bins - 1))
        return (e_idx * self.derr_bins + de_idx) * self.pending_bins + p_idx

    def step(self, action_idx):
        du = self.ACTIONS[action_idx]
        new_u = self.plant.u + du
        self.plant.set_command(new_u)

        # advance target if a scheduled change point is reached
        if self._target_ptr < len(self.change_points) and self.step_idx >= self.change_points[self._target_ptr]:
            self._target_ptr += 1
            self.target = self.targets[self._target_ptr]

        # apply this step's disturbance
        self.plant.apply_occlusion(self.k_eff_schedule[self.step_idx])
        self.plant.apply_pressure_disturbance(self.d_p_schedule[self.step_idx])
        measured = self.plant.step()

        err = self.target - measured
        derr = err - self.prev_err

        reward = -abs(err) / 20.0
        reward -= 0.02 * abs(du)
        if abs(err) > self.safe_band:
            reward -= 0.15 * (abs(err) - self.safe_band)
        if abs(err) < self.tight_band:
            reward += 0.5

        self.history.append(dict(
            t=self.step_idx * self.dt, target=self.target, measured=measured,
            command=self.plant.u, error=err, k_eff=self.plant.k_eff, d_p=self.plant.d_p,
        ))

        self.prev_err = err
        self.step_idx += 1
        done = self.step_idx >= self.n_steps
        next_state = self._discretize(err, derr, self.plant.pending_correction())
        return next_state, reward, done, {}
