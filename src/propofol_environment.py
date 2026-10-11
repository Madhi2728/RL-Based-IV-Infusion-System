"""
propofol_environment.py
------------------------

MODEL: propofol anesthesia depth (BIS) -- Schnider 3-compartment PK + Emax PD
=============================================================================
PK (amounts A1/A2/A3 in mg, RK4, dt=5 s per control step):
  dA1/dt = -(k10 + k12 + k13)*A1 + k21*A2 + k31*A3 + u(t)     u = infusion, mg/s
  dA2/dt =  k12*A1 - k21*A2
  dA3/dt =  k13*A1 - k31*A3
  Cp     =  A1 / V1                  (mg / L == ug/mL)
  dCe/dt =  ke0 * (Cp - Ce)          (effect site, integrated in the same RK4 step)
  amounts / concentrations clipped >= 0 after every step

Schnider covariates (age yr, weight kg, height cm, sex), James-formula LBM:
  LBM male   = 1.10*weight - 128*(weight/height)^2
  LBM female = 1.07*weight - 148*(weight/height)^2
  V1 = 4.27 L        V2 = 18.9 - 0.391*(age-53) L        V3 = 238 L
  Cl1 = 1.89 + 0.0456*(weight-77) - 0.0681*(LBM-59) + 0.0264*(height-177) L/min
  Cl2 = 1.29 - 0.024*(age-53) L/min                      Cl3 = 0.836 L/min
  k10=Cl1/V1  k12=Cl2/V1  k13=Cl3/V1  k21=Cl2/V2  k31=Cl3/V3   (1/min, /60 -> 1/s)
  ke0 = 0.456 /min (/60 -> 1/s)
  Guards (never active inside the env's patient ranges): Cl1 >= 0.1, Cl2 >= 0.05,
  V2 >= 1.0 -- only to keep the ODE well-posed for out-of-range slider input.
  Nominal (reference) patient: age 53, 77 kg, 177 cm, male.

PD (BISModel):
  BIS = E0 - Emax * Ce^g / (Ce50_eff^g + Ce^g)  + N(0, 1.5),  clipped to [0, 100]
  E0=95  Emax=95  Ce50=4.7 ug/mL (x U(0.85, 1.15) per episode)
                  gamma=2.0     (x U(0.80, 1.20) per episode)
  Ce50_eff = Ce50 * (1 + stimulus_shift(t))
  (v1 used sigma 2.0 and Ce50 +/-25 %; narrowed in v2 -- see DESIGN.md)

Surgical stimulus (StimulusScheduler)      TRAIN           EVAL
  events per episode                       2 - 4           3 - 4
  Ce50 upward shift per event              15 - 30 %       20 - 30 %
  event duration                           2 - 5 min       3 - 5 min
  (overlapping events -> max shift, not compounded; start uniform over episode)

Plant check (scratch numbers, constant infusion, Ce for BIS 50 = 4.46 ug/mL):
  nominal 53y/77kg/177cm/M : holding Ce ~4.5 at 30 min needs ~10 mg/kg/h;
                             12 mg/kg/h reaches it only at 11.8 min
  20y/54kg/161cm/F         : needs ~14.8 mg/kg/h at 30 min; 12 never gets there
  worst slider corner (20y/50kg/195cm/F): ~18.7 mg/kg/h at 30 min, more under
                             stimulation -> the original 0-12 mg/kg/h action
                             range could not reach BIS 50 for much of the
                             patient range. Units were checked and are correct
                             (2 mg/kg bolus -> Cp 34.7 ug/mL at 5 s, peak Ce
                             9.1 ug/mL at 1.6 min, matching Schnider's small V1).

RL environment (PropofolEnv)
  state   = (BIS_error, dBIS/dt, Ce, last_action, bolus-window phase)
              BIS_error = target - measured BIS        (same sign as IV env)
              dBIS/dt   = slope of EMA-filtered BIS (alpha 0.15), BIS/min
  bins    = 15 x 3 x 12 x 9 x 2 = 9720 states
  clips   = |BIS_error| 22.5 (3-BIS bins), |dBIS/dt| 6 BIS/min (3 bins),
            Ce 0-9 ug/mL (0.75 bins), last action 0..8, phase 0/1
  actions = 8 ABSOLUTE infusion levels [0, 4, 7, 10, 14, 19, 26, 40] mg/kg/h
            + action 8 = 1 mg/kg induction bolus (first 3 min; afterwards it
            acts as the top level) -> self.ACTIONS in mg/s for the current
            patient's weight
  episode = 3600 s = 720 steps (dt=5 s); done early if measured BIS < 20
  target  = 60 for 0-10 min (induction), linear ramp 60->50 over 10-15 min,
            50 thereafter; with p=0.5 one extra step at t ~ U(25, 50) min to
            50 +/- U(5, 10) (stays inside 40-60) held to episode end
  reward  = +1 if |e| < 5 else -0.1*(|e| - 5)
            - 2 if BIS < 40 or BIS > 60
            - 0.1*|d level|   (change of infusion level between consecutive
                               steps; fractional for PID's continuous rate)
            - 200 on early termination (BIS < 20)
  DQN obs = continuous_state(): [BIS_error, dBIS/dt, Ce, last_action,
            time_since_start (/3600 s), target_BIS] normalized
            -> PROPOFOL_STATE_DIM = 6
  obs_mode="bis" (v3, robustness study): no Ce anywhere --
            tabular = err 15 x dBIS/dt 5 x last action 9 x time bin 4
                      (0-3 | 3-10 | 10-30 | 30+ min) = 2700 states
            DQN     = [BIS_error, dBIS/dt, last_action, time, target_BIS]
                      -> PROPOFOL_BIS_STATE_DIM = 5
  ce50_jitter (default 0.15) sets the per-episode Ce50 spread; the
            robustness study uses 0.30 with bis_noise_std=2.0

Plain-Python env with reset() -> state / step(a) -> (s', r, done, info), the
same shape as IVInfusionEnv (no gymnasium -- see DESIGN.md).

INFORMATION BOUNDARY
====================
Every controller sees measured BIS. Ce is what a target-controlled-infusion
(TCI) pump displays: the Schnider model's prediction from the patient's
covariates and the infusion history. Because this simulator models no
pharmacokinetic inter-patient variability (all patient-to-patient
variability lives in the PD: Ce50, gamma, stimulus), the TCI prediction is
exactly the plant's Ce -- effectively oracle plant state. The default PID
(BISPID, src/propofol_pid_tune.py) therefore never reads it; obs_mode="ce" RL
agents (v2, the app's models) do, obs_mode="bis" agents (robustness study
headline) do not. The PD side -- Ce50, gamma, and the stimulus shift -- is
never exposed to any controller.

SEEDING
=======
`np.random.default_rng(seed)` is used ONLY to draw one episode seed per
reset(). Each episode then builds independent streams (patient, PD jitter,
stimulus schedule, target schedule, BIS noise) from SeedSequence(episode
seed).spawn(5). So for a given constructor seed, the k-th reset() yields the
same patient / PD / stimulus / target / noise sequence no matter which
controller drives the env, what actions it takes, or how early an episode
ends -- the basis of the fair PID vs Q-learning vs DQN comparison.
"""

import numpy as np


PROPOFOL_STATE_DIM = 6      # obs_mode="ce" : [BIS_error, dBIS/dt, Ce, last_action, time, target_BIS]
PROPOFOL_BIS_STATE_DIM = 5  # obs_mode="bis": [BIS_error, dBIS/dt, last_action, time, target_BIS]

NOMINAL_PATIENT = dict(age=53.0, weight=77.0, height=177.0, sex="male")


def _clip(x, lo, hi):
    """Scalar clip (np.clip on Python floats dominates step() time otherwise)."""
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------------------
# PK plant
# ---------------------------------------------------------------------------
def lean_body_mass(weight, height, sex):
    """James-formula lean body mass (kg); weight in kg, height in cm."""
    if sex == "female":
        return 1.07 * weight - 148.0 * (weight / height) ** 2
    return 1.10 * weight - 128.0 * (weight / height) ** 2


def schnider_parameters(age, weight, height, sex):
    """Schnider propofol PK parameters. Volumes in L, clearances in L/min,
    rate constants in 1/min (convert to 1/s with /60)."""
    lbm = lean_body_mass(weight, height, sex)
    v1 = 4.27
    v2 = max(1.0, 18.9 - 0.391 * (age - 53.0))
    v3 = 238.0
    cl1 = max(0.1, 1.89 + 0.0456 * (weight - 77.0) - 0.0681 * (lbm - 59.0)
              + 0.0264 * (height - 177.0))
    cl2 = max(0.05, 1.29 - 0.024 * (age - 53.0))
    cl3 = 0.836
    return dict(
        lbm=lbm, v1=v1, v2=v2, v3=v3, cl1=cl1, cl2=cl2, cl3=cl3,
        k10=cl1 / v1, k12=cl2 / v1, k13=cl3 / v1, k21=cl2 / v2, k31=cl3 / v3,
        ke0=0.456,
    )


class SchniderPlant:
    """3-compartment Schnider PK + effect-site compartment, RK4-integrated.

    State: A1, A2, A3 (mg) and Ce (ug/mL). Infusion u is held constant over
    each dt-second step (zero-order hold, like a pump rate setting).

        dA1/dt = -(k10+k12+k13)*A1 + k21*A2 + k31*A3 + u
        dA2/dt =  k12*A1 - k21*A2
        dA3/dt =  k13*A1 - k31*A3
        dCe/dt =  ke0*(A1/V1 - Ce)

    All rate constants are stored per second.
    """

    def __init__(self, age=53.0, weight=77.0, height=177.0, sex="male", dt=5.0):
        self.dt = float(dt)
        self.set_patient(age, weight, height, sex)
        self.reset()

    def set_patient(self, age, weight, height, sex):
        self.age, self.weight, self.height, self.sex = float(age), float(weight), float(height), sex
        p = schnider_parameters(self.age, self.weight, self.height, sex)
        self.params = p
        self.v1 = p["v1"]
        self.k10, self.k12, self.k13 = p["k10"] / 60.0, p["k12"] / 60.0, p["k13"] / 60.0
        self.k21, self.k31 = p["k21"] / 60.0, p["k31"] / 60.0
        self.ke0 = p["ke0"] / 60.0

    def reset(self):
        self.A1 = self.A2 = self.A3 = 0.0
        self.Ce = 0.0
        self.u = 0.0
        return self.Cp

    @property
    def Cp(self):
        return self.A1 / self.v1

    def _deriv(self, a1, a2, a3, ce, u):
        da1 = -(self.k10 + self.k12 + self.k13) * a1 + self.k21 * a2 + self.k31 * a3 + u
        da2 = self.k12 * a1 - self.k21 * a2
        da3 = self.k13 * a1 - self.k31 * a3
        dce = self.ke0 * (a1 / self.v1 - ce)
        return da1, da2, da3, dce

    def step(self, u_mg_per_s):
        """Advance one dt with constant infusion u (mg/s). Returns (Cp, Ce)."""
        u = max(0.0, float(u_mg_per_s))
        self.u = u
        h = self.dt
        a1, a2, a3, ce = self.A1, self.A2, self.A3, self.Ce
        f = self._deriv
        k1 = f(a1, a2, a3, ce, u)
        k2 = f(a1 + 0.5 * h * k1[0], a2 + 0.5 * h * k1[1], a3 + 0.5 * h * k1[2], ce + 0.5 * h * k1[3], u)
        k3 = f(a1 + 0.5 * h * k2[0], a2 + 0.5 * h * k2[1], a3 + 0.5 * h * k2[2], ce + 0.5 * h * k2[3], u)
        k4 = f(a1 + h * k3[0], a2 + h * k3[1], a3 + h * k3[2], ce + h * k3[3], u)
        w = h / 6.0
        self.A1 = max(0.0, a1 + w * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]))
        self.A2 = max(0.0, a2 + w * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
        self.A3 = max(0.0, a3 + w * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))
        self.Ce = max(0.0, ce + w * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]))
        return self.Cp, self.Ce


# ---------------------------------------------------------------------------
# PD model
# ---------------------------------------------------------------------------
class BISModel:
    """Sigmoid Emax (Hill) BIS model with per-episode Ce50/gamma jitter.

        BIS = E0 - Emax * Ce^g / (Ce50_eff^g + Ce^g),   Ce50_eff = Ce50*(1+shift)
        measured = clip(BIS + N(0, noise_std), 0, 100)
    """

    E0 = 95.0
    EMAX = 95.0
    CE50 = 4.7
    GAMMA = 2.0

    def __init__(self, ce50=CE50, gamma=GAMMA, noise_std=1.5, rng=None):
        self.ce50 = float(ce50)
        self.gamma = float(gamma)
        self.noise_std = float(noise_std)
        self.rng = rng if rng is not None else np.random.default_rng()

    def jitter(self, rng, ce50_frac=0.15, gamma_frac=0.20):
        """Draw this episode's patient sensitivity: Ce50 +/-15 %, gamma +/-20 %."""
        self.ce50 = self.CE50 * rng.uniform(1.0 - ce50_frac, 1.0 + ce50_frac)
        self.gamma = self.GAMMA * rng.uniform(1.0 - gamma_frac, 1.0 + gamma_frac)

    def nominal(self):
        self.ce50, self.gamma = self.CE50, self.GAMMA

    def true_bis(self, ce, ce50_shift=0.0):
        ce50_eff = self.ce50 * (1.0 + ce50_shift)
        ce_g = max(ce, 0.0) ** self.gamma
        return self.E0 - self.EMAX * ce_g / (ce50_eff ** self.gamma + ce_g)

    @classmethod
    def nominal_ce_for_bis(cls, bis):
        """Inverse of the NOMINAL curve (Ce50=4.7, gamma=2): the Ce that gives
        `bis` in a typical patient. Used by the cascade PID's outer loop."""
        bis = _clip(float(bis), 1.0, cls.E0 - 1e-6)
        frac = (cls.E0 - bis) / cls.EMAX            # = Ce^g / (Ce50^g + Ce^g)
        return cls.CE50 * (frac / (1.0 - frac)) ** (1.0 / cls.GAMMA)

    def measure(self, ce, ce50_shift=0.0):
        bis = self.true_bis(ce, ce50_shift)
        if self.noise_std > 0:
            bis += self.rng.normal(0.0, self.noise_std)
        return _clip(bis, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Disturbances and target profile
# ---------------------------------------------------------------------------
class StimulusScheduler:
    """Randomized surgical-stimulus timeline: per-step fractional upward shift
    of Ce50 (patient temporarily needs more drug for the same BIS).

    TRAIN: 2-4 events, +15-30 % Ce50, 2-5 min each (the specified envelope).
    EVAL : 3-4 events, +20-30 % Ce50, 3-5 min each (harsh end of the same
           envelope -- tests robustness without leaving the specified range).
    Overlapping events take the max shift (not compounded).
    """

    def __init__(self, n_steps, dt, rng, mode="train"):
        self.n_steps = n_steps
        self.dt = dt
        self.rng = rng
        self.mode = mode

    def generate(self):
        shift = np.zeros(self.n_steps)
        if self.mode == "train":
            n_events = self.rng.integers(2, 5)
            shift_range, dur_range_s = (0.15, 0.30), (120.0, 300.0)
        else:
            n_events = self.rng.integers(3, 5)
            shift_range, dur_range_s = (0.20, 0.30), (180.0, 300.0)
        for _ in range(n_events):
            dur = int(round(self.rng.uniform(*dur_range_s) / self.dt))
            start = int(self.rng.integers(0, max(self.n_steps - dur, 1)))
            amp = self.rng.uniform(*shift_range)
            shift[start:start + dur] = np.maximum(shift[start:start + dur], amp)
        return shift


INDUCTION_TARGET = 60.0
MAINTENANCE_TARGET = 50.0
RAMP_START_S, RAMP_END_S = 600.0, 900.0      # 10 -> 15 min
EXTRA_STEP_WINDOW_S = (1500.0, 3000.0)       # 25 -> 50 min
EXTRA_STEP_SIZE = (5.0, 10.0)                # |jump| from maintenance, BIS units


def build_target_schedule(n_steps, dt, induction=INDUCTION_TARGET, maintenance=MAINTENANCE_TARGET,
                          extra_step=None):
    """Per-step target BIS array plus the list of target-change step indices
    used for settling-time metrics. extra_step = (step_idx, new_level) or None."""
    t = np.arange(n_steps) * dt
    frac = np.clip((t - RAMP_START_S) / (RAMP_END_S - RAMP_START_S), 0.0, 1.0)
    sched = induction + (maintenance - induction) * frac
    change_steps = [int(RAMP_END_S / dt)]
    if extra_step is not None:
        idx, level = extra_step
        sched[idx:] = level
        change_steps.append(int(idx))
    return sched, change_steps


# ---------------------------------------------------------------------------
# RL environment
# ---------------------------------------------------------------------------
class PropofolEnv:
    """
    RL environment around SchniderPlant + BISModel + StimulusScheduler,
    mirroring IVInfusionEnv's reset()/step() API.

    STATE (discretized for tabular Q-learning), 15 x 3 x 12 x 9 x 2 = 9720:
        e      = target_BIS - measured_BIS, 15 bins over +/-22.5 (3 BIS each)
        de     = slope of EMA-filtered BIS, 3 bins over +/-6 BIS/min
                 (falling < -2 / flat / rising > +2)
        Ce     = effect-site concentration, 12 bins over 0-9 ug/mL (0.75 each)
        action = last action index 0..8 (8 = bolus)
        phase  = 1 while the induction-bolus window (first 3 min) is open

    ACTIONS (9):
        0..7 : ABSOLUTE infusion levels RATE_LEVELS_MG_KG_H =
               [0, 4, 7, 10, 14, 19, 26, 40] mg/kg/h (roughly log-spaced, so
               the step size is proportional across the 3x range of
               maintenance needs in the patient ranges)
        8    : induction bolus, 1 mg/kg delivered within one 5-s step, only
               during the first 3 min; after that it acts as level 7
        NOTE: absolute rate levels, NOT deltas like IVInfusionEnv.ACTIONS --
        propofol pumps are commanded with an absolute rate. self.ACTIONS holds
        the per-action mg/s for the current patient's weight (bolus = its
        5-s equivalent rate).

    REWARD (per 5-s step):
        +1                     if |BIS - target| < 5
        -0.1 * (|e| - 5)       otherwise
        -2                     if BIS < 40 or BIS > 60
        -0.1 * |d level|       change of infusion level between consecutive
                               steps (fractional for PID's continuous rate;
                               the bolus step itself is not penalized)
        -200                   on early termination (BIS < 20)
    """

    RATE_LEVELS_MG_KG_H = np.array([0.0, 4.0, 7.0, 10.0, 14.0, 19.0, 26.0, 40.0])
    MAX_RATE_MG_KG_H = 40.0
    BOLUS_MG_PER_KG = 1.0
    BOLUS_WINDOW_S = 180.0
    CONC_SCALE = 5.0  # ug/mL, ~Ce50, normalizes Ce for DQN
    TIME_BIN_EDGES_S = (0.0, 180.0, 600.0, 1800.0)  # obs_mode="bis" time bins (lower edges)
    TIME_SCALE_S = 3600.0                           # normalizes the DQN time feature

    def __init__(self, dt=5.0, episode_len=3600.0, seed=None, mode="train",
                 err_bins=15, dbis_bins=3, ce_bins=12, err_clip=22.5, dbis_clip=6.0, ce_max=9.0,
                 bis_noise_std=1.5, dbis_filter_alpha=0.15,
                 patient=None, randomize_pd=True, stimuli=True, extra_target_steps=True,
                 du_penalty=0.1, terminal_penalty=200.0, terminate_bis=20.0,
                 ce50_jitter=0.15, obs_mode="ce"):
        self.dt = float(dt)
        self.episode_len = float(episode_len)
        self.n_steps = int(round(episode_len / dt))
        self.mode = mode
        self.rng = np.random.default_rng(seed)  # draws one episode seed per reset()

        # obs_mode="ce" : the v2 observations above (Ce from the TCI model)
        # obs_mode="bis": BIS-derived features only -- what a BIS monitor plus
        #                 the pump's own action log provide (no Ce):
        #   tabular = err 15 x dBIS/dt 5 (over +/-6 BIS/min) x last action 9
        #             x time bin 4 (0-3 | 3-10 | 10-30 | 30+ min) = 2700 states
        #   DQN     = [BIS_error, dBIS/dt, last_action, time, target_BIS] (5)
        if obs_mode not in ("ce", "bis"):
            raise ValueError(f"obs_mode must be 'ce' or 'bis', got {obs_mode!r}")
        self.obs_mode = obs_mode
        if obs_mode == "bis" and dbis_bins == 3:
            dbis_bins = 5
        self.ce50_jitter = ce50_jitter
        self.err_bins, self.dbis_bins, self.ce_bins = err_bins, dbis_bins, ce_bins
        self.err_clip, self.dbis_clip, self.ce_max = err_clip, dbis_clip, ce_max
        self.n_levels = len(self.RATE_LEVELS_MG_KG_H)
        self.BOLUS_ACTION = self.n_levels
        self.n_actions = self.n_levels + 1
        if obs_mode == "ce":
            self.n_states = err_bins * dbis_bins * ce_bins * self.n_actions * 2
            self.state_dim = PROPOFOL_STATE_DIM
        else:
            self.n_states = err_bins * dbis_bins * self.n_actions * len(self.TIME_BIN_EDGES_S)
            self.state_dim = PROPOFOL_BIS_STATE_DIM

        self.bis_noise_std = bis_noise_std
        self.dbis_filter_alpha = dbis_filter_alpha
        self.fixed_patient = dict(patient) if patient is not None else None
        self.randomize_pd = randomize_pd
        self.stimuli = stimuli
        self.extra_target_steps = extra_target_steps
        self.du_penalty = du_penalty
        self.terminal_penalty = terminal_penalty
        self.terminate_bis = terminate_bis
        self.target_band = 5.0
        self.safe_low, self.safe_high = 40.0, 60.0

        p = self.fixed_patient or NOMINAL_PATIENT
        self.patient = dict(p)
        self.plant = SchniderPlant(dt=self.dt, **self.patient)
        self.bis_model = BISModel(noise_std=bis_noise_std)
        self._set_actions()

        self.stim_shift = np.zeros(self.n_steps)
        self.target_schedule = np.full(self.n_steps, INDUCTION_TARGET)
        self.target_change_steps = []
        self.target = INDUCTION_TARGET
        self.step_idx = 0
        self.history = []

    def _set_actions(self):
        w = self.patient["weight"]
        self.ACTIONS = np.append(self.RATE_LEVELS_MG_KG_H * w / 3600.0,
                                 self.BOLUS_MG_PER_KG * w / self.dt)

    # ---------- per-episode randomization ----------
    def _sample_patient(self, rng):
        return dict(age=float(rng.uniform(20, 80)), weight=float(rng.uniform(50, 110)),
                    height=float(rng.uniform(150, 195)),
                    sex="male" if rng.random() < 0.5 else "female")

    def _sample_extra_step(self, rng):
        if not self.extra_target_steps or rng.integers(0, 2) == 0:
            return None
        t_change = rng.uniform(*EXTRA_STEP_WINDOW_S)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        level = MAINTENANCE_TARGET + sign * rng.uniform(*EXTRA_STEP_SIZE)
        return int(t_change / self.dt), float(level)

    def reset(self):
        episode_seed = int(self.rng.integers(0, 2**63 - 1))
        pat_rng, pd_rng, stim_rng, tgt_rng, noise_rng = (
            np.random.default_rng(s) for s in np.random.SeedSequence(episode_seed).spawn(5))

        self.patient = dict(self.fixed_patient) if self.fixed_patient else self._sample_patient(pat_rng)
        self.plant.set_patient(**self.patient)
        self.plant.reset()
        self._set_actions()

        self.bis_model.rng = noise_rng
        if self.randomize_pd:
            self.bis_model.jitter(pd_rng, ce50_frac=self.ce50_jitter)
        else:
            self.bis_model.nominal()

        if self.stimuli:
            self.stim_shift = StimulusScheduler(self.n_steps, self.dt, stim_rng, self.mode).generate()
        else:
            self.stim_shift = np.zeros(self.n_steps)

        self.target_schedule, self.target_change_steps = build_target_schedule(
            self.n_steps, self.dt, extra_step=self._sample_extra_step(tgt_rng))

        self.step_idx = 0
        self.history = []
        self.target = float(self.target_schedule[0])
        self.bis = self.bis_model.measure(self.plant.Ce, self.stim_shift[0])
        self.bis_filt = self.bis
        self.dbis = 0.0
        self.err = self.target - self.bis
        self.last_rate = 0.0
        self.last_action = 0
        self.last_level_frac = 0.0
        self.total_drug_mg = 0.0
        return self._discretize()

    # ---------- observations ----------
    def bolus_window_open(self):
        return self.step_idx * self.dt < self.BOLUS_WINDOW_S

    def _discretize(self):
        e_idx = int(_clip((self.err + self.err_clip) / (2 * self.err_clip) * self.err_bins,
                          0, self.err_bins - 1))
        d_idx = int(_clip((self.dbis + self.dbis_clip) / (2 * self.dbis_clip) * self.dbis_bins,
                          0, self.dbis_bins - 1))
        if self.obs_mode == "bis":
            t = self.step_idx * self.dt
            t_idx = sum(1 for edge in self.TIME_BIN_EDGES_S[1:] if t >= edge)
            return ((e_idx * self.dbis_bins + d_idx) * self.n_actions
                    + int(self.last_action)) * len(self.TIME_BIN_EDGES_S) + t_idx
        c_idx = int(_clip(self.plant.Ce / self.ce_max * self.ce_bins, 0, self.ce_bins - 1))
        phase = 1 if self.bolus_window_open() else 0
        return (((e_idx * self.dbis_bins + d_idx) * self.ce_bins + c_idx) * self.n_actions
                + int(self.last_action)) * 2 + phase

    def continuous_state(self):
        """Normalized DQN feature vector (same role as dqn_agent._normalize_state):
        [BIS_error, dBIS/dt, Ce, last_action, time_since_start, target_BIS]
        (obs_mode="bis": the same without Ce). Time is scaled by a FIXED 3600 s,
        not episode_len, so shorter app episodes see the same feature scale as
        training."""
        err = _clip(self.err / self.err_clip, -3.0, 3.0)
        dbis = _clip(self.dbis / self.dbis_clip, -3.0, 3.0)
        act = self.last_action / (self.n_actions - 1)
        t = self.step_idx * self.dt / self.TIME_SCALE_S
        tgt = self.target / 100.0
        if self.obs_mode == "bis":
            return np.array([err, dbis, act, t, tgt], dtype=np.float32)
        ce = _clip(self.plant.Ce / self.CONC_SCALE, 0.0, 3.0)
        return np.array([err, dbis, ce, act, t, tgt], dtype=np.float32)

    def nearest_level(self, rate_mg_kg_h):
        return int(np.argmin(np.abs(self.RATE_LEVELS_MG_KG_H - rate_mg_kg_h)))

    # ---------- dynamics ----------
    def step(self, action_idx):
        """Discrete action (RL agents, and the PID's one-off induction bolus):
        0..7 = absolute infusion level, 8 = bolus (first 3 min only)."""
        a = int(action_idx)
        if a == self.BOLUS_ACTION:
            if self.bolus_window_open():
                bolus_mg = self.BOLUS_MG_PER_KG * self.patient["weight"]
                return self._advance(0.0, a, self.last_level_frac, bolus_mg=bolus_mg)
            a_rate = self.n_levels - 1          # window closed: acts as the top level
            return self._advance(float(self.RATE_LEVELS_MG_KG_H[a_rate]), a, float(a_rate))
        return self._advance(float(self.RATE_LEVELS_MG_KG_H[a]), a, float(a))

    def step_continuous(self, rate_mg_kg_h):
        """Continuous infusion rate in mg/kg/h (PID baseline), clipped to [0, 40].
        Same dynamics/reward as step(); the state's action feature uses the
        nearest discrete level, the smoothness penalty the fractional level
        (rate interpolated onto the level-index scale)."""
        rate = _clip(float(rate_mg_kg_h), 0.0, self.MAX_RATE_MG_KG_H)
        level_frac = float(np.interp(rate, self.RATE_LEVELS_MG_KG_H, np.arange(self.n_levels)))
        return self._advance(rate, self.nearest_level(rate), level_frac)

    def _advance(self, rate_mg_kg_h, action, level_frac, bolus_mg=0.0):
        u = rate_mg_kg_h * self.patient["weight"] / 3600.0 + bolus_mg / self.dt  # mg/s
        self.target = float(self.target_schedule[self.step_idx])
        shift = float(self.stim_shift[self.step_idx])

        cp, ce = self.plant.step(u)
        bis_true = self.bis_model.true_bis(ce, shift)
        bis = self.bis_model.measure(ce, shift)

        prev_filt = self.bis_filt
        self.bis_filt = prev_filt + self.dbis_filter_alpha * (bis - prev_filt)
        self.dbis = (self.bis_filt - prev_filt) / self.dt * 60.0  # BIS/min
        err = self.target - bis
        abs_e = abs(err)

        terminated = bis < self.terminate_bis
        reward = 1.0 if abs_e < self.target_band else -0.1 * (abs_e - self.target_band)
        if bis < self.safe_low or bis > self.safe_high:
            reward -= 2.0
        reward -= self.du_penalty * abs(level_frac - self.last_level_frac)
        if terminated:
            reward -= self.terminal_penalty

        self.total_drug_mg += u * self.dt
        self.history.append(dict(
            t=self.step_idx * self.dt, target=self.target, measured=bis, error=err,
            bis_true=bis_true, ce=ce, cp=cp, rate_mg_kg_h=u * 3600.0 / self.patient["weight"],
            rate_mg_s=u, bolus_mg=bolus_mg, command=rate_mg_kg_h, action=action,
            stim_shift=shift, ce50_eff=self.bis_model.ce50 * (1.0 + shift),
            reward=reward,
        ))

        self.bis, self.err = bis, err
        self.last_rate, self.last_action, self.last_level_frac = rate_mg_kg_h, action, level_frac
        self.step_idx += 1
        truncated = self.step_idx >= self.n_steps
        done = terminated or truncated
        return self._discretize(), reward, done, {"terminated": terminated,
                                                  "truncated": truncated and not terminated}
