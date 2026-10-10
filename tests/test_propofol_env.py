"""
test_propofol_env.py
---------------------

pytest suite for src/propofol_environment.py (run: pytest tests/ -v)
  PK   : Schnider parameters for the reference patient; Ce -> Cp at steady
         infusion (|Ce-Cp|/Cp < 1 % after 8 h at 6 mg/kg/h)
  PD   : 2 mg/kg bolus -> noise-free BIS strictly decreasing from the moment
         Ce starts rising until the effect-site peak (>= 60 s window), with a
         >= 15-point BIS drop
  API  : reset()/step() types and shapes, continuous_state() is 6-dim,
         ACTIONS are absolute mg/s levels + a bolus, 720-step episodes, done by
         step 720, early termination on BIS < 20, bolus only in the first 3 min
  PID  : BIS-only PID reaches the 40-60 band within 8 min for the nominal and
         a young (20 y, 54 kg, 161 cm) patient and stays > 70 % in target;
         the PID never reads plant state (Ce/Cp)
  obs  : obs_mode='bis' exposes no Ce (state/feature sizes, Ce-independence)
  Seeds: same constructor seed -> identical patient / PD / stimulus / target
         schedule on every reset regardless of the controller's actions or
         episode length (fair-comparison guarantee used by propofol_evaluate)
"""

import numpy as np
import pytest

from src.propofol_environment import (
    BISModel, NOMINAL_PATIENT, PROPOFOL_STATE_DIM, PropofolEnv, SchniderPlant, schnider_parameters,
)


def test_schnider_reference_patient_parameters():
    p = schnider_parameters(**NOMINAL_PATIENT)
    assert p["v1"] == pytest.approx(4.27)
    assert p["v2"] == pytest.approx(18.9)
    assert p["v3"] == pytest.approx(238.0)
    assert p["cl2"] == pytest.approx(1.29)
    assert p["cl3"] == pytest.approx(0.836)
    # LBM(77 kg, 177 cm, male) = 60.48 kg -> Cl1 = 1.89 - 0.0681*(60.48-59)
    assert p["lbm"] == pytest.approx(1.10 * 77 - 128 * (77 / 177) ** 2)
    assert p["cl1"] == pytest.approx(1.89 - 0.0681 * (p["lbm"] - 59.0))
    assert p["k10"] == pytest.approx(p["cl1"] / 4.27)


def test_rk4_matches_exact_linear_solution():
    """RK4 at dt=5 s vs the exact matrix-exponential solution of the same
    linear ODE (zero-order-hold infusion, 20 min on then 20 min off)."""
    from scipy.linalg import expm
    plant = SchniderPlant(dt=5.0, **NOMINAL_PATIENT)
    k10, k12, k13, k21, k31, ke0, v1 = (plant.k10, plant.k12, plant.k13, plant.k21,
                                        plant.k31, plant.ke0, plant.v1)
    A = np.array([[-(k10 + k12 + k13), k21, k31, 0.0],
                  [k12, -k21, 0.0, 0.0],
                  [k13, 0.0, -k31, 0.0],
                  [ke0 / v1, 0.0, 0.0, -ke0]])
    b = np.array([1.0, 0.0, 0.0, 0.0])
    # augmented system for exact ZOH step: d/dt [x; u] = [[A, b], [0, 0]] [x; u]
    M = np.zeros((5, 5))
    M[:4, :4], M[:4, 4] = A, b
    Phi = expm(M * 5.0)
    x = np.zeros(4)
    u = 10.0 * 77.0 / 3600.0
    for k in range(480):
        uk = u if k < 240 else 0.0
        x = (Phi @ np.append(x, uk))[:4]
        plant.step(uk)
        got = np.array([plant.A1, plant.A2, plant.A3, plant.Ce])
        # RK4 local truncation error at h*k ~ 0.08 is ~1e-5 relative
        np.testing.assert_allclose(got, x, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("patient", [
    NOMINAL_PATIENT,
    dict(age=75, weight=55, height=160, sex="female"),
    dict(age=25, weight=105, height=190, sex="male"),
])
def test_ce_converges_to_cp_under_constant_infusion(patient):
    plant = SchniderPlant(dt=5.0, **patient)
    u = 6.0 * patient["weight"] / 3600.0  # 6 mg/kg/h in mg/s
    for _ in range(int(8 * 3600 / 5)):    # 8 hours
        cp, ce = plant.step(u)
    assert cp > 1.0
    assert abs(ce - cp) / cp < 0.01
    assert min(plant.A1, plant.A2, plant.A3, plant.Ce) >= 0.0


def test_bolus_drives_bis_down_monotonically_after_ke0_lag():
    plant = SchniderPlant(dt=5.0, **NOMINAL_PATIENT)
    bis_model = BISModel(noise_std=0.0)
    bolus_mg = 2.0 * NOMINAL_PATIENT["weight"]              # 2 mg/kg
    ce_trace, bis_trace = [plant.Ce], [bis_model.true_bis(plant.Ce)]
    for k in range(120):                                     # 10 min
        _, ce = plant.step(bolus_mg / plant.dt if k == 0 else 0.0)
        ce_trace.append(ce)
        bis_trace.append(bis_model.measure(ce))
    ce_trace, bis_trace = np.array(ce_trace), np.array(bis_trace)

    start = int(np.argmax(ce_trace > 1e-6))                 # Ce starts rising
    peak = int(np.argmax(ce_trace))                          # effect-site peak
    assert (peak - start) * plant.dt >= 60.0                 # sustained window, not a blip
    window = bis_trace[start:peak + 1]
    assert np.all(np.diff(window) < 0), "BIS must fall monotonically while Ce rises"
    assert bis_trace[0] - bis_trace[peak] > 15.0
    assert bis_trace[-1] > bis_trace[peak]                   # and recovers after the peak


def test_reset_and_step_api_shapes():
    env = PropofolEnv(seed=3)
    s = env.reset()
    assert isinstance(s, int) and 0 <= s < env.n_states
    assert env.n_states == 15 * 3 * 12 * 9 * 2 and env.n_actions == 9
    assert env.ACTIONS.shape == (9,)
    assert env.ACTIONS[0] == 0.0
    assert env.ACTIONS[7] == pytest.approx(40.0 * env.patient["weight"] / 3600.0)
    assert env.ACTIONS[8] == pytest.approx(1.0 * env.patient["weight"] / env.dt)   # bolus, mg/s for 5 s

    obs = env.continuous_state()
    assert obs.shape == (PROPOFOL_STATE_DIM,) and obs.dtype == np.float32

    out = env.step(4)
    assert len(out) == 4
    s2, r, done, info = out
    assert isinstance(s2, int) and 0 <= s2 < env.n_states
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert set(info) >= {"terminated", "truncated"}
    h = env.history[-1]
    for key in ("t", "target", "measured", "error", "ce", "cp", "rate_mg_kg_h", "rate_mg_s"):
        assert key in h
    assert h["error"] == pytest.approx(h["target"] - h["measured"])


def test_episode_is_720_steps():
    env = PropofolEnv(seed=0)
    env.reset()
    n, done = 0, False
    while not done:
        _, _, done, info = env.step(0)   # no drug -> BIS never < 20 -> full episode
        n += 1
    assert n == 720 == env.n_steps
    assert info["truncated"] and not info["terminated"]
    assert len(env.history) == 720


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_done_fires_at_or_before_720_under_any_policy(seed):
    env = PropofolEnv(seed=seed)
    rng = np.random.default_rng(seed)
    for policy in (lambda: 7, lambda: int(rng.integers(9))):
        env.reset()
        n, done = 0, False
        while not done and n < 1000:
            _, _, done, _ = env.step(policy())
            n += 1
        assert done and n <= 720


def test_early_termination_on_deep_anesthesia():
    env = PropofolEnv(seed=0, patient=NOMINAL_PATIENT, stimuli=False)
    env.reset()
    env.bis_model.ce50 = 1.0   # extremely sensitive patient
    n, done, r = 0, False, 0.0
    while not done:
        _, r, done, info = env.step(7)
        n += 1
    assert info["terminated"] and n < 720
    assert env.history[-1]["measured"] < 20.0
    assert r <= -env.terminal_penalty


def _episode_signature(env):
    return (dict(env.patient), env.bis_model.ce50, env.bis_model.gamma,
            env.stim_shift.copy(), env.target_schedule.copy(), env.bis)


def _assert_same(sig_a, sig_b):
    assert sig_a[0] == sig_b[0]
    assert sig_a[1] == sig_b[1] and sig_a[2] == sig_b[2]
    np.testing.assert_array_equal(sig_a[3], sig_b[3])
    np.testing.assert_array_equal(sig_a[4], sig_b[4])
    assert sig_a[5] == sig_b[5]


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_same_seed_same_patient_and_schedule_regardless_of_controller(mode):
    env_a, env_b = PropofolEnv(seed=20000, mode=mode), PropofolEnv(seed=20000, mode=mode)
    rng = np.random.default_rng(99)
    for episode in range(3):
        env_a.reset()
        env_b.reset()
        _assert_same(_episode_signature(env_a), _episode_signature(env_b))
        # drive them differently, and for different lengths, before the next reset
        done = False
        while not done:
            _, _, done, _ = env_a.step(7)
        for _ in range(10 + 50 * episode):
            env_b.step(int(rng.integers(9)))
    assert env_a.stim_shift.max() >= 0.15        # stimuli actually present


def test_noise_sequence_shared_across_controllers():
    """Same seed -> same per-step BIS noise draw, so two controllers that
    produce the same Ce see exactly the same measurement."""
    env_a, env_b = PropofolEnv(seed=5), PropofolEnv(seed=5)
    env_a.reset()
    env_b.reset()
    for _ in range(20):
        env_a.step(0)
        env_b.step_continuous(0.0)
    assert [h["measured"] for h in env_a.history] == [h["measured"] for h in env_b.history]


def test_eval_stimuli_are_harsher_but_within_spec():
    for mode, (lo, n_min) in {"train": (0.15, 2), "eval": (0.20, 3)}.items():
        for seed in range(20):
            env = PropofolEnv(seed=seed, mode=mode)
            env.reset()
            active = env.stim_shift[env.stim_shift > 0]
            assert active.min() >= lo - 1e-12 and active.max() <= 0.30 + 1e-12
            assert (env.stim_shift > 0).sum() * env.dt >= n_min * 120.0 * 0.5  # some events present


def test_target_schedule_profile():
    env = PropofolEnv(seed=0, extra_target_steps=False)
    env.reset()
    ts = env.target_schedule
    assert ts[0] == 60.0 and ts[int(600 / 5) - 1] == 60.0
    assert ts[int(900 / 5)] == 50.0 and ts[-1] == 50.0
    for seed in range(30):
        env = PropofolEnv(seed=seed)
        env.reset()
        assert env.target_schedule.min() >= 40.0 and env.target_schedule.max() <= 60.0


def test_controllers_run_end_to_end():
    from src.dqn_agent import DQNAgent
    from src.propofol_metrics import compute_propofol_metrics
    from src.propofol_pid_tune import run_pid_episode
    from src.propofol_train import run_dqn_episode, run_qlearning_episode
    from src.q_learning_agent import QLearningAgent

    env = PropofolEnv(seed=20000, mode="eval")
    hist = run_pid_episode(env)
    m = compute_propofol_metrics(hist, env.target_change_steps, env.n_steps, env.dt)
    assert 0.0 <= m["time_in_target_pct"] <= 100.0 and m["total_propofol_mg"] > 0.0

    hist, _ = run_qlearning_episode(env, QLearningAgent(env.n_states, env.n_actions, seed=0), greedy=True)
    assert 0 < len(hist) <= 720
    hist, _ = run_dqn_episode(env, DQNAgent(env.n_actions, state_dim=PROPOFOL_STATE_DIM, seed=0), greedy=True)
    assert 0 < len(hist) <= 720


def test_metrics_pad_early_termination():
    from src.propofol_metrics import compute_propofol_metrics
    env = PropofolEnv(seed=0, patient=NOMINAL_PATIENT, stimuli=False)
    env.reset()
    env.bis_model.ce50 = 1.0
    done = False
    while not done:
        _, _, done, _ = env.step(7)
    m = compute_propofol_metrics(env.history, env.target_change_steps, env.n_steps, env.dt)
    assert m["terminated_early"] == 1.0
    assert m["pct_bis_below_40"] > 50.0   # scored as staying overdosed for the rest of the hour


def test_bolus_only_during_induction_window():
    env = PropofolEnv(seed=0, patient=NOMINAL_PATIENT, stimuli=False)
    env.reset()
    env.step(env.BOLUS_ACTION)
    h = env.history[-1]
    assert h["bolus_mg"] == pytest.approx(77.0)              # 1 mg/kg
    assert h["rate_mg_s"] * env.dt == pytest.approx(77.0)
    for _ in range(int(env.BOLUS_WINDOW_S / env.dt)):        # run past the 3-min window
        env.step(0)
    env.step(env.BOLUS_ACTION)
    h = env.history[-1]
    assert h["bolus_mg"] == 0.0
    assert h["command"] == env.RATE_LEVELS_MG_KG_H[-1]       # acts as the top infusion level


def test_nominal_inverse_bis_roundtrip():
    bm = BISModel(noise_std=0.0)
    for bis in (40.0, 50.0, 60.0):
        assert bm.true_bis(BISModel.nominal_ce_for_bis(bis)) == pytest.approx(bis)


def _flat_target_episode(patient, minutes=30, target=50.0, seed=0):
    env = PropofolEnv(seed=seed, mode="eval", patient=patient, stimuli=False,
                      extra_target_steps=False, episode_len=minutes * 60.0)
    env.reset()
    env.target_schedule[:] = target
    env.target = target
    return env


@pytest.mark.parametrize("patient", [NOMINAL_PATIENT, dict(age=20, weight=54, height=161, sex="female")])
def test_bis_only_pid_reaches_band_within_8_min(patient):
    from src.propofol_metrics import compute_propofol_metrics
    from src.propofol_pid_tune import TUNED_PID_GAINS, BISPID
    env = _flat_target_episode(patient)
    pid = BISPID(env, **TUNED_PID_GAINS)
    done = False
    while not done:
        _, _, done, _ = pid.step()
    bis = np.array([h["bis_true"] for h in env.history])
    first_in_band = int(np.argmax((bis >= 40) & (bis <= 60)))
    assert first_in_band * env.dt <= 8 * 60
    m = compute_propofol_metrics(env.history, [], env.n_steps, env.dt)
    assert m["time_in_target_pct"] > 70.0 and m["terminated_early"] == 0.0


class _NoPlantStateEnv(PropofolEnv):
    """PropofolEnv whose plant raises if anything outside the env reads Ce/Cp."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._guard = False

    @property
    def plant(self):
        if self._guard:
            raise AssertionError("controller read env.plant (Ce/Cp)")
        return self._plant

    @plant.setter
    def plant(self, value):
        self._plant = value


def test_bis_only_pid_never_reads_plant_state():
    from src.propofol_pid_tune import TUNED_PID_GAINS, BISPID
    env = _NoPlantStateEnv(seed=0, mode="eval")
    env.reset()
    pid = BISPID(env, **TUNED_PID_GAINS)
    real_advance = env._advance

    def guarded_advance(*a, **kw):        # the env itself may use its plant
        env._guard = False
        try:
            return real_advance(*a, **kw)
        finally:
            env._guard = True

    env._advance = guarded_advance
    env._guard = True
    done = False
    while not done:
        _, _, done, _ = pid.step()       # AssertionError if the PID touches env.plant


def test_bis_obs_mode_hides_ce():
    from src.propofol_environment import PROPOFOL_BIS_STATE_DIM
    a = PropofolEnv(seed=1, obs_mode="bis")
    b = PropofolEnv(seed=1, obs_mode="bis")
    a.reset()
    b.reset()
    assert a.n_states == 15 * 5 * 9 * 4 and a.state_dim == PROPOFOL_BIS_STATE_DIM == 5
    for _ in range(30):
        a.step(5)
        b.step(5)
    b.plant.Ce += 3.0                       # different hidden drug state ...
    assert a._discretize() == b._discretize()                       # ... same observation
    np.testing.assert_array_equal(a.continuous_state(), b.continuous_state())
