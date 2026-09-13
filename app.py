"""
app.py
------
Streamlit app: live side-by-side comparison of PID, tabular Q-learning, and
DQN control of the simulated IV infusion line (src/environment.py).

Pick a target flow rate, optionally trigger an occlusion event mid-run, and
watch all three controllers respond on the same chart, stepped forward in
real time.

Run with:
    streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

from src.environment import IVInfusionPlant
from src.pid_controller import PIDController
from src.q_learning_agent import QLearningAgent
from src.dqn_agent import DQNAgent, KEffEstimator, _normalize_state

st.set_page_config(page_title="IV Infusion Control: PID vs Q-learning vs DQN", layout="wide")

DT = 1.0
EPISODE_LEN_DEFAULT = 180
ERR_BINS, DERR_BINS = 21, 11
ERR_CLIP, DERR_CLIP = 60.0, 15.0
N_ACTIONS = 7
ACTIONS = np.array([-8, -4, -1, 0, 1, 4, 8], dtype=float)


def discretize(err, derr):
    e_idx = int(np.clip((err + ERR_CLIP) / (2 * ERR_CLIP) * ERR_BINS, 0, ERR_BINS - 1))
    de_idx = int(np.clip((derr + DERR_CLIP) / (2 * DERR_CLIP) * DERR_BINS, 0, DERR_BINS - 1))
    return e_idx * DERR_BINS + de_idx


@st.cache_resource
def load_qlearning_agent(path="results/qlearning_qtable.pkl"):
    agent = QLearningAgent(ERR_BINS * DERR_BINS, N_ACTIONS)
    agent.load(path)
    return agent


@st.cache_resource
def load_dqn_agent(path="results/dqn_model.pt"):
    agent = DQNAgent(N_ACTIONS)
    agent.load(path)
    return agent


class SimController:
    """Wraps one controller (PID / Q-learning / DQN) driving its own plant
    instance, so all three run the identical disturbance schedule but with
    fully independent state (command history, integrator, Q-table lookups,
    k_eff estimate) -- exactly like three separate infusion pumps on the
    same line."""

    def __init__(self, kind, seed, start_flow, qlearning_agent=None, dqn_agent=None):
        self.kind = kind
        self.plant = IVInfusionPlant(dt=DT, seed=seed)
        self.plant.reset(start_flow=start_flow)
        self.prev_err = 0.0
        self.k_eff_estimator = KEffEstimator()

        if kind == "pid":
            self.pid = PIDController(dt=DT, u_max=self.plant.u_max)
            self.pid.reset(u0=start_flow)
        elif kind == "qlearning":
            self.agent = qlearning_agent
        elif kind == "dqn":
            self.agent = dqn_agent

    def step(self, target, k_eff, d_p):
        err = target - self.plant.Q
        derr = err - self.prev_err

        if self.kind == "pid":
            u = self.pid.compute(err)
            self.plant.set_command(u)
        elif self.kind == "qlearning":
            s = discretize(err, derr)
            a = self.agent.select_action(s, greedy=True)
            self.plant.set_command(self.plant.u + ACTIONS[a])
        elif self.kind == "dqn":
            k_hat = self.k_eff_estimator.estimate
            state = _normalize_state(err, derr, k_hat, ERR_CLIP, DERR_CLIP)
            a = self.agent.select_action(state, greedy=True)
            self.plant.set_command(self.plant.u + ACTIONS[a])

        self.plant.apply_occlusion(k_eff)
        self.plant.apply_pressure_disturbance(d_p)
        measured = self.plant.step()

        if self.kind == "dqn":
            self.k_eff_estimator.update(self.plant.u, measured)

        self.prev_err = target - measured
        return measured, self.plant.u


def init_session(target, seed, episode_len, ql_agent, dqn_agent):
    start_flow = target * 0.5
    controllers = {"PID": SimController("pid", seed, start_flow)}
    if ql_agent is not None:
        controllers["Q-learning"] = SimController("qlearning", seed, start_flow, qlearning_agent=ql_agent)
    if dqn_agent is not None:
        controllers["DQN"] = SimController("dqn", seed, start_flow, dqn_agent=dqn_agent)
    st.session_state.controllers = controllers
    st.session_state.t = 0
    st.session_state.n_steps = int(episode_len / DT)
    st.session_state.target = target
    st.session_state.k_eff = 1.0
    st.session_state.d_p = 0.0
    st.session_state.occlusion_until = -1
    st.session_state.occlusion_severity = 0.5
    st.session_state.rows = []
    st.session_state.running = False


def step_all():
    ss = st.session_state
    if ss.t >= ss.n_steps:
        ss.running = False
        return

    k_eff = ss.occlusion_severity if ss.t < ss.occlusion_until else 1.0
    ss.k_eff = k_eff
    row = {"t": ss.t, "target": ss.target, "k_eff": k_eff}
    for name, ctrl in ss.controllers.items():
        measured, command = ctrl.step(ss.target, k_eff, ss.d_p)
        row[f"{name} flow"] = measured
        row[f"{name} command"] = command
    ss.rows.append(row)
    ss.t += 1


st.title("IV Infusion Flow Control: PID vs Q-learning vs DQN")
st.caption(
    "Same simulated line (src/environment.py), same disturbance schedule, three independent "
    "controllers. Trigger an occlusion mid-run and watch each one recover."
)

with st.sidebar:
    st.header("Setup")
    target_flow = st.slider("Target flow rate (mL/hr)", 50, 200, 100, step=5)
    episode_len = st.slider("Episode length (s)", 60, 300, EPISODE_LEN_DEFAULT, step=10)
    seed = st.number_input("Random seed (sensor noise)", value=0, step=1)

    ql_available, dqn_available = True, True
    try:
        ql_agent = load_qlearning_agent()
    except FileNotFoundError:
        ql_available = False
        ql_agent = None
        st.warning("No trained Q-table found at results/qlearning_qtable.pkl -- run "
                   "`python -m src.train --episodes 5000 --seed 0` first.")
    try:
        dqn_agent = load_dqn_agent()
    except FileNotFoundError:
        dqn_available = False
        dqn_agent = None
        st.warning("No trained DQN model found at results/dqn_model.pt -- run "
                   "`python -m src.dqn_agent --episodes 2000 --seed 0` first.")

    if st.button("Reset episode", type="primary"):
        init_session(target_flow, int(seed), episode_len, ql_agent, dqn_agent)

    st.divider()
    st.subheader("Live disturbance")
    occ_severity = st.slider("Occlusion severity (k_eff during event)", 0.0, 1.0, 0.4, step=0.05)
    occ_duration = st.slider("Occlusion duration (s)", 5, 60, 20, step=5)
    if st.button("Trigger occlusion now"):
        if "controllers" in st.session_state:
            st.session_state.occlusion_severity = occ_severity
            st.session_state.occlusion_until = st.session_state.t + occ_duration

    pressure_bump = st.slider("Pressure disturbance d_p (mL/hr, applied now)", -25.0, 25.0, 0.0, step=1.0)
    if st.button("Apply pressure bump"):
        if "controllers" in st.session_state:
            st.session_state.d_p = pressure_bump

    st.divider()
    col_a, col_b = st.columns(2)
    play = col_a.button("Play / Step")
    autoplay = col_b.checkbox("Auto-run", value=False)

if "controllers" not in st.session_state:
    init_session(target_flow, int(seed), episode_len, ql_agent, dqn_agent)

if play:
    step_all()

if autoplay and st.session_state.t < st.session_state.n_steps:
    step_all()

chart_placeholder = st.empty()
command_placeholder = st.empty()
metrics_placeholder = st.empty()

if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows).set_index("t")
    active_names = list(st.session_state.controllers.keys())

    flow_cols = ["target"] + [f"{name} flow" for name in active_names]
    chart_placeholder.line_chart(df[flow_cols])

    command_cols = [f"{name} command" for name in active_names]
    command_placeholder.line_chart(df[command_cols])

    with metrics_placeholder.container():
        st.subheader("Running error stats (this episode so far)")
        cols = st.columns(len(active_names))
        for col, name in zip(cols, active_names):
            err = df["target"] - df[f"{name} flow"]
            col.metric(f"{name} mean |error| (mL/hr)", f"{err.abs().mean():.2f}")
            col.metric(f"{name} current flow (mL/hr)", f"{df[f'{name} flow'].iloc[-1]:.1f}")
else:
    st.info("Click **Reset episode** in the sidebar to start.")

st.caption(f"t = {st.session_state.get('t', 0)} / {st.session_state.get('n_steps', 0)} s   "
           f"current k_eff = {st.session_state.get('k_eff', 1.0):.2f}")

if autoplay and st.session_state.t < st.session_state.n_steps:
    time.sleep(0.05)
    st.rerun()
