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

import json
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.environment import IVInfusionPlant
from src.pid_controller import PIDController
from src.q_learning_agent import QLearningAgent
from src.dqn_agent import DQNAgent, KEffEstimator, _normalize_state

st.set_page_config(page_title="IV Infusion Control: PID vs Q-learning vs DQN", layout="wide")

DT = 1.0
EPISODE_LEN_DEFAULT = 60
DELAY_STEPS = 10
ERR_BINS, DERR_BINS, PENDING_BINS = 21, 11, 7
ERR_CLIP, DERR_CLIP, PENDING_CLIP = 60.0, 15.0, 40.0
N_ACTIONS = 7
ACTIONS = np.array([-8, -4, -1, 0, 1, 4, 8], dtype=float)
SAFE_BAND = 5.0  # mL/hr, matches IVInfusionEnv.safe_band

# Fixed axis ranges so the chart doesn't rescale (and visually jump) on every
# Play/Step rerun as new data points arrive.
FLOW_Y_DOMAIN = (0, 200)
COMMAND_Y_DOMAIN = (0, 400)


def fixed_range_line_chart(df, cols, y_domain, y_title):
    """Multi-line chart with a pinned y-axis domain, built with Altair (st.line_chart
    has no axis-range control and autoranges, which is what causes the jump/flicker
    as new rows are appended)."""
    long_df = df[cols].reset_index().melt("t", var_name="series", value_name="value")
    chart = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("t", title="Time (s)"),
            y=alt.Y("value", title=y_title, scale=alt.Scale(domain=list(y_domain))),
            color=alt.Color("series", title=None),
        )
    )
    return chart


# Distinct dashed-line color + label per triggered disturbance event type,
# drawn on the flow chart at the exact t it was triggered.
EVENT_STYLE = {
    "occlusion": {"color": "orange", "label": "Blockage"},
    "pressure": {"color": "purple", "label": "Bag/arm bump"},
}


def add_event_markers(line_chart, events):
    """Layers a vertical dashed rule + text label onto `line_chart` for each
    triggered event (occlusion/pressure), at its exact trigger time `t`.
    Purely visual annotation -- does not touch the underlying data/series."""
    if not events:
        return line_chart

    ev_df = pd.DataFrame(events)
    ev_df["color"] = ev_df["type"].map(lambda k: EVENT_STYLE[k]["color"])
    ev_df["label"] = ev_df["type"].map(lambda k: EVENT_STYLE[k]["label"])

    rules = (
        alt.Chart(ev_df)
        .mark_rule(strokeDash=[5, 4], size=2)
        .encode(
            x=alt.X("t:Q"),
            color=alt.Color("color:N", scale=None, legend=None),
        )
    )
    labels = (
        alt.Chart(ev_df)
        .mark_text(align="left", baseline="top", dx=4, dy=2, fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("t:Q"),
            y=alt.value(4),
            text="label:N",
            color=alt.Color("color:N", scale=None, legend=None),
        )
    )
    return alt.layer(line_chart, rules, labels).resolve_scale(color="independent")


SPLASH_SECONDS = 3.0

# Self-contained splash: inline SVG + CSS keyframes, no external assets.
SPLASH_HTML = """
<style>
.iv-splash {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 70vh;
  text-align: center;
}
.iv-splash-title { font-size: 2.3rem; font-weight: 700; letter-spacing: -0.01em; }
.iv-splash-sub { font-size: 1.1rem; opacity: 0.75; margin-top: 0.25rem; }
.iv-splash-caption { font-size: 0.9rem; opacity: 0.6; margin-top: 0.75rem; }
.iv-splash svg { margin-top: 1.5rem; }
@keyframes iv-drip {
  0%   { transform: translateY(0px);  opacity: 0; }
  12%  { opacity: 1; }
  80%  { opacity: 1; }
  100% { transform: translateY(42px); opacity: 0; }
}
.iv-drop { animation: iv-drip 1s linear infinite; }
</style>
<div class="iv-splash">
  <div class="iv-splash-title">IV Infusion Flow Control</div>
  <div class="iv-splash-sub">PID vs Q-learning vs DQN</div>
  <svg width="170" height="240" viewBox="0 0 170 240" role="img" aria-label="Animated IV drip">
    <line x1="85" y1="2" x2="85" y2="14" stroke="#5599c7" stroke-width="2"/>
    <rect x="53" y="14" width="64" height="86" rx="10" ry="10"
          fill="#5599c7" fill-opacity="0.18" stroke="#5599c7" stroke-width="2"/>
    <path d="M53 46 h64 v44 a10 10 0 0 1 -10 10 h-44 a10 10 0 0 1 -10 -10 z"
          fill="#5599c7" fill-opacity="0.38"/>
    <path d="M79 100 L91 100 L85 112 Z" fill="#5599c7"/>
    <ellipse cx="85" cy="140" rx="16" ry="26" fill="none" stroke="#5599c7" stroke-width="2"/>
    <line x1="85" y1="166" x2="85" y2="232" stroke="#5599c7" stroke-width="2"/>
    <circle class="iv-drop" cx="85" cy="116" r="4.5" fill="#5599c7"/>
  </svg>
  <div class="iv-splash-caption">Loading simulation...</div>
</div>
"""


def show_splash_once():
    """Full-screen splash, shown only on the very first load of the session.
    Subsequent reruns (widget changes, Play/Step, auto-run ticks) skip it."""
    if st.session_state.get("splash_shown"):
        return
    placeholder = st.empty()
    placeholder.markdown(SPLASH_HTML, unsafe_allow_html=True)
    time.sleep(SPLASH_SECONDS)
    placeholder.empty()
    st.session_state.splash_shown = True


def best_by(values, lower_is_better=True):
    """(label, value) of the winning controller(s). Ties are reported as ties
    rather than silently resolved by dict order, which would always favour
    whichever controller happens to be inserted first."""
    target = min(values.values()) if lower_is_better else max(values.values())
    winners = [name for name, v in values.items() if v == target]
    label = winners[0] if len(winners) == 1 else ", ".join(winners) + " (tied)"
    return label, target


def load_batch_summary(path="results/eval_summary.json"):
    """Batch metrics written by src/evaluate.py, or None if it hasn't been run."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def batch_summary_table(batch):
    rows = {}
    for controller, metrics in batch["controllers"].items():
        rows[controller] = {k: v["mean"] for k, v in metrics.items()}
    return pd.DataFrame(rows).round(2)


# Plain-language row labels for the performance metrics table, display only --
# the underlying metric keys/values from src/evaluate.py are unchanged. Only
# these four metrics are shown in the table (see PERFORMANCE_METRIC_ROWS).
METRIC_DISPLAY_LABELS = {
    "IAE": "Total Tracking Error (IAE)",
    "ISE": "Total Error, Big Misses Count More (ISE)",
    "ITAE": "Time-Weighted Tracking Error (ITAE)",
    "ITSE": "Time-Weighted Big-Miss Error (ITSE)",
}

PERFORMANCE_METRIC_ROWS = ["IAE", "ISE", "ITAE", "ITSE"]


def discretize(err, derr, pending):
    e_idx = int(np.clip((err + ERR_CLIP) / (2 * ERR_CLIP) * ERR_BINS, 0, ERR_BINS - 1))
    de_idx = int(np.clip((derr + DERR_CLIP) / (2 * DERR_CLIP) * DERR_BINS, 0, DERR_BINS - 1))
    p_idx = int(np.clip((pending + PENDING_CLIP) / (2 * PENDING_CLIP) * PENDING_BINS, 0, PENDING_BINS - 1))
    return (e_idx * DERR_BINS + de_idx) * PENDING_BINS + p_idx


@st.cache_resource
def load_qlearning_agent(path="results/qlearning_qtable.pkl"):
    agent = QLearningAgent(ERR_BINS * DERR_BINS * PENDING_BINS, N_ACTIONS)
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
        self.plant = IVInfusionPlant(dt=DT, delay_steps=DELAY_STEPS, seed=seed)
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
        pending = self.plant.pending_correction()

        if self.kind == "pid":
            u = self.pid.compute(err)
            self.plant.set_command(u)
        elif self.kind == "qlearning":
            s = discretize(err, derr, pending)
            a = self.agent.select_action(s, greedy=True)
            self.plant.set_command(self.plant.u + ACTIONS[a])
        elif self.kind == "dqn":
            k_hat = self.k_eff_estimator.estimate
            state = _normalize_state(err, derr, pending, k_hat, ERR_CLIP, DERR_CLIP, PENDING_CLIP)
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
    controllers = {"PID (classic controller)": SimController("pid", seed, start_flow)}
    if ql_agent is not None:
        controllers["Q-learning (RL)"] = SimController("qlearning", seed, start_flow, qlearning_agent=ql_agent)
    if dqn_agent is not None:
        controllers["DQN (RL)"] = SimController("dqn", seed, start_flow, dqn_agent=dqn_agent)
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
    st.session_state.events = []


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


show_splash_once()

st.title("IV Infusion Flow Control: PID vs Q-learning vs DQN")
st.caption(
    "Same simulated line (src/environment.py), same disturbance schedule, three independent "
    "controllers. Trigger an occlusion mid-run and watch each one recover."
)

if "auto_running" not in st.session_state:
    st.session_state.auto_running = False

st.markdown(
    """
    <style>
    div.st-key-start_stop_controls button[kind="primary"] {
        background-color: #d32f2f;
        border-color: #d32f2f;
        color: white;
    }
    div.st-key-start_stop_controls button[kind="primary"]:hover {
        background-color: #b71c1c;
        border-color: #b71c1c;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="start_stop_controls"):
    col_start, col_stop, _col_spacer = st.columns([1, 1, 4])
    if col_start.button("START", type="primary"):
        st.session_state.auto_running = True
    if col_stop.button("STOP", type="primary"):
        st.session_state.auto_running = False

with st.sidebar:
    st.header("Setup")
    target_flow = st.slider("Target flow rate (mL/hr)", 50, 200, 100, step=5)
    episode_len = st.slider(
        "Episode length", 50, 200, EPISODE_LEN_DEFAULT, step=10,
        help="Episode length (s)",
    )
    seed = st.number_input(
        "Repeat-test number", value=0, step=1,
        help="Same number = identical random sensor noise, so you can compare "
             "controllers fairly on the exact same run. Change it for a fresh "
             "random run. (Random seed / sensor noise)",
    )

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
    occ_severity = st.slider(
        "Line blockage severity", 0.0, 1.0, 0.5, step=0.05,
        help="How blocked the IV line is during the event. 1.0 = fully open, "
             "lower = more blocked (e.g. 0.4 = only 40% of the flow gets through). "
             "(Occlusion severity / k_eff during event)",
    )
    occ_duration = st.slider(
        "Blockage length (seconds)", 5, 60, 20, step=5,
        help="How many seconds the line stays blocked before clearing. "
             "(Occlusion duration)",
    )
    if st.button("Simulate a blockage in the flow", help="Trigger occlusion now"):
        if "controllers" in st.session_state:
            st.session_state.occlusion_severity = occ_severity
            st.session_state.occlusion_until = st.session_state.t + occ_duration
            st.session_state.events.append({"t": st.session_state.t, "type": "occlusion"})

    bag_height_cm = st.slider(
        "Bag height", 1, 100, 50, step=1,
        help="Height of the IV bag above the patient's arm, in cm. 50 cm is "
             "the neutral baseline (no extra pressure). Raising or lowering "
             "it pushes flow up or down, separate from any blockage. "
             "(Converted to pressure disturbance d_p, mL/hr, applied now)",
    )
    # UI-only unit-conversion layer: maps the physically-flavored "bag height"
    # input (cm) onto the existing pressure-disturbance quantity d_p (mL/hr)
    # that the simulation actually consumes. 50cm is the neutral baseline
    # (d_p = 0); linear from there to each end of the slider range so
    # 1cm -> -25 mL/hr and 100cm -> +25 mL/hr, matching the old direct
    # -25..25 mL/hr slider's range. The simulation/controller logic itself
    # is untouched -- only how d_p is derived from the sidebar input changes.
    BAG_HEIGHT_BASELINE_CM = 50
    BAG_HEIGHT_MIN_CM, BAG_HEIGHT_MAX_CM = 1, 100
    PRESSURE_BUMP_RANGE = 25.0  # mL/hr, same magnitude as the old slider
    if bag_height_cm >= BAG_HEIGHT_BASELINE_CM:
        pressure_bump = ((bag_height_cm - BAG_HEIGHT_BASELINE_CM)
                          / (BAG_HEIGHT_MAX_CM - BAG_HEIGHT_BASELINE_CM) * PRESSURE_BUMP_RANGE)
    else:
        pressure_bump = ((bag_height_cm - BAG_HEIGHT_BASELINE_CM)
                          / (BAG_HEIGHT_BASELINE_CM - BAG_HEIGHT_MIN_CM) * PRESSURE_BUMP_RANGE)

    if st.button("Simulate a bag or arm movement", help="Apply pressure bump"):
        if "controllers" in st.session_state:
            st.session_state.d_p = pressure_bump
            st.session_state.events.append({"t": st.session_state.t, "type": "pressure"})

if "controllers" not in st.session_state:
    init_session(target_flow, int(seed), episode_len, ql_agent, dqn_agent)

if st.session_state.auto_running and st.session_state.t < st.session_state.n_steps:
    step_all()

if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows).set_index("t")
    active_names = list(st.session_state.controllers.keys())

    flow_cols = ["target"] + [f"{name} flow" for name in active_names]
    flow_chart = fixed_range_line_chart(df, flow_cols, FLOW_Y_DOMAIN, "Flow (mL/hr)")
    flow_chart = add_event_markers(flow_chart, st.session_state.get("events", []))
    st.altair_chart(flow_chart, width="stretch", key="flow_chart")

    st.subheader("Running error stats (this episode so far)")
    abs_err = {name: (df["target"] - df[f"{name} flow"]).abs() for name in active_names}
    cols = st.columns(len(active_names))
    for col, name in zip(cols, active_names):
        col.metric(f"{name}: average tracking error (mL/hr)", f"{abs_err[name].mean():.2f}")
        col.metric(f"{name}: flow right now (mL/hr)", f"{df[f'{name} flow'].iloc[-1]:.1f}")

    mean_errors = {name: float(e.mean()) for name, e in abs_err.items()}
    max_errors = {name: float(e.max()) for name, e in abs_err.items()}
    in_band = {name: float((e <= SAFE_BAND).mean() * 100.0) for name, e in abs_err.items()}

    best_mean, best_mean_val = best_by(mean_errors)
    best_max, best_max_val = best_by(max_errors)
    best_band, best_band_val = best_by(in_band, lower_is_better=False)

    st.markdown(f"- **Most accurate overall:** `{best_mean}` at {best_mean_val:.2f} mL/hr")
    st.markdown(f"- **Best at avoiding big spikes:** `{best_max}` at {best_max_val:.2f} mL/hr")
    st.markdown(f"- **Most time staying on-target (+-{SAFE_BAND:.0f} mL/hr):** `{best_band}` at {best_band_val:.1f}%")

    st.caption(
        "Different metrics favour different controllers - mean error rewards steady "
        "tracking, worst-case error rewards avoiding large excursions."
    )

    batch = load_batch_summary()
    st.subheader("Performance Metrics")
    if batch is None:
        st.info(
            "No batch summary found at `results/eval_summary.json` -- run "
            "`python -m src.evaluate --n_eval 30 --dqn_model results/dqn_model.pt` to generate it."
        )
    else:
        st.caption(
            f"Averaged over {batch['n_eval']} test runs with different random conditions "
            "- this is the reliable comparison, since a single run can be noisy. Lower is "
            "better for every row except time on-target."
        )
        table = batch_summary_table(batch)
        table = table.reindex([r for r in PERFORMANCE_METRIC_ROWS if r in table.index])
        table = table.rename(index=METRIC_DISPLAY_LABELS)
        st.dataframe(table, width="stretch")
else:
    st.info("Click **Reset episode** in the sidebar to start.")

st.caption(f"t = {st.session_state.get('t', 0)} / {st.session_state.get('n_steps', 0)} s   "
           f"current k_eff = {st.session_state.get('k_eff', 1.0):.2f}")

if st.session_state.auto_running and st.session_state.t < st.session_state.n_steps:
    time.sleep(0.05)
    st.rerun()
