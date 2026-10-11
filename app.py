"""
app.py
------

Dashboard: steps PID, Q-learning and DQN forward live on one chart.
Q-table (results/qlearning_qtable.pkl) and DQN weights (results/dqn_model.pt)
are frozen: loaded from results/, never trained or updated here.
PID, Q and DQN all run on the same simulated line (IVInfusionPlant, delay_steps=10).
Batch metrics shown come from results/eval_summary.json.
Sidebar "Simulator" selectbox: "IV Flow Control" (default, the dashboard
below, unchanged) or "Propofol (Schnider)" (render_propofol_dashboard():
PropofolSimController x3 on PropofolEnv, BIS / Ce / infusion-rate charts,
patient sliders, live "Simulate surgical stimulus"; frozen models
results/propofol_v2_qlearning_qtable.pkl and results/propofol_v2_dqn_model.pt;
PID = CascadePID with TUNED_PID_GAINS; live this-episode metrics table via
src/propofol_metrics.py; no batch table in this mode -- batch numbers live in
results/propofol_eval_summary.json from src/propofol_evaluate.py).
Streamlit app: live side-by-side comparison of PID, tabular Q-learning, and
DQN control of the simulated IV infusion line (src/environment.py).

Pick a target flow rate, optionally trigger an occlusion event mid-run, and
watch all three controllers respond on the same chart, stepped forward in
real time.

Run with:
    streamlit run app.py
"""

import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.environment import IVInfusionPlant, iv_reward
from src.pid_controller import PIDController
from src.q_learning_agent import QLearningAgent
from src.dqn_agent import DQNAgent, KEffEstimator, _normalize_state
from src.propofol_environment import PropofolEnv
from src.propofol_metrics import compute_propofol_metrics
from src.propofol_pid_tune import TUNED_PID_GAINS, CascadePID
from src.propofol_train import DECISION_EVERY, MODEL_PATHS, make_dqn_agent, model_prefix

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

# Fixed per-controller line colors, legible on the light theme's white chart
# background -- matched by substring so they apply regardless of the " flow"/
# " BIS"/etc. suffix a given chart's series names carry.
CONTROLLER_COLORS = [
    ("PID", "#1565C0"),
    ("Q-learning", "#EF6C00"),
    ("DQN", "#2E7D32"),
    ("target", "#D32F2F"),
]
DEFAULT_SERIES_COLOR = "#616161"


def series_color_scale(series_names):
    """alt.Scale mapping each series name to a fixed color via substring match
    against CONTROLLER_COLORS, so PID/Q-learning/DQN/target keep the same
    color across every chart in both simulator modes."""
    def color_for(name):
        for key, color in CONTROLLER_COLORS:
            if key.lower() in name.lower():
                return color
        return DEFAULT_SERIES_COLOR

    names = list(series_names)
    return alt.Scale(domain=names, range=[color_for(n) for n in names])


AXIS_COLORS = dict(labelColor="#1F2430", titleColor="#1F2430", gridColor="#E0E0E0",
                    domainColor="#BDBDBD", tickColor="#BDBDBD")


def _apply_light_config(chart):
    """Explicit light axis/legend/view styling, independent of whatever Vega
    theme Streamlit's own light config.toml theme applies -- keeps charts
    readable even if that theming isn't picked up."""
    return (
        chart.configure_axis(**AXIS_COLORS)
        .configure_legend(labelColor="#1F2430", titleColor="#1F2430")
        .configure_view(strokeWidth=0)
    )


REWARD_SMOOTH_WINDOWS = [("Q-learning", 200), ("DQN", 50)]
DEFAULT_REWARD_SMOOTH_WINDOW = 50


def _reward_smooth_window(name):
    for key, window in REWARD_SMOOTH_WINDOWS:
        if key.lower() in name.lower():
            return window
    return DEFAULT_REWARD_SMOOTH_WINDOW


def load_reward_curve(path):
    """Per-episode training rewards saved by a training script (np.save), or
    None if that training run hasn't been done / saved yet."""
    try:
        return np.load(path)
    except FileNotFoundError:
        return None


def training_reward_chart(curves):
    """Raw (very faint) + smoothed (thick moving average; window from
    REWARD_SMOOTH_WINDOWS -- 200 episodes for Q-learning, 50 for DQN)
    training-reward-per-episode lines, one pair per curve in `curves`
    ({label: np.ndarray | None}). Returns None if every curve is missing."""
    raw_frames, smooth_frames = [], []
    for name, arr in curves.items():
        if arr is None or len(arr) == 0:
            continue
        episodes = np.arange(len(arr))
        raw_frames.append(pd.DataFrame({"episode": episodes, "series": name, "value": arr}))
        window = min(_reward_smooth_window(name), len(arr))
        smoothed = pd.Series(arr).rolling(window, min_periods=1).mean().to_numpy()
        smooth_frames.append(pd.DataFrame({"episode": episodes, "series": name, "value": smoothed}))
    if not raw_frames:
        return None

    names = sorted(curves.keys())
    scale = series_color_scale(names)
    raw_line = (
        alt.Chart(pd.concat(raw_frames, ignore_index=True))
        .mark_line(opacity=0.15, strokeWidth=1)
        .encode(
            x=alt.X("episode", title="Training episode"),
            y=alt.Y("value", title="Episode reward"),
            color=alt.Color("series", title=None, scale=scale),
        )
    )
    smooth_line = (
        alt.Chart(pd.concat(smooth_frames, ignore_index=True))
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("episode", title="Training episode"),
            y=alt.Y("value", title="Episode reward"),
            color=alt.Color("series", title=None, scale=scale),
        )
    )
    return alt.layer(raw_line, smooth_line)


def reward_chart(df, cols, x_title, y_title, rename_suffix=""):
    """Per-step line chart for reward-shaped columns (no fixed y-domain --
    unlike the plant-variable charts, reward includes rare large penalties
    e.g. the propofol early-termination penalty, which should be visible)."""
    long_df = df[cols].reset_index().melt("t", var_name="series", value_name="value")
    if rename_suffix:
        long_df["series"] = long_df["series"].str.replace(rename_suffix, "", regex=False)
    names = sorted(long_df["series"].unique().tolist())
    return (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("t", title=x_title),
            y=alt.Y("value", title=y_title),
            color=alt.Color("series", title=None, scale=series_color_scale(names)),
        )
    )


def cumulative_reward_df(df, names, reward_suffix=" reward"):
    data = {n: df[f"{n}{reward_suffix}"].cumsum() for n in names}
    return pd.DataFrame(data, index=df.index)


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
            color=alt.Color("series", title=None, scale=series_color_scale(cols)),
        )
    )
    return chart


# Distinct dashed-line color + label per triggered disturbance event type,
# drawn on the flow chart at the exact t it was triggered.
EVENT_STYLE = {
    "occlusion": {"color": "orange", "label": "Blockage"},
    "pressure": {"color": "purple", "label": "Bag/arm bump"},
    "stimulus": {"color": "red", "label": "Surgical stimulus"},  # propofol mode
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
            old_u = self.plant.u
            u = self.pid.compute(err)
            self.plant.set_command(u)
            du = self.plant.u - old_u  # PID has no discrete action; use the actual commanded change
        elif self.kind == "qlearning":
            s = discretize(err, derr, pending)
            a = self.agent.select_action(s, greedy=True)
            du = ACTIONS[a]
            self.plant.set_command(self.plant.u + du)
        elif self.kind == "dqn":
            k_hat = self.k_eff_estimator.estimate
            state = _normalize_state(err, derr, pending, k_hat, ERR_CLIP, DERR_CLIP, PENDING_CLIP)
            a = self.agent.select_action(state, greedy=True)
            du = ACTIONS[a]
            self.plant.set_command(self.plant.u + du)

        self.plant.apply_occlusion(k_eff)
        self.plant.apply_pressure_disturbance(d_p)
        measured = self.plant.step()

        if self.kind == "dqn":
            self.k_eff_estimator.update(self.plant.u, measured)

        new_err = target - measured
        reward = iv_reward(new_err, du)  # same formula IVInfusionEnv.step trains on
        self.prev_err = new_err
        return measured, self.plant.u, reward


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
        measured, command, reward = ctrl.step(ss.target, k_eff, ss.d_p)
        row[f"{name} flow"] = measured
        row[f"{name} command"] = command
        row[f"{name} reward"] = reward
    ss.rows.append(row)
    ss.t += 1


# ===========================================================================
# Propofol (Schnider) mode -- an additive second simulator. Everything below
# uses its own "pf_"-prefixed session-state keys, so switching modes never
# touches the IV-flow session above. Selected via the sidebar "Simulator" box;
# the IV-flow dashboard further down is unchanged and remains the default.
# ===========================================================================
SIM_IV = "IV Flow Control"
SIM_PROPOFOL = "Propofol (Schnider)"

PF_QTABLE_PATH = MODEL_PATHS["qlearning"]   # results/propofol_v2_qlearning_qtable.pkl
PF_DQN_PATH = MODEL_PATHS["dqn"]             # results/propofol_v2_dqn_model.pt
PF_DT = 5.0                 # s per control step, matches PropofolEnv
PF_STEPS_PER_TICK = 6       # 30 simulated seconds per screen refresh
PF_SAFE_BAND = (40.0, 60.0)
PF_BIS_Y_DOMAIN = (0, 100)
PF_CE_Y_DOMAIN = (0, 8)
PF_RATE_Y_DOMAIN = (0, 42)   # infusion levels go to 40; the 1 mg/kg bolus step clips at the top

PF_METRIC_ROWS = {
    "time_in_target_pct": "Time in target (+/-5 BIS) %",
    "pct_bis_below_40": "% time BIS < 40 (too deep)",
    "pct_bis_above_60": "% time BIS > 60 (too light)",
    "IAE": "Total tracking error (IAE, BIS*s)",
    "total_propofol_mg": "Total propofol delivered (mg)",
}


@st.cache_resource
def load_propofol_qlearning_agent(path=PF_QTABLE_PATH):
    probe = PropofolEnv()
    agent = QLearningAgent(probe.n_states, probe.n_actions)
    agent.load(path)
    return agent


@st.cache_resource
def load_propofol_dqn_agent(path=PF_DQN_PATH):
    agent = make_dqn_agent(PropofolEnv().n_actions)
    agent.load(path)
    return agent


class PropofolSimController:
    """Propofol counterpart of SimController: wraps one controller (PID /
    Q-learning / DQN) driving its own PropofolEnv. All three envs are built
    from the same seed and patient, so they share the same PD sensitivity
    (Ce50, gamma) and BIS-noise sequence; stimuli are applied live by the app
    to every controller identically. The target is held flat at the chosen
    BIS (the env's 60->50 training profile is overridden)."""

    def __init__(self, kind, seed, patient, target_bis, episode_len_s, agent=None):
        self.kind = kind
        self.agent = agent
        self.env = PropofolEnv(seed=seed, mode="eval", patient=patient, stimuli=False,
                               extra_target_steps=False, episode_len=episode_len_s)
        self.env.reset()
        self.env.target_schedule[:] = target_bis
        self.env.target = float(target_bis)
        self.env.err = self.env.target - self.env.bis
        self.done = False
        if kind == "pid":
            self.pid = CascadePID(self.env, **TUNED_PID_GAINS)
        else:
            # same decision interval / bolus-hold rule as training (ActionHold)
            self.hold_k = DECISION_EVERY[kind]
            self.action, self.level = 0, 0

    def step(self, stim_shift):
        if self.done:
            return None
        env = self.env
        env.stim_shift[env.step_idx] = stim_shift
        if self.kind == "pid":
            _, _, self.done, _ = self.pid.step()
            return env.history[-1]
        if env.step_idx % self.hold_k == 0:          # new decision
            obs = env._discretize() if self.kind == "qlearning" else env.continuous_state()
            if self.action != env.BOLUS_ACTION:
                self.level = self.action
            self.action = self.agent.select_action(obs, greedy=True)
            sub = self.action
        else:
            sub = self.action if self.action != env.BOLUS_ACTION else self.level
        _, _, self.done, _ = env.step(sub)
        return env.history[-1]


def pf_init_session(seed, patient, target_bis, episode_min, ql_agent, dqn_agent):
    episode_len_s = episode_min * 60.0
    controllers = {"PID (classic controller)": PropofolSimController(
        "pid", seed, patient, target_bis, episode_len_s)}
    if ql_agent is not None:
        controllers["Q-learning (RL)"] = PropofolSimController(
            "qlearning", seed, patient, target_bis, episode_len_s, agent=ql_agent)
    if dqn_agent is not None:
        controllers["DQN (RL)"] = PropofolSimController(
            "dqn", seed, patient, target_bis, episode_len_s, agent=dqn_agent)
    ss = st.session_state
    ss.pf_controllers = controllers
    ss.pf_t = 0
    ss.pf_n_steps = int(episode_len_s / PF_DT)
    ss.pf_target = float(target_bis)
    ss.pf_stim_until = -1
    ss.pf_stim_shift = 0.0
    ss.pf_rows = []
    ss.pf_events = []


def pf_step_all():
    ss = st.session_state
    for _ in range(PF_STEPS_PER_TICK):
        if ss.pf_t >= ss.pf_n_steps:
            ss.pf_auto_running = False
            return
        shift = ss.pf_stim_shift if ss.pf_t < ss.pf_stim_until else 0.0
        row = {"t": ss.pf_t * PF_DT / 60.0, "target": ss.pf_target, "stimulus": shift}
        for name, ctrl in ss.pf_controllers.items():
            h = ctrl.step(shift)
            row[f"{name} BIS"] = h["measured"] if h else np.nan
            row[f"{name} Ce"] = h["ce"] if h else np.nan
            row[f"{name} rate"] = h["rate_mg_kg_h"] if h else np.nan
            row[f"{name} reward"] = h["reward"] if h else np.nan
        ss.pf_rows.append(row)
        ss.pf_t += 1


def pf_line_chart(df, cols, y_domain, y_title, rename_suffix=""):
    """Like fixed_range_line_chart, but time in minutes and series labels
    stripped of the per-chart suffix (" BIS" / " Ce" / " rate")."""
    long_df = df[cols].reset_index().melt("t", var_name="series", value_name="value")
    if rename_suffix:
        long_df["series"] = long_df["series"].str.replace(rename_suffix, "", regex=False)
    series_names = sorted(long_df["series"].unique().tolist())
    return (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("t", title="Time (min)"),
            y=alt.Y("value", title=y_title, scale=alt.Scale(domain=list(y_domain), clamp=True)),
            color=alt.Color("series", title=None, scale=series_color_scale(series_names)),
        )
    )


def pf_bis_chart(df, names, events):
    band = (
        alt.Chart(pd.DataFrame({"lo": [PF_SAFE_BAND[0]], "hi": [PF_SAFE_BAND[1]]}))
        .mark_rect(color="#E8F5E9", opacity=0.9)
        .encode(y=alt.Y("lo:Q", scale=alt.Scale(domain=list(PF_BIS_Y_DOMAIN))), y2="hi:Q")
    )
    lines = pf_line_chart(df, ["target"] + [f"{n} BIS" for n in names], PF_BIS_Y_DOMAIN,
                          "BIS (0-100)", rename_suffix=" BIS")
    return add_event_markers(alt.layer(band, lines), events)


def pf_live_metrics(ss):
    rows = {}
    for name, ctrl in ss.pf_controllers.items():
        hist = ctrl.env.history
        if not hist:
            continue
        # n_steps = elapsed steps: a controller that ended the episode early
        # (BIS < 20) is scored as staying there up to "now"
        m = compute_propofol_metrics(hist, [], max(ss.pf_t, len(hist)), PF_DT)
        row = {label: m[key] for key, label in PF_METRIC_ROWS.items()}
        row["Total episode reward"] = float(sum(h["reward"] for h in hist))
        rows[name] = row
    return pd.DataFrame(rows).round(1)


def render_propofol_dashboard():
    ss = st.session_state
    st.title("Propofol Anesthesia Control: PID vs Q-learning vs DQN")
    if "pf_auto_running" not in ss:
        ss.pf_auto_running = False

    st.markdown(
        """
        <style>
        div.st-key-pf_start_stop_controls button[kind="primary"] {
            background-color: #d32f2f; border-color: #d32f2f; color: white;
        }
        div.st-key-pf_start_stop_controls button[kind="primary"]:hover {
            background-color: #b71c1c; border-color: #b71c1c; color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="pf_start_stop_controls"):
        col_start, col_stop, _col_spacer = st.columns([1, 1, 4])
        if col_start.button("START", type="primary", key="pf_start"):
            ss.pf_auto_running = True
        if col_stop.button("STOP", type="primary", key="pf_stop"):
            ss.pf_auto_running = False

    with st.sidebar:
        st.header("Setup")
        target_bis = st.slider("Target BIS", 40, 60, 50, step=1,
                               help="Depth-of-anesthesia target. 40-60 is the usual general-anesthesia band.")
        episode_min = st.slider("Episode length (minutes)", 10, 60, 30, step=5)
        seed = st.number_input(
            "Repeat-test number", value=0, step=1, key="pf_seed",
            help="Same number = identical patient sensitivity (Ce50, gamma) and BIS "
                 "sensor noise for all three controllers. (Random seed)",
        )
        st.subheader("Patient")
        age = st.slider("Age (years)", 20, 80, 53)
        weight = st.slider("Weight (kg)", 50, 110, 77)
        height = st.slider("Height (cm)", 150, 195, 177)
        sex = st.radio("Sex", ["male", "female"], horizontal=True)
        st.caption("Defaults = the Schnider reference patient (53 y, 77 kg, 177 cm, male).")

        ql_agent, dqn_agent = None, None
        try:
            ql_agent = load_propofol_qlearning_agent()
        except FileNotFoundError:
            st.warning(f"No trained propofol Q-table found at {PF_QTABLE_PATH} -- run "
                       "`python -m src.propofol_train --agent qlearning --episodes 20000 --seed 0` first.")
        try:
            dqn_agent = load_propofol_dqn_agent()
        except FileNotFoundError:
            st.warning(f"No trained propofol DQN model found at {PF_DQN_PATH} -- run "
                       "`python -m src.propofol_train --agent dqn --episodes 1500 --seed 0` first.")

        patient = dict(age=float(age), weight=float(weight), height=float(height), sex=sex)
        if st.button("Reset episode", type="primary", key="pf_reset"):
            pf_init_session(int(seed), patient, target_bis, episode_min, ql_agent, dqn_agent)

        st.divider()
        st.subheader("Live disturbance")
        stim_pct = st.slider(
            "Stimulus strength (% Ce50 increase)", 15, 30, 25, step=1,
            help="Surgical stimulation (incision, intubation) makes the patient need "
                 "more drug for the same depth: the effective Ce50 rises by this much.",
        )
        stim_min = st.slider("Stimulus length (minutes)", 2, 5, 3, step=1)
        if st.button("Simulate surgical stimulus", help="Trigger a Ce50 upward shift now"):
            if "pf_controllers" in ss:
                ss.pf_stim_shift = stim_pct / 100.0
                ss.pf_stim_until = ss.pf_t + int(stim_min * 60 / PF_DT)
                ss.pf_events.append({"t": ss.pf_t * PF_DT / 60.0, "type": "stimulus"})

    if "pf_controllers" not in ss:
        pf_init_session(int(seed), patient, target_bis, episode_min, ql_agent, dqn_agent)

    if ss.pf_auto_running and ss.pf_t < ss.pf_n_steps:
        pf_step_all()

    if ss.pf_rows:
        df = pd.DataFrame(ss.pf_rows).set_index("t")
        names = list(ss.pf_controllers.keys())
        events = ss.get("pf_events", [])

        st.subheader("Graph 1 - BIS (Depth of Anesthesia) vs Time")
        st.caption(f"Main result. The closer a controller's line stays to the target "
                   f"({ss.pf_target:.0f}) and inside the 40-60 band, the better.")
        st.altair_chart(_apply_light_config(pf_bis_chart(df, names, events)), width="stretch", key="pf_bis_chart")
        st.subheader("Graph 2 - Effect-Site Concentration (Ce) vs Time")
        st.caption("Drug concentration at the brain. BIS follows this curve with a delay.")
        ce_chart = pf_line_chart(df, [f"{n} Ce" for n in names], PF_CE_Y_DOMAIN, "Ce (ug/mL)",
                                 rename_suffix=" Ce")
        st.altair_chart(_apply_light_config(add_event_markers(ce_chart, events)), width="stretch", key="pf_ce_chart")
        st.subheader("Graph 3 - Propofol Infusion Rate vs Time")
        st.caption("The control action. Smooth is better, constant spikes mean unstable control.")
        rate_chart = pf_line_chart(df, [f"{n} rate" for n in names], PF_RATE_Y_DOMAIN,
                                   "Infusion (mg/kg/h)", rename_suffix=" rate")
        st.altair_chart(_apply_light_config(add_event_markers(rate_chart, events)), width="stretch", key="pf_rate_chart")

        st.subheader("Graph 4 - Training Reward per Episode (RL learning curve)")
        st.caption("Reward the agent earned in each training episode. A rising, flattening "
                   "curve means it is learning.")
        pf_curves = {
            "Q-learning (RL)": load_reward_curve(f"{model_prefix('qlearning')}_rewards.npy"),
            "DQN (RL)": load_reward_curve(f"{model_prefix('dqn')}_rewards.npy"),
        }
        pf_train_chart = training_reward_chart(pf_curves)
        if pf_train_chart is not None:
            st.altair_chart(_apply_light_config(pf_train_chart), width="stretch", key="pf_train_reward_chart")
        missing = [n for n, a in pf_curves.items() if a is None]
        if missing:
            st.info(
                "No saved training-reward log for " + ", ".join(missing) + " -- run "
                "`python -m src.propofol_train --agent qlearning --episodes 20000 --seed 0` and/or "
                "`python -m src.propofol_train --agent dqn --episodes 1500 --seed 0` to generate it."
            )

        st.subheader("Graph 5 - Reward per Step (this episode)")
        st.caption("Higher is better. Shows how each controller scores moment by moment under "
                   "the same reward function. PID does not learn from this reward -- it is "
                   "computed on PID's trajectory for comparison only.")
        reward_cols = [f"{n} reward" for n in names]
        pf_reward_chart = reward_chart(df, reward_cols, "Time (min)", "Reward per step",
                                       rename_suffix=" reward")
        st.altair_chart(_apply_light_config(add_event_markers(pf_reward_chart, events)),
                        width="stretch", key="pf_reward_step_chart")

        st.subheader("Graph 5b - Cumulative Reward (this episode)")
        st.caption("Running total of Graph 5. Higher (less negative) is better.")
        pf_cum_df = cumulative_reward_df(df, names)
        pf_cum_chart = reward_chart(pf_cum_df, names, "Time (min)", "Cumulative reward")
        st.altair_chart(_apply_light_config(pf_cum_chart), width="stretch", key="pf_cum_reward_chart")

        st.subheader("Table 1 - Performance Metrics (this episode)")
        st.caption("Time in target, % time BIS<40 (too deep), % time BIS>60 (too light), IAE, "
                   "total propofol.")
        st.dataframe(pf_live_metrics(ss), width="stretch")
    else:
        st.info("Click **START** to run, or **Reset episode** in the sidebar to apply new settings.")

    any_env = next(iter(ss.pf_controllers.values())).env
    st.caption(f"t = {ss.pf_t * PF_DT / 60.0:.1f} / {ss.pf_n_steps * PF_DT / 60.0:.0f} min   "
               f"patient sensitivity this run: Ce50 = {any_env.bis_model.ce50:.2f} ug/mL, "
               f"gamma = {any_env.bis_model.gamma:.2f}   current Ce50 shift = "
               f"{(ss.pf_stim_shift if ss.pf_t < ss.pf_stim_until else 0.0) * 100:.0f}%")

    if ss.pf_auto_running and ss.pf_t < ss.pf_n_steps:
        time.sleep(0.05)
        st.rerun()


show_splash_once()

simulator = st.sidebar.selectbox("Simulator", [SIM_IV, SIM_PROPOFOL], index=0, key="simulator")
if simulator == SIM_PROPOFOL:
    render_propofol_dashboard()
    st.stop()

st.title("IV Infusion Flow Control: PID vs Q-learning vs DQN")

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

    st.subheader("Graph 1 - Flow Rate vs Time (controller comparison)")
    st.caption("Main result. The closer a controller's line stays to the target flow, the better; "
               "dashed markers show triggered blockages / bag movements.")
    flow_cols = ["target"] + [f"{name} flow" for name in active_names]
    flow_chart = fixed_range_line_chart(df, flow_cols, FLOW_Y_DOMAIN, "Flow (mL/hr)")
    flow_chart = add_event_markers(flow_chart, st.session_state.get("events", []))
    st.altair_chart(_apply_light_config(flow_chart), width="stretch", key="flow_chart")

    st.subheader("Graph 2 - Training Reward per Episode (RL learning curve)")
    st.caption("Reward the agent earned in each training episode. A rising, flattening "
               "curve means it is learning.")
    iv_curves = {
        "Q-learning (RL)": load_reward_curve("results/qlearning_rewards.npy"),
        "DQN (RL)": load_reward_curve("results/dqn_rewards.npy"),
    }
    iv_train_chart = training_reward_chart(iv_curves)
    if iv_train_chart is not None:
        st.altair_chart(_apply_light_config(iv_train_chart), width="stretch", key="iv_train_reward_chart")
    missing = [n for n, a in iv_curves.items() if a is None]
    if missing:
        st.info(
            "No saved training-reward log for " + ", ".join(missing) + " -- run "
            "`python -m src.train --episodes 5000 --seed 0` and/or "
            "`python -m src.dqn_agent --episodes 2000 --seed 0` to generate it."
        )

    st.subheader("Graph 3 - Reward per Step (this episode)")
    st.caption("Higher is better. Shows how each controller scores moment by moment under "
               "the same reward function. PID does not learn from this reward -- it is "
               "computed on PID's trajectory for comparison only.")
    iv_reward_cols = [f"{name} reward" for name in active_names]
    iv_reward_chart = reward_chart(df, iv_reward_cols, "Time (s)", "Reward per step",
                                   rename_suffix=" reward")
    st.altair_chart(_apply_light_config(add_event_markers(iv_reward_chart, st.session_state.get("events", []))),
                    width="stretch", key="iv_reward_step_chart")

    st.subheader("Running error stats (this episode so far)")
    abs_err = {name: (df["target"] - df[f"{name} flow"]).abs() for name in active_names}
    cols = st.columns(len(active_names))
    for col, name in zip(cols, active_names):
        col.metric(f"{name}: average tracking error (mL/hr)", f"{abs_err[name].mean():.2f}")
        col.metric(f"{name}: flow right now (mL/hr)", f"{df[f'{name} flow'].iloc[-1]:.1f}")

else:
    st.info("Click **Reset episode** in the sidebar to start.")

st.caption(f"t = {st.session_state.get('t', 0)} / {st.session_state.get('n_steps', 0)} s   "
           f"current k_eff = {st.session_state.get('k_eff', 1.0):.2f}")

if st.session_state.auto_running and st.session_state.t < st.session_state.n_steps:
    time.sleep(0.05)
    st.rerun()
