"""
propofol_metrics.py
--------------------

Metrics for PropofolEnv episodes (e = target_BIS - measured_BIS, dt = 5 s):
  Reused from src/metrics.compute_metrics (unchanged):
    IAE = sum|e|*dt   ISE = sum e^2*dt   ITAE = sum t|e|*dt   ITSE = sum t e^2*dt
    RMSE = sqrt(mean e^2)
    time_in_target_pct = mean(|e| <= 5) * 100   (= compute_metrics' pct_time_in_safe_band
                                                  with settle_band=5)
  Propofol-specific:
    pct_bis_below_40 / pct_bis_above_60 = % of the 60-min horizon with measured
                                          BIS < 40 (too deep) / > 60 (too light)
    induction_time_s   = first t at which |e| <= 5 holds for 12 consecutive steps
                         (60 s); horizon length if never
    settling_time_s    = mean over target changes (end of the 60->50 ramp at
                         15 min, plus the optional extra step) of the time from
                         the change until |e| <= 5 holds for 12 consecutive
                         steps, censored at the end of that target segment
    total_propofol_mg  = sum(u_mg_per_s * dt)
    total_propofol_mg includes boluses; bolus_mg = bolus part alone
    mean_rate_mg_kg_h  = mean commanded infusion (excluding boluses) over the
                         steps actually run
    terminated_early   = 1.0 if the episode ended on BIS < 20

Early-terminated episodes are padded to the full horizon by holding the last
measured BIS (with zero infusion) so a controller that overdoses and ends the
episode early is scored on the full hour rather than rewarded with a short,
low-IAE history. compute_metrics' overshoot/settling are not reported: with
BIS starting at ~95 and a downward induction step they are not meaningful.
"""

import numpy as np

from src.metrics import compute_metrics

SUSTAIN_STEPS = 12  # 60 s at dt=5 s


def pad_history(history, n_steps, dt):
    """Pad an early-terminated history to n_steps by holding the last BIS."""
    if len(history) >= n_steps or not history:
        return list(history)
    padded = list(history)
    last = history[-1]
    for k in range(len(history), n_steps):
        row = dict(last)
        row.update(t=k * dt, rate_mg_kg_h=0.0, rate_mg_s=0.0, command=0.0, bolus_mg=0.0, padded=True)
        padded.append(row)
    return padded


def _first_sustained(abs_err, start, stop, band, sustain):
    """First index k in [start, stop) such that abs_err <= band for `sustain`
    consecutive steps starting at k (all of a segment shorter than `sustain`).
    None if it never happens."""
    need = min(sustain, stop - start)
    run = 0
    for k in range(start, stop):
        run = run + 1 if abs_err[k] <= band else 0
        if run >= need:
            return k - run + 1
    return None


def compute_propofol_metrics(history, change_steps, n_steps, dt=5.0, band=5.0):
    padded = pad_history(history, n_steps, dt)
    base = compute_metrics(padded, settle_band=band)

    bis = np.array([h["measured"] for h in padded])
    err = np.array([h["error"] for h in padded])
    abs_err = np.abs(err)
    horizon = len(padded) * dt

    k_ind = _first_sustained(abs_err, 0, len(padded), band, SUSTAIN_STEPS)
    induction_time = horizon if k_ind is None else k_ind * dt

    settle_times = []
    bounds = sorted(int(c) for c in change_steps if 0 <= c < len(padded))
    for i, c in enumerate(bounds):
        stop = bounds[i + 1] if i + 1 < len(bounds) else len(padded)
        k = _first_sustained(abs_err, c, stop, band, SUSTAIN_STEPS)
        settle_times.append((stop - c) * dt if k is None else (k - c) * dt)
    settling = float(np.mean(settle_times)) if settle_times else float("nan")

    return dict(
        IAE=base["IAE"], ISE=base["ISE"], ITAE=base["ITAE"], ITSE=base["ITSE"], RMSE=base["RMSE"],
        time_in_target_pct=base["pct_time_in_safe_band"],
        pct_bis_below_40=float(np.mean(bis < 40.0) * 100.0),
        pct_bis_above_60=float(np.mean(bis > 60.0) * 100.0),
        induction_time_s=float(induction_time),
        settling_time_s=settling,
        total_propofol_mg=float(sum(h["rate_mg_s"] for h in history) * dt),
        mean_rate_mg_kg_h=float(np.mean([h["command"] for h in history])) if history else 0.0,
        bolus_mg=float(sum(h.get("bolus_mg", 0.0) for h in history)),
        terminated_early=float(len(history) < n_steps),
    )
