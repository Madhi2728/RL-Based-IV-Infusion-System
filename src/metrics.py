"""
metrics.py
----------
Standard control-engineering metrics computed from an episode history
(list of dicts with keys: t, target, measured, error), used to compare
the Q-learning agent against the PID baseline on equal footing.
"""

import numpy as np


def compute_metrics(history, settle_band=5.0):
    t = np.array([h["t"] for h in history])
    target = np.array([h["target"] for h in history])
    measured = np.array([h["measured"] for h in history])
    error = np.array([h["error"] for h in history])
    dt = t[1] - t[0] if len(t) > 1 else 1.0

    iae = np.sum(np.abs(error)) * dt                 # integral of absolute error
    ise = np.sum(error ** 2) * dt                     # integral of squared error
    rmse = float(np.sqrt(np.mean(error ** 2)))

    # settling time: first time after which |error| stays within settle_band
    # for the remainder of the (current target segment's) run
    settle_time = None
    for i in range(len(error)):
        if np.all(np.abs(error[i:]) <= settle_band):
            settle_time = t[i]
            break
    if settle_time is None:
        settle_time = t[-1]  # never settled within the episode

    # overshoot relative to first target step, as % of that target
    first_target = target[0]
    if first_target > 0:
        overshoot_pct = max(0.0, (np.max(measured) - first_target) / first_target * 100.0)
    else:
        overshoot_pct = 0.0

    pct_in_band = float(np.mean(np.abs(error) <= settle_band) * 100.0)

    return dict(
        IAE=float(iae), ISE=float(ise), RMSE=rmse,
        settling_time_s=float(settle_time), overshoot_pct=float(overshoot_pct),
        pct_time_in_safe_band=pct_in_band,
    )
