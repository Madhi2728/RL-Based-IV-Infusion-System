"""
pid_controller.py
------------------

PID law (discrete, per compute(error)):
    u = Kp*e + Ki*integral(e dt) + Kd*de/dt
    integral   = clamp(integral + e*dt, -integral_limit, +integral_limit)  (anti-windup)
    derivative = (e - e_prev) / dt
    output     = clamp(u, u_min, u_max)
Defaults (constructor): Kp=0.6  Ki=0.08  Kd=0.2  dt=1.0
    u_min=0.0  u_max=400.0  integral_limit=1500.0
    derivative_filter=None (optional EMA low-pass on the D term,
    d_f += a*(de/dt - d_f); used by the propofol PID, off for the IV PID)
Conventional PID controller used as the baseline against the Q-learning agent.
Operates on the same error signal (target - measured flow) and outputs a
pump command adjustment, so it can be dropped into the same environment loop.

u(t) = Kp*e(t) + Ki*integral(e) + Kd*de/dt   (with integral anti-windup clamp)
"""


class PIDController:
    """Discrete-time PID controller with integral anti-windup.

    Mathematical model (per compute(error), dt seconds per call):

        integral   = clip(integral + error * dt, -integral_limit, integral_limit)
        derivative = (error - prev_error) / dt
        u          = Kp * error + Ki * integral + Kd * derivative
        u          = clip(u, u_min, u_max)

      Anti-windup: the integral term is clamped to [-integral_limit,
      integral_limit] before being used, preventing it from growing
      unbounded while the output is saturated at u_min/u_max.
    """

    def __init__(self, kp=0.6, ki=0.08, kd=0.2, dt=1.0, u_min=0.0, u_max=400.0,
                 integral_limit=1500.0, derivative_filter=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.u_min, self.u_max = u_min, u_max
        self.integral_limit = integral_limit
        # Optional first-order low-pass on the D term (EMA weight in (0, 1];
        # None = unfiltered, the original behaviour the IV baseline uses).
        self.derivative_filter = derivative_filter
        self.integral = 0.0
        self.prev_err = 0.0
        self.d_filt = 0.0
        self.u = 0.0

    def reset(self, u0=0.0):
        self.integral = 0.0
        self.prev_err = 0.0
        self.d_filt = 0.0
        self.u = u0

    def compute(self, error):
        self.integral += error * self.dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        derivative = (error - self.prev_err) / self.dt
        if self.derivative_filter is not None:
            self.d_filt += self.derivative_filter * (derivative - self.d_filt)
            derivative = self.d_filt
        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_err = error
        self.u = max(self.u_min, min(self.u_max, u))
        return self.u
