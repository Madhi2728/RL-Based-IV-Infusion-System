"""
pid_controller.py
------------------
Conventional PID controller used as the baseline against the Q-learning agent.
Operates on the same error signal (target - measured flow) and outputs a
pump command adjustment, so it can be dropped into the same environment loop.

u(t) = Kp*e(t) + Ki*integral(e) + Kd*de/dt   (with integral anti-windup clamp)
"""


class PIDController:
    def __init__(self, kp=0.6, ki=0.08, kd=0.2, dt=1.0, u_min=0.0, u_max=400.0,
                 integral_limit=1500.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.u_min, self.u_max = u_min, u_max
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_err = 0.0
        self.u = 0.0

    def reset(self, u0=0.0):
        self.integral = 0.0
        self.prev_err = 0.0
        self.u = u0

    def compute(self, error):
        self.integral += error * self.dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        derivative = (error - self.prev_err) / self.dt
        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_err = error
        self.u = max(self.u_min, min(self.u_max, u))
        return self.u
