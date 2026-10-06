"""
q_learning_agent.py
--------------------

Update rule:  Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
              (target is just r when the episode is done)
Policy:       epsilon-greedy; eps <- max(eps_end, eps * eps_decay), once per episode
Defaults:     alpha=0.15  gamma=0.95  eps_start=1.0  eps_end=0.05  eps_decay=0.995
Q-table:      n_states x n_actions, zero-initialised
              (IVInfusionEnv: 1617 x 7)
Tabular Q-learning agent for discretized IV infusion flow control.

Update rule:
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Epsilon-greedy exploration with exponential decay across episodes.
"""

import numpy as np
import pickle


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration.

    Update rule (per update(s, a, r, s_next, done)), with alpha=0.15 and
    gamma=0.95 (this module's defaults):

        target  = r                              if done
                = r + gamma * max_a' Q(s', a')    otherwise
        Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))

    Action selection (per select_action(state, greedy)):

        with probability eps (unless greedy=True): choose a uniformly random
        action; otherwise choose argmax_a Q(state, a).

    Epsilon decay schedule (per decay_epsilon(), called once per episode):

        eps <- max(eps_end, eps * eps_decay)

      starting at eps_start=1.0, decaying by eps_decay=0.995 per episode,
      floored at eps_end=0.05 (this module's defaults).
    """

    def __init__(self, n_states, n_actions, alpha=0.15, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state, greedy=False):
        if (not greedy) and self.rng.random() < self.eps:
            return self.rng.integers(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)
