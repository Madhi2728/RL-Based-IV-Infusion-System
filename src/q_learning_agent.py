"""
q_learning_agent.py
--------------------
Tabular Q-learning agent for discretized IV infusion flow control.

Update rule:
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Epsilon-greedy exploration with exponential decay across episodes.
"""

import numpy as np
import pickle


class QLearningAgent:
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
