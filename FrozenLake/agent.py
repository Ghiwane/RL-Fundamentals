import numpy as np
import random

class QLearningAgent():
    """Q-Learning agent for tabular reinforcement learning."""

    def __init__(self, n_states, n_actions):
        """Initialize hyperparameters and Q-table."""
        self.alpha = 0.1              # Learning rate
        self.gamma = 0.95              # Discount factor
        self.eps = 1.0                # Initial exploration rate (epsilon)
        self.eps_min = 0.1            # Minimum exploration rate
        self.eps_decay = 0.999        # Epsilon decay factor per episode
        self.n_actions = n_actions    # Number of actions
        self.q_table = np.zeros((n_states, n_actions)) # Q-table initialized with zeros: (state, actions)

    def choose_action(self, state):
        """Select an action using the epsilon-greedy policy."""
        rand_nb = np.random.rand()
        if rand_nb < self.eps:
            return random.choice(range(self.n_actions)) # Explore: select a random action
        else:
            return np.argmax(self.q_table[state]) # Exploit: select the best action from Q-table

    def update(self, state, action, reward, next_state, done):
        """Update Q-value for a state-action pair using TD learning rule."""
        current_q = self.q_table[state, action] # Current Q-value
        if done:
            target = reward # Terminal state: no future rewards
        else: 
            target = reward + self.gamma * np.max(self.q_table[next_state]) # Bellman target

        td_error = target - current_q # Temporal Difference (TD) error
        self.q_table[state][action] = current_q + self.alpha * td_error # Update Q-value

    def decay_epsilon(self):
        """Decay the exploration rate (epsilon) after each episode."""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)