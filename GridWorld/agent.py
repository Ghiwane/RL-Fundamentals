import numpy as np
import random

class QLearningAgent():
    """Q-Learning agent for tabular reinforcement learning."""

    def __init__(self):
        """Initialize hyperparameters and Q-table."""
        self.alpha = 0.5              # Learning rate
        self.gamma = 0.5              # Discount factor
        self.eps = 1.0                # Initial exploration rate (epsilon)
        self.eps_min = 0.1            # Minimum exploration rate
        self.eps_decay = 0.996        # Epsilon decay factor per episode
        self.q_table = np.zeros((5, 5, 4)) # Q-table initialized with zeros: (grid_x, grid_y, actions)

    def choose_action(self, state):
        """Select an action using the epsilon-greedy policy."""
        rand_nb = np.random.rand()
        if rand_nb < self.eps:
            return random.choice((0, 1, 2, 3)) # Explore: select a random action
        else:
            x, y = state
            return np.argmax(self.q_table[x, y]) # Exploit: select the best action from Q-table

    def update(self, state, action, reward, next_state, done):
        """Update Q-value for a state-action pair using TD learning rule."""
        s_x, s_y = state
        ns_x, ns_y = next_state
        
        current_q = self.q_table[s_x, s_y, action] # Current Q-value
        if done:
            target = reward # Terminal state: no future rewards
        else: 
            target = reward + self.gamma * np.max(self.q_table[ns_x, ns_y]) # Bellman target

        td_error = target - current_q # Temporal Difference (TD) error
        self.q_table[s_x, s_y][action] = current_q + self.alpha * td_error # Update Q-value

    def decay_epsilon(self):
        """Decay the exploration rate (epsilon) after each episode."""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)