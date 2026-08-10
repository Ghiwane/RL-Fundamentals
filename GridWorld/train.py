from agent import QLearningAgent
from environment import GridWorld
import matplotlib.pyplot as plt
import numpy as np

# Initialize the environment and Q-Learning agent
env = GridWorld()
agent = QLearningAgent()

episode_rewards = []  # List to store cumulative reward for each episode
episodes = 2000       # Total number of training episodes

# Main training loop over episodes
for e in range(episodes):
    state = env.reset()       # Reset environment to initial state
    done = False
    total_rewards = 0        # Accumulator for total reward in the current episode

    # Step-by-step episode loop
    while not done:
        action = agent.choose_action(state)                     # Select action using epsilon-greedy policy
        next_state, reward, done = env.step(action)             # Execute action in environment
        agent.update(state, action, reward, next_state, done)   # Update Q-table with step outcome
        state = next_state                                      # Transition to next state
        total_rewards += reward                                 # Accumulate step reward

    agent.decay_epsilon()                  # Decay exploration rate (epsilon) after each episode
    episode_rewards.append(total_rewards)  # Record total episode reward

    # Log training progress every 50 episodes
    if e % 50 == 0:
        print(f'Episode : {e} | Reward : {total_rewards} | Current epsilon : {agent.eps}')


# --- MATPLOTLIB PLOT ---
window_size = 50  # Window size for calculating the moving average

# Compute rolling average using 1D convolution
rolling_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')

# Plot the smoothed learning curve (moving average)
plt.plot(range(window_size - 1, episodes), rolling_avg, color='blue', linewidth=2, label=f'Moyenne glissante ({window_size} épisodes)')

# Configure plot title, axis labels, reference line, legend, and grid
plt.title('Training Q-Learning - GridWorld')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.axhline(y=3, color='r', linestyle='--', alpha=0.5)  # Reference benchmark line
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Render and display the plot window
plt.show()