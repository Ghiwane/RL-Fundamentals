import gymnasium as gym
from agent import CartpoleAgent
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

env = gym.make("CartPole-v1")

state, _ = env.reset(seed=SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = CartpoleAgent(state_dim, action_dim)

episodes = 5000
episode_rewards = []

for i in range(episodes):
    state, _ = env.reset()
    total_rewards = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.buffer.push(state, action, reward, next_state, done)
        loss = agent.train_step()
        state = next_state
        total_rewards += reward  

    agent.decay_epsilon()
    episode_rewards.append(total_rewards)

    if i % 200 == 0:
            print(f'Episode : {i} | Reward : {total_rewards} | Current epsilon : {agent.eps}')

    if i % 200 == 0:  # ou une autre condition pour ne pas spammer
        print(f"Loss: {loss}")

torch.save(agent.q_network.state_dict(), "trained_model.pth")

# --- MATPLOTLIB PLOT ---
window_size = 200  # Window size for calculating the moving average

# Compute rolling average using 1D convolution
rolling_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')

# Plot the smoothed learning curve (moving average)
plt.plot(range(window_size - 1, episodes), rolling_avg, color='blue', linewidth=2, label=f'Moyenne glissante ({window_size} épisodes)')

# Configure plot title, axis labels, reference line, legend, and grid
plt.title('Training - Cartpole-v1')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.axhline(y=500, color='r', linestyle='--', alpha=0.5)  # Reference benchmark line
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Render and display the plot window
plt.show()