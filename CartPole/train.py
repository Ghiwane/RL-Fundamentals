import gymnasium as gym
from agent import CartpoleAgent
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

# Set random seeds so the results are repeatable
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Reset the environment to get the first state
state, _ = env.reset(seed=SEED)

# Get state size (4) and number of actions (2)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize the agent
agent = CartpoleAgent(state_dim, action_dim)

# Training settings
episodes = 6000
episode_rewards = []

best_eval_reward = -float('inf')
eval_freq = 100  # Evaluate every 100 episodes

# Main training loop
for i in range(episodes):
    state, _ = env.reset()
    total_rewards = 0
    done = False

    # Play one full episode
    while not done:
        # Choose action (exploration vs exploitation)
        action = agent.choose_action(state)
        
        # Take action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Save experience in the replay buffer
        agent.buffer.push(state, action, reward, next_state, done)
        
        # Train the neural network on a mini-batch
        loss = agent.train_step()
        
        # Move to the next state
        state = next_state
        total_rewards += reward  

    # Reduce exploration rate (epsilon) after each episode
    agent.decay_epsilon()
    episode_rewards.append(total_rewards)
    
    # Evaluate the agent periodically
    if i % eval_freq == 0:
        eval_reward = agent.evaluate(env, n_episodes=10)
        print(f'Episode eval {i} : {eval_reward:.1f}')
        
        # Save the model if it achieves a new best score
        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            torch.save(agent.q_network.state_dict(), "trained_model.pth")
            print(f'New best model saved ({eval_reward:.1f})')

    # Print training progress every 20 episodes
    if i % 20 == 0:
        print(f'Episode : {i} | Reward : {total_rewards} | Current epsilon : {agent.eps}')

    if i % 20 == 0:
        print(f"Loss: {loss}")


# --- MATPLOTLIB PLOT ---
window_size = 500  # Window size for the moving average

# Calculate smooth moving average over time
rolling_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')

# Plot the smoothed learning curve
plt.plot(range(window_size - 1, episodes), rolling_avg, color='blue', linewidth=2, label=f'Moving Average ({window_size} episodes)')

# Add title, labels, target line, legend, and grid
plt.title('Training - Cartpole-v1')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.axhline(y=500, color='r', linestyle='--', alpha=0.5)  # Target reward line at 500
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Display the plot
plt.show()