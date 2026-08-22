import gymnasium as gym
import torch
from network import QNetwork

# Create the CartPole environment with visual rendering
env = gym.make("CartPole-v1", render_mode="human")

# Get state size (4) and number of actions (2)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize the neural network and load trained weights
q_network = QNetwork(state_dim, action_dim)
q_network.load_state_dict(torch.load("trained_model.pth"))
q_network.eval()  # Set network to evaluation mode (no dropout/batchnorm)

n_episodes = 10

# Demo loop across multiple episodes
for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    # Run single episode using trained model
    while not done:
        with torch.no_grad():  # Disable gradient tracking for testing
            state_tensor = torch.FloatTensor(state)
            action = q_network(state_tensor).argmax().item()  # Pick best action
            
        # Step the environment with chosen action
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f'Episode {episode} | Reward : {total_reward}')

# Close environment window
env.close()