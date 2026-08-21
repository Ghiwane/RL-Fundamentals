import gymnasium as gym
import torch
from network import QNetwork

env = gym.make("CartPole-v1", render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_network = QNetwork(state_dim, action_dim)
q_network.load_state_dict(torch.load("trained_model.pth"))
q_network.eval()

n_episodes = 10

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action = q_network(state_tensor).argmax().item()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f'Episode {episode} | Reward : {total_reward}')

env.close()