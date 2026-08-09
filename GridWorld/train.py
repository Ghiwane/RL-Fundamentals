from agent import QLearningAgent
from environment import GridWorld
import matplotlib.pyplot as plt
import numpy as np

env = GridWorld()
agent = QLearningAgent()

episode_rewards =[]
episodes = 2000

for e in range(episodes):
    state = env.reset()
    done = False
    total_rewards = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_rewards += reward

    agent.decay_epsilon()
    episode_rewards.append(total_rewards)

    if e % 50 == 0:
        print(f'Episode : {e} | Reward : {total_rewards} | Current epsilon : {agent.eps}')


# --- MATPLOTLIB PLOT ---
window_size = 50

rolling_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')


plt.plot(range(window_size - 1, episodes), rolling_avg, color='blue', linewidth=2, label=f'Moyenne glissante ({window_size} épisodes)')

plt.title('Entraînement Q-Learning - GridWorld')
plt.xlabel('Épisode')
plt.ylabel('Récompense totale')
plt.axhline(y=3, color='r', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()