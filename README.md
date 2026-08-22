# RL-Fundamentals

A collection of reinforcement learning projects built from scratch to understand the core ideas behind RL, from simple tabular Q-Learning to Deep Q-Networks.

Each folder is a self-contained environment with its own agent, training script, and (where relevant) a demo script to watch the trained agent play.

## Projects

### GridWorld
A custom 5x5 grid environment built entirely from scratch (no Gymnasium). The agent has to navigate around walls to reach a goal, learning purely through trial and error with tabular Q-Learning.

- Custom environment logic (movement, rewards, walls, episode termination)
- Q-table based agent with epsilon-greedy exploration
- Converges to the optimal path, verified with a pure-exploitation evaluation run

### FrozenLake
First adaptation of the Q-Learning agent to the Gymnasium API (`terminated` / `truncated`, `observation_space.n`). Tested on both the slippery and non-slippery variants of the environment.

### Taxi
A generalization check: the same Q-Learning agent, with only the state/action space changed, applied to a larger and more complex environment (`Taxi-v4`). No structural changes needed a good sanity check that the underlying logic was solid.

### CartPole (Deep Q-Network)
The step up from tabular Q-Learning to Deep RL. The state space here is continuous, so a neural network replaces the Q-table.

- **Network**: small MLP (`Input(4) → 64 → 64 → Output(2)`) predicting a Q-value per action
- **Experience Replay**: past transitions are stored and sampled randomly to decorrelate training data
- **Target Network**: a periodically-synced copy of the main network, used to compute stable Bellman targets
- **Double DQN**: action selection and action evaluation are split across the main and target networks to reduce Q-value overestimation
- **Best-model checkpointing**: the agent is evaluated periodically in pure exploitation mode, and only saved when it beats its previous best score

The full write-up of the debugging process, hyperparameter tuning, and the DQN instability issues encountered (and how they were solved) is in [`CartPole/README.md`](./CartPole/README.md).

## What each project builds on the last

| Project | New concept introduced |
|---|---|
| GridWorld | MDPs, Bellman equation, epsilon-greedy, TD learning |
| FrozenLake | Gymnasium API |
| Taxi | Generalizing an agent to a new, larger environment |
| CartPole | Function approximation with neural networks, Experience Replay, Target Network, Double DQN |

## Stack

Python, NumPy, Gymnasium, PyTorch, Matplotlib.

## What's next

The concepts and DQN implementation from CartPole are the foundation for a larger project: a Dino Chrome agent trained with Double DQN on a numerical (non-pixel) state representation, coded from scratch in Pygame.
