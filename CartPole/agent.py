from network import QNetwork
from replay_buffer import ReplayBuffer
import torch
import numpy as np
import random
import torch.nn.functional as F

class CartpoleAgent():
    def __init__(self, state, action):
        # Main network used to choose actions and learn
        self.q_network = QNetwork(state, action)
        
        # Target network used to compute stable Bellman targets (Double DQN)
        self.target_network = QNetwork(state, action)

        # Copy initial weights from main network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer for updating the main network's weights
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Replay memory buffer storing past transitions
        self.buffer = ReplayBuffer(capacity=10000)

        # Hyperparameters
        self.gamma = 0.99          # Discount factor for future rewards
        self.eps = 1.0             # Initial exploration rate (100% random)
        self.eps_min = 0.01        # Minimum exploration rate (1% random)
        self.eps_decay = 0.999     # Decay factor applied after each episode
        self.batch_size = 64       # Mini-batch size for gradient descent
        self.n_actions = action    # Number of possible actions (2)
        self.n_step = 0            # Global step counter
        self.target_update_freq = 1000  # Steps frequency to sync target network
        self.train_nstep = 4       # Train the network once every 4 steps

    def choose_action(self, state):
        # Epsilon-greedy strategy: explore randomly or exploit network knowledge
        rand_nb = np.random.rand()
        if rand_nb < self.eps:
            return random.choice(range(self.n_actions))  # Random action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                return self.q_network(state_tensor).argmax().item()  # Best predicted action

    def train_step(self):
        self.n_step += 1 
        
        # Train only every 4 steps
        if self.n_step % self.train_nstep == 0:
            # Wait until buffer has enough samples
            if len(self.buffer) < self.batch_size:
                return
            
            # Sample a random mini-batch from the buffer
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            
            # Predict Q-values for current states and keep chosen actions
            q_values = self.q_network(states)
            selected_q_values = q_values.gather(1, actions)

            # Compute Target Q-values using Double DQN logic
            with torch.no_grad():
                # 1. Main network selects best action for next state
                best_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                
                # 2. Target network evaluates that selected action
                max_next_q = self.target_network(next_states).gather(1, best_actions)
                
                # 3. Bellman target calculation (zero out future rewards if episode is done)
                target = rewards + self.gamma * max_next_q * (1 - dones.float())
            
            # MSE loss between predicted Q-values and Bellman targets
            loss = F.mse_loss(selected_q_values, target)

            # Periodically update target network weights
            if self.n_step % self.target_update_freq == 0:
                self.update_target()

            # Backpropagation and gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=100)
            self.optimizer.step()

            return loss.item()

    def update_target(self):
        # Sync target network weights with main network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        # Decay exploration rate down to eps_min
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def evaluate(self, env, n_episodes=10):
        # Evaluate agent performance without exploration noise (pure exploitation)
        total_rewards = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state)
                    action = self.q_network(state_tensor).argmax().item()
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)