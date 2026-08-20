from network import QNetwork
from replay_buffer import ReplayBuffer
import torch
import numpy as np
import random
import torch.nn.functional as F

class CartpoleAgent():
    def __init__(self, state, action):
        self.q_network = QNetwork(state, action)
        self.target_network = QNetwork(state, action)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        self.buffer = ReplayBuffer(capacity=10000)

        self.gamma = 0.99
        self.eps = 1.0 
        self.eps_min = 0.1
        self.eps_decay = 0.999
        self.batch_size = 64
        self.n_actions = action
        self.n_step = 0
        self.target_update_freq = 500

    def choose_action(self, state):
            rand_nb = np.random.rand()
            if rand_nb < self.eps:
                return random.choice(range(self.n_actions))
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state)
                    return self.q_network(state_tensor).argmax().item()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        q_values = self.q_network(states)
        selected_q_values = q_values.gather(1, actions)

        with torch.no_grad():
            best_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_network(next_states).gather(1, best_actions)
            target = rewards + self.gamma * max_next_q * (1 - dones.float())
        loss = F.mse_loss(selected_q_values, target)

        self.n_step += 1 
        if self.n_step % self.target_update_freq == 0:
            self.update_target()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)