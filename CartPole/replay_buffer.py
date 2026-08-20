from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states))
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones