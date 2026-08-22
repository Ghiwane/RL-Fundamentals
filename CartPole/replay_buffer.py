from collections import deque
import random
import torch
import numpy as np

# Replay memory buffer to store and sample past experience transitions
class ReplayBuffer():
    def __init__(self, capacity):
        # Double-ended queue with fixed capacity (automatically discards oldest elements)
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        # Return total number of stored transitions
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        # Save a single step transition into the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a mini-batch of transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack tuple elements into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert arrays and lists into PyTorch Float and Long tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)          # Shape: (batch_size, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)        # Shape: (batch_size, 1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)  # Shape: (batch_size, 1)
        
        return states, actions, rewards, next_states, dones