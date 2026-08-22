import torch 
import torch.nn as nn

# Multi-Layer Perceptron (MLP) mapping states to Q-values
class QNetwork(nn.Module):
    def __init__(self, state, action):
        super().__init__()
        
        # Neural network layers: 4 inputs -> 64 -> 64 -> 2 outputs
        self.net = nn.Sequential(
            nn.Linear(state, 64),   # Input layer (state dimension)
            nn.ReLU(),              # Activation function
            nn.Linear(64, 64),      # Hidden layer
            nn.ReLU(),              # Activation function
            nn.Linear(64, action)   # Output layer (Q-value for each action)
        )

    def forward(self, x):
        # Forward pass: returns predicted Q-values for given input state x
        return self.net(x)