import torch 
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state, action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action)
        )

    def forward(self, x):
        return self.net(x)