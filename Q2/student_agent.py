import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(5, 512), nn.SiLU(),
                nn.Linear(512, 512), nn.SiLU(),
                nn.Linear(512, 512), nn.SiLU(),
                )
        self.h1 = nn.Linear(512, 1)
        self.h2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.h1(x), torch.clamp(self.h2(x), -20, 2).exp()


# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.device = torch.device("cuda")
        self.actor = Actor().to(self.device)
        self.actor.load_state_dict(torch.load('model.pth'))

    def act(self, observation):
        with torch.no_grad():
            state = torch.tensor(observation[None,:].astype(np.float32), device=self.device)
            mu, sigma = self.actor(state)
            z = Normal(mu, sigma).sample()
            action = torch.tanh(z).item()
        return action
    

