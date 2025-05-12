import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(3, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU())
        self.h1 = nn.Linear(128, 1)
        self.h2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.h1(x), torch.clamp(self.h2(x), -20, 2).exp()

 
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.device = torch.device('cuda')
        self.actor = Actor().to(self.device)
        self.actor.load_state_dict(torch.load('model.pth'))

    def act(self, observation):
        with torch.no_grad():
            state = torch.tensor(observation[None,:], device=self.device)
            mu, sigma = self.actor(state)
            action = torch.tanh(mu).item()*2
        return [action]
