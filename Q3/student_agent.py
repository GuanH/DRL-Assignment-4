import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(67, 2048), nn.SiLU(),
            nn.Linear(2048, 2048), nn.SiLU(),
            nn.Linear(2048, 2048), nn.SiLU(),
        )
        self.h1 = nn.Sequential(
            nn.Linear(2048, 2048), nn.SiLU(),
            nn.Linear(2048, 21))
        self.h2 = nn.Sequential(
            nn.Linear(2048, 2048), nn.SiLU(),
            nn.Linear(2048, 21))

    def forward(self, x):
        x = self.fc(x)
        return self.h1(x), torch.clamp(self.h2(x), -20, 2).exp()




# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.device = torch.device("cuda")
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.actor = Actor().to(self.device)
        self.actor.load_state_dict(torch.load('model.pth'), map_location=self.device)

    def act(self, observation):
        with torch.no_grad():
            state = torch.tensor(observation[None,:].astype(np.float32), device=self.device)
            mu, sigma = self.actor(state)
        return np.clip(mu.cpu().numpy(), -1, 1)
