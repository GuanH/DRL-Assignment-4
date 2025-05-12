import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torch.distributions import Normal
from tqdm import tqdm

env = gym.make("Pendulum-v1", render_mode='rgb_array')

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

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(3, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 1))
    def forward(self, x):
        return self.fc(x)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(4, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 1))

    def forward(self, s, a):
        x = torch.cat((s,a), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.size = 0
        self.state = np.empty((cap, 3), dtype=np.float32)
        self.action = np.empty((cap, 1), dtype=np.float32)
        self.reward = np.empty((cap, 1), dtype=np.float32)
        self.next_state = np.empty((cap, 3), dtype=np.float32)
        self.done = np.empty((cap, 1), dtype=np.float32)
        self.pos = 0

    def add(self, state, next_state, action, reward, done):
        self.state[self.pos] = state
        self.next_state[self.pos] = next_state
        self.action[self.pos, 0] = action
        self.reward[self.pos, 0] = reward
        self.done[self.pos, 0] = done
        self.pos = (self.pos+1)%self.cap
        self.size = min(self.size+1, self.cap)

    def sample(self, n):
        idx = np.random.choice(self.size, n)
        states = self.state[idx]
        next_states = self.next_state[idx]
        actions = self.action[idx]
        rewards = self.reward[idx]
        dones = self.done[idx]
        return states, next_states, actions, rewards, dones


class Agent:
    def __init__(self, device):
        self.device = torch.device(device)
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.target_critic = Critic().to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.Q = QNet().to(self.device)
        self.opt1 = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.opt2 = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.opt3 = torch.optim.Adam(self.Q.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(10000)
        self.steps = 0

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state[None,:], device=self.device)
            mu, sigma = self.actor(state)
            z = Normal(mu, sigma).sample()
            action = torch.tanh(z).item() * 2
        return action
    
    def train(self):
        states, next_states, actions, rewards, dones = self.replay_buffer.sample(128)
        states = torch.tensor(states, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        pred_v = self.critic(states)
        pred_q = self.Q(states, actions)
        with torch.no_grad():
            target_q = rewards + 0.99 * self.target_critic(next_states) * (1 - dones)

        mu, sigma = self.actor(next_states)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z) * 2
        log_prob = dist.log_prob(z)
        pred_nq = self.Q(next_states, action)
        target_v = pred_nq - log_prob

        loss_v = F.mse_loss(pred_v, target_v.detach())
        loss_q = F.mse_loss(pred_q, target_q.detach())
        loss_pi = (log_prob * (log_prob - (pred_nq - pred_v))).mean()

        # tqdm.write(f'{loss_v.item()}')
        # tqdm.write(f'{loss_q.item()}')
        # tqdm.write(f'{loss_pi.item()}\n')

        loss = loss_v + loss_q + loss_pi
        loss.backward()
        self.opt1.step()
        self.opt2.step()
        self.opt3.step()
        self.opt1.zero_grad()
        self.opt2.zero_grad()
        self.opt3.zero_grad()
        if self.steps % 100 == 0:
            for t, v in zip(self.target_critic.parameters(), self.critic.parameters()):
                t.data.copy_(v.data)
        self.steps += 1
        return loss.item()



    
agent = Agent("cuda")
reward_hist = np.zeros(100)
reward_pos = 0
loss_hist = np.zeros(100)
loss_pos = 0
for epoch in tqdm(range(600)):
    state, _ = env.reset()
    state = state.copy()
    done = False
    steps = 0
    tot_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, trun, _ = env.step([action])
        next_state = next_state.copy()
        done = done or trun
        agent.replay_buffer.add(state, next_state, action, reward, done)
        loss = agent.train()
        state = next_state
        tot_reward += reward
        loss_hist[loss_pos] = loss
        loss_pos = (loss_pos+1)%100
        steps += 1
    reward_hist[reward_pos] = tot_reward
    reward_pos = (reward_pos+1)%100
    if epoch % 10 == 9:
        tqdm.write(f'Epoch:{epoch+1} reward:{np.mean(reward_hist):.5f} loss:{np.mean(loss_hist):.5f}')
        torch.save(agent.actor.state_dict(), 'model.pth')

        

