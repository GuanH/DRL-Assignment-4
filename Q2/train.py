import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from dmc import make_dmc_env
import numpy as np
from torch.distributions import Normal
from tqdm import tqdm

env_name = "cartpole-balance"
env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)


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

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(5, 512), nn.SiLU(),
                nn.Linear(512, 512), nn.SiLU(),
                nn.Linear(512, 512), nn.SiLU(),
                nn.Linear(512, 1))
    def forward(self, x):
        return self.fc(x)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(6, 512), nn.SiLU(),
                nn.Linear(512, 512), nn.SiLU(),
                nn.Linear(512, 512), nn.SiLU(),
                nn.Linear(512, 1))

    def forward(self, s, a):
        x = torch.cat((s,a), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.size = 0
        self.state = np.empty((cap, 5), dtype=np.float32)
        self.action = np.empty((cap, 1), dtype=np.float32)
        self.reward = np.empty((cap, 1), dtype=np.float32)
        self.next_state = np.empty((cap, 5), dtype=np.float32)
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
        self.learning_critic = Critic().to(self.device)
        self.target_critic = Critic().to(self.device)
        self.target_critic.load_state_dict(self.learning_critic.state_dict())
        self.Q1 = QNet().to(self.device)
        self.Q2 = QNet().to(self.device)

        self.opt1 = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.opt2 = torch.optim.Adam(self.learning_critic.parameters(), lr=1e-4)
        self.opt3 = torch.optim.Adam(self.Q1.parameters(), lr=1e-4)
        self.opt4 = torch.optim.Adam(self.Q2.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(50000)
        self.steps = 0
        self.alpha = 0.2
        self.clip_norm = 100

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state[None,:].astype(np.float32), device=self.device)
            mu, sigma = self.actor(state)
            dist = Normal(mu, sigma)
            z = dist.sample()
            action = torch.tanh(z).item()
        return action
    
    def train(self):
        states, next_states, actions, rewards, dones = self.replay_buffer.sample(512)
        states = torch.tensor(states, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        pred_v = self.learning_critic(states)

        mu, sigma = self.actor(states)
        dist = Normal(mu, sigma)
        zs = dist.rsample()
        sample_actions = torch.tanh(zs)
        log_ps = dist.log_prob(zs) - torch.log(1 - sample_actions.pow(2) + 1e-6)

        target_v = torch.min(self.Q1(states, sample_actions), self.Q2(states, sample_actions)) - self.alpha * log_ps
        loss_v = F.mse_loss(pred_v, target_v.detach())
        self.opt2.zero_grad()
        loss_v.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.learning_critic.parameters(), self.clip_norm)
        self.opt2.step()



        pred_q1 = self.Q1(states, actions)
        pred_q2 = self.Q2(states, actions)
        with torch.no_grad():
            target_q = rewards + 0.99 * self.target_critic(next_states) * (1 - dones)
        loss_q1 = F.mse_loss(pred_q1, target_q)
        loss_q2 = F.mse_loss(pred_q2, target_q)

        self.opt3.zero_grad()
        loss_q1.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.Q1.parameters(), self.clip_norm)
        self.opt3.step()

        self.opt4.zero_grad()
        loss_q2.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.Q2.parameters(), self.clip_norm)
        self.opt4.step()


        norm_q = torch.min(self.Q1(states, sample_actions), self.Q2(states, sample_actions)) - pred_v
        loss_pi = (log_ps*(self.alpha * log_ps - norm_q).detach()).mean()
        self.opt1.zero_grad()
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_norm)
        self.opt1.step()

        self.steps += 1
        if self.steps % 1000 == 0:
            for t, v in zip(self.target_critic.parameters(), self.learning_critic.parameters()):
                t.data.copy_(v.data)
        self.clip_norm = max(self.clip_norm*0.9992, 0.5)
        return loss_v.item()+loss_q1.item()+loss_q2.item()+loss_pi.item()



    
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
        next_state, reward, done, trun, _ = env.step(action)
        next_state = next_state.copy()
        done = done or trun
        agent.replay_buffer.add(state, next_state, action, reward, done)
        steps += 1
        loss = agent.train()
        loss_hist[loss_pos] = loss
        loss_pos = (loss_pos+1)%100
        state = next_state
        tot_reward += reward
    reward_hist[reward_pos] = tot_reward
    reward_pos = (reward_pos+1)%100
    if epoch % 10 == 9:
        tqdm.write(f'Epoch:{epoch+1} reward:{np.mean(reward_hist):.5f} loss:{np.mean(loss_hist):.5f}')
        torch.save(agent.actor.state_dict(), 'model.pth')

        

