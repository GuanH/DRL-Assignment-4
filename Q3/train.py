import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from dmc import make_dmc_env
import numpy as np
from torch.distributions import Normal
from tqdm import tqdm

env_name = "humanoid-walk"
env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)


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

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(67 + 21, 2048), nn.SiLU(),
                nn.Linear(2048, 2048), nn.SiLU(),
                nn.Linear(2048, 2048), nn.SiLU(),
                nn.Linear(2048, 1))

    def forward(self, x):
        return self.fc(x)


class Critics(nn.Module):
    def __init__(self):
        super().__init__()
        self.Q1 = Critic()
        self.Q2 = Critic()

    def forward(self, s, a):
        x = torch.cat((s,a), -1)
        return self.Q1(x), self.Q2(x)

def softmax(x):
    x = np.exp(x-np.max(x))
    return x / np.sum(x)

class PER:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = max(alpha, 1e-5)
        self.data = np.empty(capacity, dtype=np.int32)
        self.delta = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def add(self, x):
        self.data[self.pos] = x
        self.delta[self.pos] = 200
        p = self.pos
        self.pos = (self.pos+1)%self.capacity
        return p

    def sample(self, n):
        d = self.delta * self.alpha
        p = d / np.sum(d)
        idx = np.random.choice(self.capacity, n, p=p)
        w = 1 / (p[idx] * n)
        return self.data[idx], idx, w

    def set_delta(self, idx, d):
        self.delta[idx] = d


class ReplayBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.size = 0
        self.state = np.empty((cap, 67), dtype=np.float32)
        self.action = np.empty((cap, 21), dtype=np.float32)
        self.reward = np.empty((cap, 1), dtype=np.float32)
        self.next_state = np.empty((cap, 67), dtype=np.float32)
        self.done = np.empty((cap, 1), dtype=np.float32)
        # self.per = PER(cap, 0.1)
        self.pos = 0

    def add(self, state, next_state, action, reward, done):
        self.state[self.pos] = state
        self.next_state[self.pos] = next_state
        self.action[self.pos] = action
        self.reward[self.pos, 0] = reward
        self.done[self.pos, 0] = done
        # p = self.per.add(self.pos)
        # if done:
        #     self.per.set_delta(p, 500)
        self.last_pos = self.pos


        self.pos = (self.pos+1)%self.cap
        self.size = min(self.size+1, self.cap)

    def sample(self, n):
        idx = np.random.choice(self.size, n)
        # idx, per_idx, w = self.per.sample(n)
        states = self.state[idx]
        next_states = self.next_state[idx]
        actions = self.action[idx]
        rewards = self.reward[idx]
        dones = self.done[idx]
        # return states, next_states, actions, rewards, dones, per_idx
        return states, next_states, actions, rewards, dones


class Agent:
    def __init__(self, device):
        self.device = torch.device(device)
        self.actor = Actor().to(self.device)
        self.learning_critic = Critics().to(self.device)
        self.target_critic = Critics().to(self.device)

        # self.actor.load_state_dict(torch.load('model.pth'))
        # self.learning_critic.load_state_dict(torch.load('model_critic.pth'))
        self.target_critic.load_state_dict(self.learning_critic.state_dict())


        self.opt_A = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.opt_C = torch.optim.Adam(self.learning_critic.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(1000000)
        self.steps = 0
        self.alpha = 0.02
        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=1e-5)
        self.clip_norm = 0.5

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state[None,:].astype(np.float32), device=self.device)
            mu, sigma = self.actor(state)
            dist = Normal(mu, sigma)
            z = dist.sample()
            action = torch.tanh(z).cpu().numpy()
        return np.clip(action[0]+np.random.normal()*0.2, -1, 1)
    
    def train(self):
        # states, next_states, actions, rewards, dones, per_idx = self.replay_buffer.sample(512)
        states, next_states, actions, rewards, dones = self.replay_buffer.sample(512)
        states = torch.tensor(states, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device)


        with torch.no_grad():
            mu, sigma = self.actor(next_states)
            dist = Normal(mu, sigma)
            zs = dist.sample()
            next_actions = torch.tanh(zs)
            next_log_ps = (dist.log_prob(zs) - torch.log(1 - next_actions.pow(2) + 1e-6)).sum(-1, keepdim=True)
            q1, q2 = self.target_critic(next_states, next_actions)
            next_q = torch.min(q1, q2)
            next_q = next_q - self.alpha * next_log_ps
            target_q = rewards + 0.99 * next_q * (1 - dones)

        pred_q1, pred_q2 = self.learning_critic(states, actions)
        delta2 = pred_q1 - target_q
        delta3 = pred_q2 - target_q
        loss_q1 = (delta2**2).mean()
        loss_q2 = (delta3**2).mean()
        loss_q = 0.5 * (loss_q1 + loss_q2)
        self.opt_C.zero_grad()
        loss_q.backward()
        self.opt_C.step()




        # self.replay_buffer.per.set_delta(per_idx, (delta1.abs() + delta2.abs() + delta3.abs()).detach().cpu().numpy().reshape(-1))

        mu, sigma = self.actor(states)
        dist = Normal(mu, sigma)
        zs = dist.rsample()
        sample_actions = torch.tanh(zs)
        log_ps = (dist.log_prob(zs) - torch.log(1 - sample_actions.pow(2) + 1e-6)).sum(-1, keepdim=True)

        q1, q2 = self.learning_critic(states, sample_actions)
        q = torch.min(q1, q2)
        loss_pi = (self.alpha * log_ps - q).mean()
        self.opt_A.zero_grad()
        loss_pi.backward()
        self.opt_A.step()


        # nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_norm)

        self.opt_alpha.zero_grad()
        loss_alpha = self.log_alpha * (-log_ps.mean().item() + 21)
        loss_alpha.backward()
        self.opt_alpha.step()
        self.alpha = np.exp(self.log_alpha.item())

        self.steps += 1
        for t, v in zip(self.target_critic.parameters(), self.learning_critic.parameters()):
            t.data.copy_(0.995*t.data + 0.005*v.data)
        # self.clip_norm = max(self.clip_norm*0.99992, 0.5)
        return loss_q.item()+loss_pi.item()



    
agent = Agent("cuda")
reward_hist = np.zeros(100)
reward_pos = 0
loss_hist = np.zeros(100)
loss_pos = 0
pbar = tqdm(range(1000000))

max_steps = 1000
for epoch in pbar:
    state, _ = env.reset()
    state = state.copy()
    done = False
    steps = 0
    tot_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, trun, _ = env.step(action)
        next_state = next_state.copy()
        tot_reward += reward
        done = done or trun
        # if steps > max_steps:
        #     done = True
        agent.replay_buffer.add(state, next_state, action, reward, done)
        steps += 1
        if steps % 64 == 0:
            loss = agent.train()
            loss_hist[loss_pos] = loss
            loss_pos = (loss_pos+1)%100
        state = next_state
    # max_steps += 0.01
    pbar.set_description(f'Reward : {tot_reward:.5f} alpha : {agent.alpha:.5f}')
    reward_hist[reward_pos] = tot_reward
    reward_pos = (reward_pos+1)%100
    if epoch % 100 == 99:
        tqdm.write(f'Epoch:{epoch+1} reward:{np.mean(reward_hist):.5f} loss:{np.mean(loss_hist):.5f} max steps:{int(max_steps)}')
        torch.save(agent.actor.state_dict(), 'model.pth')
        torch.save(agent.learning_critic.state_dict(), 'model_critic.pth')

        

