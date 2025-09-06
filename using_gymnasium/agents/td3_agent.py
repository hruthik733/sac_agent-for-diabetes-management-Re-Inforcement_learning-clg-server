# agents/td3_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Standard Feed-Forward Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        # Use Tanh to output action in [-1, 1] range, then scale by max_action
        return self.max_action * torch.tanh(self.layer_3(x))

# Standard Feed-Forward Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Critic, self).__init__()
        # Critic 1
        self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, 1)
        # Critic 2
        self.layer_4 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.layer_5 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_6 = nn.Linear(n_latent_var, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)

        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        return q1

# TD3 Agent Logic (This part is stable and correct)
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, n_latent_var, lr, gamma, tau, policy_noise, noise_clip, policy_freq, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, n_latent_var, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, n_latent_var, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim, n_latent_var).to(device)
        self.critic_target = Critic(state_dim, action_dim, n_latent_var).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.actor(state).cpu().data.numpy()

    def update(self, replay_buffer, batch_size):
        self.total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(0, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath):
        torch.save(self.actor.state_dict(), filepath)

    def load(self, filepath):
        self.actor.load_state_dict(torch.load(filepath, map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())