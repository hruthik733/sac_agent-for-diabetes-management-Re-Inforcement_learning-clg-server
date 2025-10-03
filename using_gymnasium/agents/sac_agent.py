# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.distributions import Normal

# # --- Soft Actor-Critic (SAC) Network Definitions ---
# class Critic(nn.Module): # Q-Network
#     def __init__(self, state_dim, action_dim, n_latent_var):
#         super(Critic, self).__init__()
#         self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
#         self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
#         self.layer_3 = nn.Linear(n_latent_var, 1)

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         x = F.relu(self.layer_1(x))
#         x = F.relu(self.layer_2(x))
#         return self.layer_3(x)

# class Actor(nn.Module): # Policy Network
#     def __init__(self, state_dim, action_dim, n_latent_var, max_action):
#         super(Actor, self).__init__()
#         self.layer_1 = nn.Linear(state_dim, n_latent_var)
#         self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
#         self.mean_layer = nn.Linear(n_latent_var, action_dim)
#         self.log_std_layer = nn.Linear(n_latent_var, action_dim)
#         self.max_action = max_action

#     def forward(self, state):
#         x = F.relu(self.layer_1(state))
#         x = F.relu(self.layer_2(x))
#         mean = self.mean_layer(x)
#         log_std = self.log_std_layer(x)
#         log_std = torch.clamp(log_std, min=-20, max=2)
#         return mean, log_std

#     def sample(self, state):
#         mean, log_std = self.forward(state)
#         std = log_std.exp()
#         normal = Normal(mean, std)
#         x_t = normal.rsample()
#         y_t = torch.tanh(x_t)
#         action = y_t * self.max_action
#         log_prob = normal.log_prob(x_t)
#         log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
#         log_prob = log_prob.sum(dim=1, keepdim=True)
#         return action, log_prob

# # --- SAC Agent Implementation ---
# class SACAgent:
#     def __init__(self, env, state_dim, action_dim, n_latent_var, lr, gamma, tau, alpha, device=None):
#         self.gamma = gamma
#         self.tau = tau
#         self.alpha = alpha
#         self.max_action = float(env.action_space.high[0])
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

#         # Networks
#         self.actor = Actor(state_dim, action_dim, n_latent_var, self.max_action).to(self.device)
#         self.critic_1 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_1_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
#         self.critic_2 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_2_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_2_target.load_state_dict(self.critic_2.state_dict())

#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
#         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
#         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

#     def select_action(self, state):
#         """
#         state: np.ndarray shape (state_dim,) or (batch, state_dim)
#         Returns: np.ndarray actions of shape (action_dim,) or (batch, action_dim)
#         """
#         if not isinstance(state, np.ndarray):
#             state = np.array(state)
#         state = torch.FloatTensor(state).to(self.device)
#         if state.ndim == 1:
#             state = state.unsqueeze(0)
#         with torch.no_grad():
#             action, _ = self.actor.sample(state)
#         action = action.detach().cpu().numpy()
#         if action.shape[0] == 1:
#             return action[0]
#         else:
#             return action

#     def update(self, replay_buffer, batch_size):
#         state, action, reward, next_state, done = replay_buffer.sample(batch_size)

#         state = torch.FloatTensor(state).to(self.device)
#         action = torch.FloatTensor(action).to(self.device)
#         reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
#         next_state = torch.FloatTensor(next_state).to(self.device)
#         done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

#         with torch.no_grad():
#             next_action, next_log_prob = self.actor.sample(next_state)
#             target_q1 = self.critic_1_target(next_state, next_action)
#             target_q2 = self.critic_2_target(next_state, next_action)
#             target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
#             target_q = reward + (1 - done) * self.gamma * target_q

#         current_q1 = self.critic_1(state, action)
#         current_q2 = self.critic_2(state, action)
        
#         critic_1_loss = F.mse_loss(current_q1, target_q)
#         critic_2_loss = F.mse_loss(current_q2, target_q)

#         self.critic_1_optimizer.zero_grad()
#         critic_1_loss.backward()
#         self.critic_1_optimizer.step()

#         self.critic_2_optimizer.zero_grad()
#         critic_2_loss.backward()
#         self.critic_2_optimizer.step()

#         # Actor update
#         new_action, log_prob = self.actor.sample(state)
#         q1_new = self.critic_1(state, new_action)
#         q2_new = self.critic_2(state, new_action)
#         actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         # Soft update
#         for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#         for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)














# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal
# import numpy as np
# import os

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, n_latent_var):
#         super(Critic, self).__init__()
#         self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
#         self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
#         self.layer_3 = nn.Linear(n_latent_var, 1)

#     def forward(self, state, action):
#         x = torch.cat([state, action], 1)
#         x = F.relu(self.layer_1(x))
#         x = F.relu(self.layer_2(x))
#         return self.layer_3(x)

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, n_latent_var, max_action):
#         super(Actor, self).__init__()
#         self.layer_1 = nn.Linear(state_dim, n_latent_var)
#         self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
#         self.mean_layer = nn.Linear(n_latent_var, action_dim)
#         self.log_std_layer = nn.Linear(n_latent_var, action_dim)
#         self.max_action = max_action

#     def forward(self, state):
#         x = F.relu(self.layer_1(state))
#         x = F.relu(self.layer_2(x))
#         mean = self.mean_layer(x)
#         log_std = self.log_std_layer(x)
#         log_std = torch.clamp(log_std, min=-20, max=2)
#         return mean, log_std

#     def sample(self, state):
#         mean, log_std = self.forward(state)
#         std = log_std.exp()
#         normal = Normal(mean, std)
#         x_t = normal.rsample()
#         y_t = torch.tanh(x_t)
#         action = y_t * self.max_action
#         log_prob = normal.log_prob(x_t)
#         log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
#         log_prob = log_prob.sum(-1, keepdim=True)
#         return action, log_prob

# class SACAgent:
#     # ! FIX: Removed 'env' and now accept 'max_action' directly
#     def __init__(self, state_dim, action_dim, max_action, n_latent_var, actor_lr, critic_lr, gamma, tau, device):
#         self.gamma = gamma
#         self.tau = tau
#         self.device = device

#         # ! FIX: Use the provided max_action value, converted to a tensor
#         self.max_action_tensor = torch.FloatTensor([max_action]).to(device)

#         # Actor Network
#         self.actor = Actor(state_dim, action_dim, n_latent_var, self.max_action_tensor).to(self.device)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

#         # Critic Networks
#         self.critic_1 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_1_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_1_target.load_state_dict(self.critic_1.state_dict())
#         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)

#         self.critic_2 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_2_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_2_target.load_state_dict(self.critic_2.state_dict())
#         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

#         # Automatic Entropy Tuning
#         self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(device)).item()
#         self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
#         self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=actor_lr)

#     @property
#     def alpha(self):
#         return self.log_alpha.exp()

#     def select_action(self, state):
#         with torch.no_grad():
#             if not isinstance(state, torch.Tensor):
#                 state = torch.FloatTensor(state).to(self.device)
#             if state.dim() == 1:
#                 state = state.unsqueeze(0)
#             action, _ = self.actor.sample(state)
#         return action.detach().cpu().numpy()

#     def update(self, replay_buffer, batch_size):
#         state, action, reward, next_state, done = replay_buffer.sample(batch_size)

#         with torch.no_grad():
#             next_action, next_log_prob = self.actor.sample(next_state)
#             target_q1 = self.critic_1_target(next_state, next_action)
#             target_q2 = self.critic_2_target(next_state, next_action)
#             target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
#             target_q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.gamma * target_q

#         current_q1 = self.critic_1(state, action)
#         current_q2 = self.critic_2(state, action)
        
#         critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

#         self.critic_1_optimizer.zero_grad()
#         self.critic_2_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_1_optimizer.step()
#         self.critic_2_optimizer.step()

#         # Update Actor and Alpha
#         new_action, log_prob = self.actor.sample(state)
#         q1_new = self.critic_1(state, new_action)
#         q2_new = self.critic_2(state, new_action)
#         actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_new, q2_new)).mean()
        
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         # Update Alpha
#         alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
#         self.alpha_optimizer.zero_grad()
#         alpha_loss.backward()
#         self.alpha_optimizer.step()

#         # Soft update target networks
#         for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#         for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
















import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

class Actor(nn.Module):
    """
    Standard SAC Actor with a single Gaussian policy.
    """
    def __init__(self, state_dim, action_dim, n_latent_var, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.layer_1 = nn.Linear(state_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.mu_head = nn.Linear(n_latent_var, action_dim)
        self.log_std_head = nn.Linear(n_latent_var, action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action

        # Calculate log probability with squashing correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, n_latent_var, lr, gamma, tau, alpha, device):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, n_latent_var, self.max_action).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor)
            return action.detach().cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_1_target(next_state, next_action), self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic_1(state, action), self.critic_2(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic_1(state, new_action), self.critic_2(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

