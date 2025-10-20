# agents/td3_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Actor Network for TD3 ---
class TD3Actor(nn.Module):
    """
    Deterministic actor network for TD3.
    Outputs a single deterministic action for each state.
    """
    def __init__(self, state_dim, action_dim, n_latent_var, max_action):
        super(TD3Actor, self).__init__()
        self.max_action = max_action
        
        self.layer_1 = nn.Linear(state_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, action_dim)
        
    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x))
        return x * self.max_action


# --- Critic Network for TD3 ---
class TD3Critic(nn.Module):
    """
    Critic network for TD3.
    Takes state and action as input and outputs Q-value.
    """
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(TD3Critic, self).__init__()
        
        self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)


# --- TD3 Agent ---
class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent.
    
    Key features:
    1. Twin Q-networks to reduce overestimation bias
    2. Delayed policy updates for stability
    3. Target policy smoothing for robustness
    4. Deterministic policy with exploration noise
    
    TD3 provides stable, deterministic policies that complement
    SAC's stochastic exploration in the ensemble.
    """
    
    def __init__(self, state_dim, action_dim, max_action, n_latent_var, 
                 lr, gamma, tau, policy_delay, device, 
                 policy_noise=0.2, noise_clip=0.5):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
            n_latent_var: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update parameter
            policy_delay: Delay between policy updates
            device: PyTorch device
            policy_noise: Noise added to target policy
            noise_clip: Range to clip target policy noise
        """
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.max_action = max_action
        self.device = device
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.action_dim = action_dim
        
        # Actor networks
        self.actor = TD3Actor(state_dim, action_dim, n_latent_var, max_action).to(device)
        self.actor_target = TD3Actor(state_dim, action_dim, n_latent_var, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic networks (twin Q-networks)
        self.critic_1 = TD3Critic(state_dim, action_dim, n_latent_var).to(device)
        self.critic_1_target = TD3Critic(state_dim, action_dim, n_latent_var).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        
        self.critic_2 = TD3Critic(state_dim, action_dim, n_latent_var).to(device)
        self.critic_2_target = TD3Critic(state_dim, action_dim, n_latent_var).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        
        self.total_iterations = 0
        
    def select_action(self, state, add_noise=True, noise_scale=0.1):
        """
        Select action using the deterministic policy.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            noise_scale: Scale of exploration noise
            
        Returns:
            action: Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy().flatten()
            
            if add_noise:
                # Add Gaussian noise for exploration
                noise = np.random.normal(0, noise_scale * self.max_action, size=self.action_dim)
                action = action + noise
                action = np.clip(action, -self.max_action, self.max_action)
                
        return action
    
    def update(self, replay_buffer, batch_size):
        """
        Update TD3 agent using a batch from the replay buffer.
        
        TD3 update procedure:
        1. Sample batch from replay buffer
        2. Update critics using Bellman equation with target smoothing
        3. Delayed policy update (every policy_delay steps)
        4. Soft update of target networks
        """
        self.total_iterations += 1
        
        # Sample batch from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # Move batch to device
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            
            next_action = self.actor_target(next_state)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value using minimum of twin Q-networks
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Update critics
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # Delayed policy update
        if self.total_iterations % self.policy_delay == 0:
            # Compute actor loss (policy gradient)
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)
    
    def _soft_update(self, source, target):
        """Soft update of target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )