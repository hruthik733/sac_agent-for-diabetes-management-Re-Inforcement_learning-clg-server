# agents/ensemble_agent.py

import torch
import numpy as np
from agents.sac_agent_mdn import SACAgent
from agents.td3_agent import TD3Agent

class EnsembleAgent:
    """
    Ensemble agent combining SAC-MDN and TD3 algorithms.
    
    The ensemble uses a weighted voting mechanism where:
    - SAC-MDN handles exploration and stochastic policy learning
    - TD3 provides deterministic, stable policy gradients
    
    Strategy:
    1. Both agents learn from the same replay buffer
    2. Action selection uses confidence-based weighting
    3. Ensemble weights can be adjusted based on performance
    """
    
    def __init__(self, state_dim, action_dim, max_action, n_latent_var, 
                 lr, gamma, tau, alpha, policy_delay, ensemble_weights, device):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
            n_latent_var: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update parameter
            alpha: SAC entropy coefficient
            policy_delay: TD3 policy update delay
            ensemble_weights: Dict with 'sac' and 'td3' weights
            device: PyTorch device
        """
        self.device = device
        self.ensemble_weights = ensemble_weights
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Initialize SAC-MDN agent
        self.sac_agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            n_latent_var=n_latent_var,
            lr=lr,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            device=device
        )
        
        # Initialize TD3 agent
        self.td3_agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            n_latent_var=n_latent_var,
            lr=lr,
            gamma=gamma,
            tau=tau,
            policy_delay=policy_delay,
            device=device
        )
        
        self.update_counter = 0
        
        # Adaptive weight adjustment parameters
        self.sac_performance_history = []
        self.td3_performance_history = []
        self.adaptation_window = 50
        
    def select_action(self, state, evaluate=False):
        """
        Select action using ensemble strategy.
        
        Strategy:
        1. Get actions from both agents
        2. Use weighted average based on ensemble weights
        3. Add uncertainty-based selection for exploration
        
        Returns:
            action: Selected action
            selected_agent: String indicating which agent was primarily used
        """
        # Get SAC action
        sac_action = self.sac_agent.select_action(state)
        
        # Get TD3 action
        td3_action = self.td3_agent.select_action(state, add_noise=not evaluate)
        
        if evaluate:
            # During evaluation, use weighted combination
            action = (self.ensemble_weights['sac'] * sac_action + 
                     self.ensemble_weights['td3'] * td3_action)
            selected_agent = 'ensemble'
        else:
            # During training, use confidence-based selection
            # Calculate uncertainty from SAC's entropy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, log_prob = self.sac_agent.actor.sample(state_tensor)
                sac_confidence = torch.exp(-torch.abs(log_prob)).item()
            
            # TD3 is deterministic, so we use distance from bounds as confidence
            td3_confidence = 1.0 - (np.abs(td3_action).mean() / self.max_action)
            
            # Normalize confidences
            total_confidence = sac_confidence + td3_confidence
            sac_weight = sac_confidence / (total_confidence + 1e-8)
            td3_weight = td3_confidence / (total_confidence + 1e-8)
            
            # Combine actions based on dynamic confidence
            action = sac_weight * sac_action + td3_weight * td3_action
            
            # Track which agent had higher confidence
            selected_agent = 'sac' if sac_weight > td3_weight else 'td3'
        
        return action, selected_agent
    
    def update(self, replay_buffer, batch_size):
        """
        Update both agents using the shared replay buffer.
        
        Both agents learn from the same experiences but with different
        learning strategies, allowing the ensemble to capture both
        stochastic exploration (SAC) and deterministic exploitation (TD3).
        """
        # Update SAC agent
        self.sac_agent.update(replay_buffer, batch_size)
        
        # Update TD3 agent
        self.td3_agent.update(replay_buffer, batch_size)
        
        self.update_counter += 1
        
        # Optionally adapt ensemble weights based on performance
        if self.update_counter % 1000 == 0:
            self._adapt_weights()
    
    def _adapt_weights(self):
        """
        Adaptively adjust ensemble weights based on recent performance.
        This is optional and can be disabled if fixed weights work better.
        """
        # This is a placeholder for adaptive weight adjustment
        # In practice, you could track validation performance and adjust weights
        # For now, we keep weights fixed as specified in hyperparameters
        pass
    
    def save(self, filepath):
        """Save both agents' models."""
        torch.save({
            'sac_actor': self.sac_agent.actor.state_dict(),
            'sac_critic_1': self.sac_agent.critic_1.state_dict(),
            'sac_critic_2': self.sac_agent.critic_2.state_dict(),
            'td3_actor': self.td3_agent.actor.state_dict(),
            'td3_critic_1': self.td3_agent.critic_1.state_dict(),
            'td3_critic_2': self.td3_agent.critic_2.state_dict(),
            'ensemble_weights': self.ensemble_weights,
        }, filepath)
    
    def load(self, filepath, device):
        """Load both agents' models."""
        checkpoint = torch.load(filepath, map_location=device)
        
        self.sac_agent.actor.load_state_dict(checkpoint['sac_actor'])
        self.sac_agent.critic_1.load_state_dict(checkpoint['sac_critic_1'])
        self.sac_agent.critic_2.load_state_dict(checkpoint['sac_critic_2'])
        
        self.td3_agent.actor.load_state_dict(checkpoint['td3_actor'])
        self.td3_agent.critic_1.load_state_dict(checkpoint['td3_critic_1'])
        self.td3_agent.critic_2.load_state_dict(checkpoint['td3_critic_2'])
        
        self.ensemble_weights = checkpoint['ensemble_weights']
    
    def set_eval_mode(self):
        """Set both agents to evaluation mode."""
        self.sac_agent.actor.eval()
        self.td3_agent.actor.eval()
    
    def set_train_mode(self):
        """Set both agents to training mode."""
        self.sac_agent.actor.train()
        self.td3_agent.actor.train()