# -----------------------------------------------------------------------------
# 3. Prioritized Replay Buffer (utils/replay_buffer.py)
# -----------------------------------------------------------------------------
"""
Prioritized Experience Replay for better learning:
"""

import collections
import numpy as np

class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer that samples important transitions more frequently
    Important = high TD error or safety-critical states
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)
        self.alpha = alpha  # How much prioritization (0=uniform, 1=full priority)
        self.beta = beta    # Importance sampling correction
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done, priority=None):
        """
        Add transition with priority
        
        Args:
            priority: If None, use max_priority for new samples
        """
        if priority is None:
            priority = self.max_priority
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        """
        Sample batch with prioritization
        
        Returns:
            batch + importance weights for correcting bias
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*samples)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done), weights, indices)
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions (based on TD error)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)