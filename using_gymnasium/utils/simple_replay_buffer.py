# utils/simple_replay_buffer.py

import torch
import numpy as np
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        # Store experiences as NumPy arrays on the CPU
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sample a batch of experiences
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        
        # Convert the batch to tensors on the correct device
        state_tensor = torch.FloatTensor(np.array(state)).to(self.device)
        action_tensor = torch.FloatTensor(np.array(action)).to(self.device)
        reward_tensor = torch.FloatTensor(np.array(reward)).to(self.device)
        next_state_tensor = torch.FloatTensor(np.array(next_state)).to(self.device)
        done_tensor = torch.FloatTensor(np.array(done)).to(self.device)

        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor

    def __len__(self):
        return len(self.buffer)