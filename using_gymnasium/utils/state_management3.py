import collections
import numpy as np
from scipy.stats import gamma

def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    """
    Pharmacokinetics/Pharmacodynamics insulin absorption curve.
    Models how insulin is absorbed and acts over time.
    """
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values
    return f_k, F_k

class StateRewardManager:
    """
    Manages state construction, normalization, and reward calculation for the RL agent.
    """
    def __init__(self, state_dim):
        self.glucose_history = collections.deque(maxlen=2)
        self.insulin_history = collections.deque(maxlen=160)
        self.reset()
        _, self.F_k = get_pkpd_discount_factors(
            t_peak=55, t_end=480, n_steps=160
        )
        self.running_state_mean = np.zeros(state_dim)
        self.running_state_std = np.zeros(state_dim) # Use variance for Welford's algorithm
        self.n_observations = 0

    def update_normalization_stats(self, state):
        """
        Updates running mean and variance using Welford's algorithm for numerical stability.
        """
        self.n_observations += 1
        old_mean = self.running_state_mean.copy()
        self.running_state_mean += (state - self.running_state_mean) / self.n_observations
        # Update running variance
        self.running_state_std += (state - old_mean) * (state - self.running_state_mean)

    def get_normalized_state(self, state):
        """
        Normalizes the state using the running mean and standard deviation.
        """
        self.update_normalization_stats(state)
        # Calculate standard deviation from variance, handle n=1 case
        variance = self.running_state_std / (self.n_observations - 1) if self.n_observations > 1 else np.ones_like(self.running_state_mean)
        std_dev = np.sqrt(variance)
        return (state - self.running_state_mean) / (std_dev + 1e-8)

    def calculate_iob(self):
        """
        Calculates Insulin On Board (IOB) from history and the PK/PD curve.
        """
        history = list(self.insulin_history)
        # Pad history if shorter than the PK/PD curve length
        padded_history = np.pad(history, (len(self.F_k) - len(history), 0), 'constant')
        return np.sum(np.array(padded_history[::-1]) * (1 - self.F_k))

    def get_full_state(self, observation, upcoming_carbs=0):
        """
        Constructs the full state vector from the current glucose observation.
        State vector: [glucose, rate_of_change, insulin_on_board, future_carbs]
        """
        self.glucose_history.append(observation)
        rate = (
            (self.glucose_history[1] - self.glucose_history[0]) / 5.0 # Assuming 5-minute steps
            if len(self.glucose_history) == 2
            else 0.0
        )
        iob = self.calculate_iob()
        return np.array([observation, rate, iob, upcoming_carbs])

    def get_reward_heuristic_based(self, state):
        """
        The original, well-designed heuristic reward function. Serves as the baseline.
        This is an 'engineered' solution with multiple components.
        """
        g, rate, iob, _ = state
        # ... (code is unchanged)
        target_glucose = 100
        sigma = 20.0
        proximity_reward = 10.0 * np.exp(-0.5 * ((g - target_glucose) / sigma) ** 2)
        reward = proximity_reward
        if g < 70:
            reward -= 200 * (1 + (70 - g) / 10)
        if g > 180:
            reward -= min((g - 180) * 0.2, 50)
        reward -= 0.5 * (iob ** 2)
        reward -= 0.1 * (abs(rate) ** 1.5)
        return reward

    def get_reward_risk_based(self, state):
        """
        A reward based on the negative Blood Glucose Risk Index (BGRI).
        This is a continuous, clinically-derived risk function from a 'first principles' approach.
        """
        glucose = state[0]
        if glucose <= 0:
            glucose = 1.0
        risk = 1.509 * ((np.log(glucose)**1.084) - 5.381)**2
        return -risk

    def get_reward_hybrid_based(self, state):
        """
        NEW: A hybrid reward that combines the best of both approaches.
        Uses the principled risk metric as its core, but adds scaling and shaping.
        """
        g, rate, iob, _ = state
        
        # 1. Core Component: Scaled Clinical Risk
        # We start with the BGRI risk and scale it to make the signal stronger.
        risk_reward = self.get_reward_risk_based(state)
        SCALING_FACTOR = 10.0 # Hyperparameter to tune
        reward = risk_reward * SCALING_FACTOR

        # 2. Shaping Terms: Add back the stability-promoting penalties
        # Penalty for high IOB to discourage insulin stacking
        reward -= 0.5 * (iob ** 2)

        # Penalty for high glucose volatility to encourage stability
        reward -= 0.1 * (abs(rate) ** 1.5)

        return reward

    def get_reward(self, state, reward_type='heuristic'):
        """
        Main reward function that acts as a switcher for experiments.
        This allows the main training script to select the desired reward function.
        """
        if reward_type == 'risk':
            return self.get_reward_risk_based(state)
        elif reward_type == 'hybrid':
            return self.get_reward_hybrid_based(state)
        else: # Default to your original heuristic function
            return self.get_reward_heuristic_based(state)

    def reset(self):
        """
        Resets the historical data to a safe starting point for a new episode.
        """
        self.glucose_history.clear()
        for _ in range(2):
            self.glucose_history.append(140) # Start in a safe mid-range
        self.insulin_history.clear()
        for _ in range(160):
            self.insulin_history.append(0) # Start with zero insulin history

