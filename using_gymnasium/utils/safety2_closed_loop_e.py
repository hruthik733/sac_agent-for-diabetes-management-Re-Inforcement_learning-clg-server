# -----------------------------------------------------------------------------
# 2. Enhanced Safety Layer (utils/safety2_closed_loop.py)
# -----------------------------------------------------------------------------
"""
More sophisticated safety constraints with predictive elements:
"""
import numpy as np
class EnhancedSafetyLayer:
    """
    Advanced safety layer with:
    - Predictive glucose trajectory
    - IOB-aware constraints
    - Adaptive thresholds
    """
    def __init__(self, hypo_threshold=75, predictive_low=110, hyper_threshold=170):
        self.hypo_threshold = hypo_threshold
        self.predictive_low = predictive_low
        self.hyper_threshold = hyper_threshold
        
        # Track recent actions for smoothness
        self.recent_actions = []
        self.max_action_change = 2.0  # Maximum change between timesteps
        
    def predict_glucose_trajectory(self, glucose, rate, iob, insulin_dose):
        """
        Simple linear prediction of glucose in next 30 minutes
        More sophisticated version could use learned model
        """
        # Estimate glucose drop from IOB + new insulin
        estimated_drop = (iob * 0.5 + insulin_dose * 2.0)
        
        # Linear extrapolation
        predicted_glucose = glucose + (rate * 6) - estimated_drop
        
        return predicted_glucose
    
    def apply(self, action, state):
        """
        Enhanced safety checks with predictive elements
        
        Args:
            action: proposed insulin dose
            state: [glucose, rate, iob]
        """
        glucose, rate, iob = state
        
        # === 1. Hard constraints ===
        # Never give insulin if already hypoglycemic
        if glucose < self.hypo_threshold:
            self.recent_actions.append(0.0)
            return np.array([0.0])
        
        # === 2. Predictive constraints ===
        # Predict glucose in 30 minutes
        predicted_glucose = self.predict_glucose_trajectory(
            glucose, rate, iob, action[0]
        )
        
        # Block insulin if prediction shows hypoglycemia risk
        if predicted_glucose < 70:
            self.recent_actions.append(0.0)
            return np.array([0.0])
        
        # === 3. IOB-based limiting ===
        # If IOB is high, reduce maximum allowed insulin
        if iob > 4.0:
            max_allowed = 5.0 * (1.0 - (iob - 4.0) / 10.0)
            action = np.clip(action, 0, max_allowed)
        
        # === 4. Rate-of-change constraints ===
        # Reduce insulin if glucose dropping rapidly
        if rate < -2.0 and glucose < 130:
            action = action * 0.5  # Half dose
        
        # === 5. Smoothness constraint ===
        # Limit large changes in insulin dose for stability
        if len(self.recent_actions) > 0:
            last_action = self.recent_actions[-1]
            max_change = self.max_action_change
            
            if action[0] > last_action + max_change:
                action = np.array([last_action + max_change])
            elif action[0] < last_action - max_change:
                action = np.array([last_action - max_change])
        
        # === 6. Final clipping ===
        safe_action = np.clip(action, 0, 5.0)
        
        # Update history
        self.recent_actions.append(safe_action[0])
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        
        return safe_action
    
    def reset(self):
        """Reset safety layer state"""
        self.recent_actions = []