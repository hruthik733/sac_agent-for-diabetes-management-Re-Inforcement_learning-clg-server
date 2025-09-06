# # utils/safety.py

# import numpy as np

# class SafetyLayer:
#     def __init__(self):
#         self.hypo_threshold = 80
#         self.predictive_low_threshold = 110

#     def apply(self, action, state):
#         glucose, rate_of_change, _ = state
#         if glucose < self.hypo_threshold or (glucose < self.predictive_low_threshold and rate_of_change < -1.0):
#             return np.array([0.0])
#         return action





# utils/safety.py

import numpy as np

class SafetyLayer:
    def __init__(self):
        self.hypo_threshold = 80
        self.predictive_low_threshold = 110

    def apply(self, action, state):
        # ! FIX: Unpack the first two values and ignore the rest (*)
        glucose, rate_of_change, *_ = state
        
        if glucose < self.hypo_threshold or (glucose < self.predictive_low_threshold and rate_of_change < -1.0):
            return np.array([0.0])
            
        return action