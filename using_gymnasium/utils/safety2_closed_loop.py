# utils/safety2.py

import numpy as np

class SafetyLayer:
    """
    Safety layer enforcing:
    - No insulin if glucose below hypo_threshold.
    - No insulin if glucose below predictive_low_threshold AND dropping rapidly.
    - Optionally scale down insulin for moderate lows with rapid drop.
    - Insulin allowed (clipped) only if glucose sufficiently high or rising.

    Parameters:
    - hypo_threshold: hard hypoglycemia cutoff (mg/dL)
    - predictive_low_threshold: warning margin above hypoglycemia (mg/dL)
    - hyper_threshold: lower bound to allow full insulin action (mg/dL)
    """

    def __init__(self, hypo_threshold=80, predictive_low_threshold=110, hyper_threshold=170):
        self.hypo_threshold = hypo_threshold
        self.predictive_low_threshold = predictive_low_threshold
        self.hyper_threshold = hyper_threshold

    def apply(self, action, state):
        # MODIFIED: Unpack the 3-dimensional state [glucose, rate, iob].
        # The underscore '_' is used to indicate that 'iob' is unpacked but intentionally not used in this logic.
        glucose, rate_of_change, _ = state

        # Hard no insulin if glucose below hypo threshold
        if glucose < self.hypo_threshold:
            return np.array([0.0])

        # No insulin if rapidly dropping glucose nearing low safe threshold
        if glucose < self.predictive_low_threshold and rate_of_change < -1.0:
            return np.array([0.0])

        # Optional: reduce insulin if moderately low glucose and sharply dropping
        if glucose < 130 and rate_of_change < -2.0:
            reduced_action = 0.5 * action
            return np.clip(reduced_action, 0, 5.0)

        # Hyperglycemia range: allow agent action but clip
        if glucose > self.hyper_threshold or (rate_of_change > 0.5 and glucose > 150):
            return np.clip(action, 0, 5.0)

        # Otherwise, allow agent action clipped
        return np.clip(action, 0, 5.0)


# # utils/safety2_closed_loop.py lam code

# import numpy as np

# class SafetyLayer:
#     """
#     A more proactive safety layer that scales insulin dosage based on glucose levels.

#     - No insulin if hypoglycemic or dropping fast towards hypoglycemia.
#     - Aggressively scales down insulin in the lower-normal range.
#     - Moderately scales down insulin in the upper-normal range.
#     - Allows the agent's full proposed action only when hyperglycemic.
#     """
#     def __init__(self,
#                  hypo_threshold=80.0,
#                  caution_threshold=120.0,
#                  normal_threshold=180.0,
#                  max_allowed_dose=5.0):
        
#         self.hypo_threshold = hypo_threshold
#         self.caution_threshold = caution_threshold
#         self.normal_threshold = normal_threshold
#         self.max_allowed_dose = max_allowed_dose

#     def apply(self, proposed_action, state):
#         """
#         Adjusts the proposed insulin dose based on safety rules.

#         Args:
#             proposed_action (float): The insulin dose proposed by the RL agent.
#             state (np.array): The current state [glucose, rate, iob].

#         Returns:
#             np.array: A single-element array with the safe, adjusted insulin dose.
#         """
#         glucose, rate_of_change, _ = state
        
#         # Rule 1: Hard safety stop for hypoglycemia or predictive hypoglycemia.
#         # If glucose is already low OR it's below caution and dropping fast.
#         if glucose < self.hypo_threshold or (glucose < 100 and rate_of_change < -2.0):
#             return np.array([0.0])

#         # Rule 2: Aggressive reduction in the "caution" zone (e.g., 80-120 mg/dL).
#         # We scale the action by a factor that is 0 at the hypo_threshold and
#         # increases linearly to 0.5 at the caution_threshold.
#         if self.hypo_threshold <= glucose < self.caution_threshold:
#             scaling_factor = 0.5 * (glucose - self.hypo_threshold) / (self.caution_threshold - self.hypo_threshold)
#             action = proposed_action * scaling_factor
        
#         # Rule 3: Moderate reduction in the "normal" zone (e.g., 120-180 mg/dL).
#         # Here we allow a bit more insulin, scaling from 0.5 up to 1.0.
#         elif self.caution_threshold <= glucose < self.normal_threshold:
#             # We start at 0.5 and scale up to 1.0
#             base_scaling = 0.5
#             additional_scaling = 0.5 * (glucose - self.caution_threshold) / (self.normal_threshold - self.caution_threshold)
#             action = proposed_action * (base_scaling + additional_scaling)
            
#         # Rule 4: Allow full agent action only when clearly hyperglycemic.
#         else: # glucose >= self.normal_threshold
#             action = proposed_action
        
#         # Final safeguard: Always clip the final action to be non-negative and
#         # below the absolute maximum allowed dose for the pump.
#         return np.clip(np.array([action]), 0.0, self.max_allowed_dose)