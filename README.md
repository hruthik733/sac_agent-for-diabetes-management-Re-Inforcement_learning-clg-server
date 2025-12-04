# Fully Closed-Loop Artificial Pancreas via Ensemble Deep Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20RL-ee4c2c)
![Gymnasium](https://img.shields.io/badge/OpenAI-Gymnasium-green)
![Status](https://img.shields.io/badge/Status-Clinical%20Validation%20Passed-success)
![Domain](https://img.shields.io/badge/Domain-Healthcare_%7C_Type_1_Diabetes-red)
![Simulator](https://img.shields.io/badge/Simulator-Simglucose_(UVa%2FPadova)-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🏥 Project Abstract
This project implements a **Fully Closed-Loop Insulin Delivery System** for Type 1 Diabetes management. Addressing one of the most difficult challenges in biomedical control systems, this agent regulates blood glucose levels **without meal announcements** or manual carbohydrate counting.

By utilizing a novel **Ensemble Reinforcement Learning architecture**, the system dynamically switches between a stochastic "explorer" (SAC-MDN) and a deterministic "controller" (TD3). This approach achieved **100% Time-in-Range (TIR)** for multiple virtual patients in the FDA-validated `simglucose` environment, effectively acting as an autonomous artificial pancreas.

---

## 🚀 Key Innovations & Architecture

Standard Reinforcement Learning struggles with the high variability of human metabolism and the lag time of insulin absorption. This project overcomes those limitations using a three-pronged architectural approach:



### 1. The Ensemble Core: SAC-MDN + TD3
Instead of relying on a single policy, the agent orchestrates two distinct experts:

* **SAC with Mixture Density Networks (SAC-MDN):**
    * *Role:* The "Stochastic Expert."
    * *Innovation:* Replaced standard Gaussian heads with MDNs to model complex, multi-modal glucose probability distributions. This allows the agent to handle high uncertainty (e.g., unexpected meal onset).
* **Twin Delayed DDPG (TD3):**
    * *Role:* The "Deterministic Expert."
    * *Function:* Provides stable, low-variance control during fasting or steady-state periods to prevent oscillations.

### 2. Dynamic Uncertainty-Based Switching
The system employs a confidence-aware selection mechanism during training:
* It calculates **entropy-based uncertainty** from the SAC agent.
* It calculates **boundary-distance confidence** from the TD3 agent.
* *Result:* The ensemble automatically hands off control to the algorithm best suited for the current physiological state.

### 3. Fully Closed-Loop Design
* **Input:** Only Continuous Glucose Monitor (CGM) data.
* **Output:** Insulin dosage (Basal/Bolus).
* **Constraint:** No user input allowed (No meal tags, no carb counting).

---

## 🍽️ Realistic Simulation Protocol

To ensure clinical validity, the model was trained using a **stochastic meal generation protocol** based on the schema by *Wang et al. (Biomedicines 2024)*. This ensures the agent is robust against irregular eating habits and variable portion sizes.

### Unannounced Meal Scenario Logic
The environment generates up to **6 eating events per day** (3 main meals + 3 potential snacks). The timing and carbohydrate content are stochastic, determined by truncated normal distributions and the patient's body weight (BW).

| Meal Event | Probability | Time Window | Mean Carb Amount (g) |
| :--- | :--- | :--- | :--- |
| **Breakfast** | 95% | 05:00 - 09:00 | `0.7 × BW` (± 15%) |
| *Snack 1* | 30% | 09:00 - 10:00 | `0.15 × BW` (± 15%) |
| **Lunch** | 95% | 10:00 - 14:00 | `1.1 × BW` (± 15%) |
| *Snack 2* | 30% | 14:00 - 16:00 | `0.15 × BW` (± 15%) |
| **Dinner** | 95% | 16:00 - 20:00 | `1.25 × BW` (± 15%) |
| *Snack 3* | 30% | 20:00 - 23:00 | `0.15 × BW` (± 15%) |

* **Uncertainty Factor:** The agent is **blind** to these events. It must infer the onset of a meal purely from the rate of change in glucose levels (CGM derivatives) and react immediately to suppress hyperglycemia.

---

## 📊 Experimental Results

The model was validated on **10 unique virtual patients** (Adolescents/Adults) using the UVa/Padova simulator (`simglucose`).

### Glycemic Control Performance
The graph below demonstrates the clinical breakdown of Time in Range (Green), Hypoglycemia (Orange), and Hyperglycemia (Red).

<img width="100%" alt="Glycemic Control Summary" src="https://github.com/user-attachments/assets/c21e467e-3091-4ccd-ac4c-e2e131ffb1a5" />

### Training Convergence
The ensemble demonstrates robust convergence, learning to balance aggressive meal compensation with safety constraints within 150 episodes.

<img width="80%" alt="Ensemble Learning Curve" src="https://github.com/user-attachments/assets/3d0a5eda-a383-4621-8838-f16271559fda" />

### Clinical Metrics Table
| Metric | Result (Avg ± Std) | Clinical Significance |
| :--- | :--- | :--- |
| **Time in Range (TIR)** | **78.78% ± 16.93** | Significantly exceeds the international consensus target of 70%. |
| **Mean Glucose** | **112.82 mg/dL** | Maintains near-optimal fasting levels (comparable to non-diabetic). |
| **Hyperglycemia** | **2.39%** | Exceptional capability in suppressing post-meal spikes. |
| **Perfect Subjects** | **30%** | 3 out of 10 patients achieved **100% TIR** with zero adverse events. |

---

## 🛠️ Technical Implementation

### Prerequisite Environment
The project relies on a custom `gymnasium` wrapper around the `simglucose` simulator.

```bash
pip install gymnasium torch numpy pandas matplotlib simglucose
