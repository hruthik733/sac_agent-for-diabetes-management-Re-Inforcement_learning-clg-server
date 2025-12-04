# Ensemble SAC-TD3: Experimental Results

This directory contains the training logs, performance plots, and detailed metrics for the **Ensemble SAC-TD3** agent trained on 10 distinct virtual patients using the `simglucose` environment.

The objective was to evaluate the efficacy of a hybrid reinforcement learning approach (switching between Soft Actor-Critic and Twin Delayed DDPG) in maintaining blood glucose levels within the healthy range (70-180 mg/dL).

## 📊 Visual Summaries

### 1. Glycemic Control Summary
The chart below illustrates the Percentage of Time in Range (TIR), Hypoglycemia (<70 mg/dL), and Hyperglycemia (>180 mg/dL) for all 10 patients.
- **Green:** Time in Range (Target > 70%)
- **Orange:** Time Hypo
- **Red:** Time Hyper

<img width="4800" height="2700" alt="ensemble_summary_plot" src="https://github.com/user-attachments/assets/c21e467e-3091-4ccd-ac4c-e2e131ffb1a5" />


### 2. Learning Curve
The training progression shows the Cumulative Reward over 250 episodes. The shaded region represents the standard deviation, indicating the variability in learning across the ensemble.
- **Convergence:** The agent begins to stabilize and achieve positive rewards consistently around episode 150.

<img width="3600" height="1800" alt="ensemble_learning_curve" src="https://github.com/user-attachments/assets/3d0a5eda-a383-4621-8838-f16271559fda" />

---

## 📈 Quantitative Performance Metrics

### Aggregate Statistics
Across all 10 virtual patients, the Ensemble model achieved the following average performance:

| Metric | Average Value (± Std Dev) | Notes |
| :--- | :--- | :--- |
| **Time in Range (TIR)** | **78.78% ± 16.93** | *Above clinical target of 70%* |
| **Mean Glucose** | 112.82 ± 21.83 mg/dL | *Near optimal fasting levels* |
| **Time Hypo** | 18.83% ± 18.37 | *Indicates aggressive control policy* |
| **Time Hyper** | 2.39% ± 5.79 | *Excellent prevention of hyperglycemia* |

### Agent Utilization Strategy
The ensemble mechanism allows dynamic switching between SAC (stochastic) and TD3 (deterministic) policies based on uncertainty estimation.
* **Average SAC Usage:** 34.12%
* **Average TD3 Usage:** 65.88%
* *Observation:* The model relied more heavily on TD3, suggesting that for many patients, a deterministic policy provided more stable control once the initial exploration phase was complete.

---

## 📝 Detailed Patient-by-Patient Breakdown

The table below details the final evaluation metrics for each individual patient model.

| Patient ID | Mean Glucose (mg/dL) | Glucose Std | **Time in Range (%)** | Time Hypo (%) | Time Hyper (%) | SAC Usage | TD3 Usage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **adult#001** | 124.08 | 12.00 | **100.00%** 🟢 | 0.00% | 0.00% | 25.0% | 75.0% |
| **adult#002** | 141.93 | 8.90 | **100.00%** 🟢 | 0.00% | 0.00% | 38.5% | 61.5% |
| **adult#003** | 99.82 | 31.29 | 74.74% 🟢 | 25.26% | 0.00% | 25.7% | 74.3% |
| **adult#004** | 89.18 | 34.38 | 62.28% | 37.72% | 0.00% | 36.8% | 63.2% |
| **adult#005** | 98.19 | 44.09 | 67.80% | 32.20% | 0.00% | 37.9% | 62.1% |
| **adult#006** | 134.12 | 43.77 | 82.01% 🟢 | 0.00% | 17.99% | 33.0% | 67.0% |
| **adult#007** | 119.43 | 41.64 | 82.01% 🟢 | 12.11% | 5.88% | 51.0% | 49.0% |
| **adult#008** | 141.53 | 7.90 | **100.00%** 🟢 | 0.00% | 0.00% | 51.4% | 48.6% |
| **adult#009** | 88.21 | 44.40 | 54.26% | 45.74% | 0.00% | 26.9% | 73.1% |
| **adult#010** | 91.69 | 39.01 | 64.71% | 35.29% | 0.00% | 14.9% | 85.1% |

### Key Observations
1.  **Perfect Control:** Patients `adult#001`, `adult#002`, and `adult#008` achieved **100% Time in Range** with zero adverse events.
2.  **Hypoglycemia Challenges:** Patients `adult#004`, `adult#009`, and `adult#010` experienced significant hypoglycemia (Time Hypo > 30%). This suggests the agent may be over-correcting for glucose spikes in these specific metabolic profiles.
3.  **Usage Correlation:** In high-performing cases (e.g., `adult#008`), the split between SAC and TD3 was nearly even (51% vs 48%), whereas in some lower-performing cases (e.g., `adult#010`), the model leaned heavily on TD3 (85%).

---

## 📂 File Structure
- `ensemble_summary_plot.png`: Bar chart of TIR metrics.
- `ensemble_learning_curve.png`: Training reward progression.
- `ensemble_summary.csv`: Raw CSV data of the metrics table above.
