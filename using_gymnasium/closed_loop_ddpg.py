# import gymnasium
# from gymnasium.envs.registration import register
# import numpy as np
# import torch
# from datetime import datetime
# import matplotlib.pyplot as plt
# import os
# import random
# import pandas as pd
# import multiprocessing as mp

# # --- Local Imports ---
# from agents.ddpg_agent import DDPGAgent  # <<< CHANGED
# from utils.replay_buffer import ReplayBuffer
# from utils.state_management_closed_loop import StateRewardManager
# from utils.safety2_closed_loop import SafetyLayer
# import simglucose.simulation.scenario_gen as scgen
# from simglucose.simulation.scenario import CustomScenario

# def train_and_evaluate_patient(patient_name, gpu_id, seed, hyperparameters):
#     """
#     This function contains the entire training and evaluation loop for a single patient
#     using the DDPG agent in a closed-loop setting.
#     """
#     # 1. SETUP AND CONFIGURATION
#     # =================================================================
#     device = torch.device(f"cuda:{gpu_id}")
#     print(f"Process for {patient_name} starting on device: {device}")

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.set_device(device)
#     torch.cuda.manual_seed_all(seed)

#     # Unpack hyperparameters
#     max_episodes = hyperparameters['max_episodes']
#     lr = hyperparameters['lr']
#     gamma_val = hyperparameters['gamma_val']
#     tau = hyperparameters['tau']
#     batch_size = hyperparameters['batch_size']
#     replay_buffer_size = hyperparameters['replay_buffer_size']
#     max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
#     learning_starts = hyperparameters['learning_starts']
#     ETA = hyperparameters['ETA']
#     expl_noise = hyperparameters['expl_noise'] # <<< DDPG specific

#     AGENT_NAME = 'ddpg_personalized_exp_closed_loop' # <<< CHANGED
#     model_dir = f'./models/{AGENT_NAME}'
#     results_dir = f'./results/{AGENT_NAME}'
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(results_dir, exist_ok=True)
#     actor_path = f'{model_dir}/actor_{patient_name.replace("#", "-")}.pth'
    
#     now = datetime.now()
#     start_time = datetime.combine(now.date(), datetime.min.time())

#     meal_scenario = scgen.RandomScenario(start_time=start_time, seed=seed)
#     clean_patient_name = patient_name.replace('#', '-')
#     env_id = f'simglucose/ddpg-{clean_patient_name}-v0' # <<< CHANGED for unique registration
    
#     try:
#         register(
#             id=env_id,
#             entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
#             max_episode_steps=max_timesteps_per_episode,
#             kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario}
#         )
#     except gymnasium.error.Error:
#         pass

#     env = gymnasium.make(env_id)
#     env.action_space.seed(seed)
    
#     state_dim = 3
#     action_dim = 1
#     i_max = float(env.action_space.high[0])
#     max_action = 1.0

#     agent = DDPGAgent(  # <<< CHANGED
#         state_dim=state_dim, action_dim=action_dim, max_action=max_action,
#         lr=lr, gamma=gamma_val, tau=tau, device=device
#     )

#     manager = StateRewardManager(state_dim)
#     safety_layer = SafetyLayer()
#     replay_buffer = ReplayBuffer(replay_buffer_size)

#     # 2. TRAINING LOOP
#     # =================================================================
#     print(f"--- Starting DDPG Training for Patient: {patient_name} on {device} ---")
#     total_timesteps_taken = 0
#     training_rewards_history = []

#     for i_episode in range(1, max_episodes + 1):
#         obs_array, info = env.reset(seed=seed + i_episode)
#         manager.reset()
#         unnormalized_state = manager.get_full_state(obs_array[0])
#         current_state = manager.get_normalized_state(unnormalized_state)
#         episode_reward = 0

#         for t in range(max_timesteps_per_episode):
#             if total_timesteps_taken < learning_starts:
#                 raw_action = np.random.uniform(low=-max_action, high=max_action, size=(action_dim,))
#             else:
#                 # <<< CHANGED: DDPG is deterministic, so add noise for exploration
#                 action = agent.select_action(current_state)
#                 noise = np.random.normal(0, max_action * expl_noise, size=action_dim)
#                 raw_action = (action + noise).clip(-max_action, max_action)

#             insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
#             safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
            
#             manager.insulin_history.append(safe_action[0])
#             next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
#             done = terminated or truncated

#             next_unnormalized_state = manager.get_full_state(next_obs_array[0])
#             next_state = manager.get_normalized_state(next_unnormalized_state)
#             reward = manager.get_reward(unnormalized_state)
            
#             replay_buffer.push(current_state, raw_action, reward, next_state, done)
            
#             current_state = next_state
#             unnormalized_state = next_unnormalized_state
#             episode_reward += reward
#             total_timesteps_taken += 1

#             if total_timesteps_taken > learning_starts and len(replay_buffer) > batch_size:
#                 agent.update(replay_buffer, batch_size)

#             if done:
#                 break
        
#         training_rewards_history.append(episode_reward)
#         if i_episode % 50 == 0:
#             print(f"[{patient_name} on {device}] Episode {i_episode}/{max_episodes} | Reward: {episode_reward:.2f}")
    
#     env.close()
#     print(f"--- Training Finished for {patient_name} ---")
#     torch.save(agent.actor.state_dict(), actor_path)
#     print(f"Saved trained model to {actor_path}")

#     # Plot individual learning curve
#     plt.figure(figsize=(12, 6))
#     plt.plot(training_rewards_history, label='Episode Reward', alpha=0.6)
#     moving_avg = pd.Series(training_rewards_history).rolling(window=20, min_periods=1).mean()
#     plt.plot(moving_avg, label='Moving Average (20 episodes)', color='red', linewidth=2)
#     plt.title(f'Learning Curve for {patient_name} (DDPG)')
#     plt.xlabel('Episode')
#     plt.ylabel('Cumulative Reward')
#     plt.legend()
#     plt.grid(True)
#     learning_curve_path = f'{results_dir}/learning_curve_{clean_patient_name}.png'
#     plt.savefig(learning_curve_path)
#     plt.close()
#     print(f"Saved learning curve plot to {learning_curve_path}")

#     # 3. EVALUATION LOOP
#     # =================================================================
#     print(f"\n--- Starting Evaluation for {patient_name} ---")
#     meal_times = [7 * 60, 12 * 60, 18 * 60]
#     meal_carbs = [45, 70, 80]
#     eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
    
#     try:
#         eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
#     except gymnasium.error.Error:
#         eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
    
#     eval_agent = DDPGAgent( # <<< CHANGED
#         state_dim=state_dim, action_dim=action_dim, max_action=max_action,
#         lr=lr, gamma=gamma_val, tau=tau, device=device
#     )
#     eval_agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
#     eval_agent.actor.eval()

#     manager.reset()
#     obs_array, info = eval_env.reset()
    
#     unnormalized_state = manager.get_full_state(obs_array[0])
#     current_state = manager.get_normalized_state(unnormalized_state)
    
#     glucose_history = [obs_array[0]]
#     eval_insulin_history = []

#     for t in range(max_timesteps_per_episode):
#         with torch.no_grad():
#             raw_action = eval_agent.select_action(current_state) # DDPG is deterministic, no noise in eval

#         insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
#         safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
        
#         eval_insulin_history.append(safe_action[0])
#         obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)

#         unnormalized_state = manager.get_full_state(obs_array[0])
#         current_state = manager.get_normalized_state(unnormalized_state)
#         glucose_history.append(obs_array[0])

#         if terminated or truncated:
#             break
            
#     eval_env.close()

#     # Plot evaluation results
#     time_axis_minutes = np.arange(len(glucose_history)) * 5
#     if len(eval_insulin_history) != len(glucose_history):
#        eval_insulin_history.append(0)

#     fig, ax1 = plt.subplots(figsize=(15, 7))
#     color = 'tab:blue'
#     ax1.set_xlabel('Time (minutes)', fontsize=12)
#     ax1.set_ylabel('Blood Glucose (mg/dL)', color=color, fontsize=12)
#     ax1.plot(time_axis_minutes, glucose_history, color=color, label='Blood Glucose')
#     ax1.tick_params(axis='y', labelcolor=color)
#     ax1.grid(which='major', axis='y', linestyle='--', alpha=0.7)
#     ax1.axhline(y=180, color='r', linestyle=':', label='Hyper Threshold (180)')
#     ax1.axhline(y=70, color='orange', linestyle=':', label='Hypo Threshold (70)')
#     ax2 = ax1.twinx()
#     color = 'tab:gray'
#     ax2.set_ylabel('Insulin Dose (U / 5min)', color=color, fontsize=12)
#     ax2.bar(time_axis_minutes, eval_insulin_history, width=5, color=color, alpha=0.6, label='Insulin Dose')
#     ax2.tick_params(axis='y', labelcolor=color)
#     meal_labels_seen = set()
#     for meal_time, meal_amount in zip(meal_times, meal_carbs):
#         label = f'Meal ({meal_amount}g)'
#         if label not in meal_labels_seen:
#             ax1.axvline(x=meal_time, color='black', linestyle='--', label=label)
#             meal_labels_seen.add(label)
#         else:
#             ax1.axvline(x=meal_time, color='black', linestyle='--')
#     fig.suptitle(f'Closed-Loop Performance for {patient_name}', fontsize=16, weight='bold')
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     h1, l1 = ax1.get_legend_handles_labels()
#     h2, l2 = ax2.get_legend_handles_labels()
#     ax1.legend(h1 + h2, l1 + l2, loc='upper left')
#     plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
#     plt.savefig(plot_path)
#     plt.close()
#     print(f"Saved evaluation plot to {plot_path}")
    
#     # Calculate evaluation metrics
#     glucose_history = np.array(glucose_history)
#     time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
#     time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
#     time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
#     mean_glucose = np.mean(glucose_history)

#     return {
#         "eval_metrics": {
#             "Patient": patient_name,
#             "Mean Glucose (mg/dL)": mean_glucose,
#             "Time in Range (%)": time_in_range,
#             "Time Hypo (%)": time_hypo,
#             "Time Hyper (%)": time_hyper,
#         },
#         "training_rewards": training_rewards_history
#     }

# if __name__ == '__main__':
#     try:
#         mp.set_start_method('spawn')
#     except RuntimeError:
#         pass

#     # <<< CHANGED HYPERPARAMETERS
#     hyperparameters = {
#         'max_episodes': 200, 
#         'lr': 1e-4,  # DDPG often benefits from a slightly lower learning rate
#         'gamma_val': 0.99, 
#         'tau': 0.005,
#         'batch_size': 256, 
#         'replay_buffer_size': 1000000,
#         'max_timesteps_per_episode': 288, 
#         'learning_starts': 1500, 
#         'ETA': 4.0,
#         'expl_noise': 0.1 # DDPG-specific exploration noise
#     }
    
#     results_dir = f'./results/ddpg_personalized_exp_closed_loop' # <<< CHANGED
#     os.makedirs(results_dir, exist_ok=True)

#     adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
#     tasks = []
#     num_gpus = torch.cuda.device_count()
#     if num_gpus == 0:
#         raise ValueError("This script requires at least one GPU.")

#     for i, patient_name in enumerate(adult_patients):
#         gpu_id = i % num_gpus
#         tasks.append((patient_name, gpu_id, 42 + i, hyperparameters))

#     print("\n" + "="*50)
#     print(f"Starting parallel DDPG training for {len(tasks)} patients on {num_gpus} GPUs...")
#     print("="*50)
    
#     num_processes = min(len(tasks), mp.cpu_count()) 
#     with mp.Pool(processes=num_processes) as pool:
#         results = pool.starmap(train_and_evaluate_patient, tasks)

#     print("\n" + "="*50)
#     print("--- ALL DDPG TRAINING PROCESSES COMPLETE ---")
#     print("="*50)

#     # 4. AGGREGATE, DISPLAY, AND PLOT FINAL RESULTS
#     # =================================================================
#     if results:
#         eval_results = [res['eval_metrics'] for res in results]
#         training_rewards_list = [res['training_rewards'] for res in results]

#         results_df = pd.DataFrame(eval_results)
#         column_order = [
#             "Patient", "Mean Glucose (mg/dL)", "Time in Range (%)",
#             "Time Hypo (%)", "Time Hyper (%)"
#         ]
#         results_df = results_df[column_order]
        
#         # Calculate the mean across all patients
#         average_metrics = results_df.select_dtypes(include=np.number).mean()
        
#         # <<< ADD THIS LINE to calculate the standard deviation
#         std_dev_metrics = results_df.select_dtypes(include=np.number).std()

#         print("\n" + "="*70)
#         print("--- OVERALL EVALUATION SUMMARY (CLOSED-LOOP) ---")
#         print("="*70)
#         print("\n--- Performance per Patient ---")
#         print(results_df.to_string(index=False))
        
#         # <<< MODIFIED THIS BLOCK to print in "mean ± std" format
#         print("\n\n--- Average Performance Across All Patients ---")
#         print(f"Mean Glucose (mg/dL): {average_metrics['Mean Glucose (mg/dL)']:.2f} ± {std_dev_metrics['Mean Glucose (mg/dL)']:.2f}")
#         print(f"Time in Range (%):    {average_metrics['Time in Range (%)']:.2f} ± {std_dev_metrics['Time in Range (%)']:.2f}")
#         print(f"Time Hypo (%):        {average_metrics['Time Hypo (%)']:.2f} ± {std_dev_metrics['Time Hypo (%)']:.2f}")
#         print(f"Time Hyper (%):       {average_metrics['Time Hyper (%)']:.2f} ± {std_dev_metrics['Time Hyper (%)']:.2f}")
        
#         summary_path = os.path.join(results_dir, 'overall_summary.csv')
#         results_df.to_csv(summary_path, index=False, float_format='%.2f')
#         print(f"\n✅ Saved overall summary table to {summary_path}")
#         print("\n" + "="*70)

#         # --- Display Overall Summary Plot (Unchanged) ---
#         print("\n--- Generating Overall Summary Plot ---")
#         try:
#             plot_data = results_df.set_index('Patient')
#             time_metrics_df = plot_data[['Time in Range (%)', 'Time Hypo (%)', 'Time Hyper (%)']]
#             ax = time_metrics_df.plot(
#                 kind='bar', stacked=True, figsize=(16, 9),
#                 color={'Time in Range (%)': 'green', 'Time Hypo (%)': 'orange', 'Time Hyper (%)': 'red'},
#                 edgecolor='black', width=0.8
#             )
#             plt.title('Glycemic Control Summary (Closed-Loop)', fontsize=18, weight='bold')
#             plt.ylabel('Percentage of Time (%)', fontsize=14)
#             plt.xlabel('Patient ID', fontsize=14)
#             plt.xticks(rotation=45, ha='right', fontsize=12)
#             plt.yticks(fontsize=12)
#             plt.ylim(0, 100)
#             plt.grid(axis='y', linestyle='--', alpha=0.7)
#             plt.axhline(y=70, color='blue', linestyle='--', label='70% Time in Range Target')
#             plt.legend(title='Glycemic Range', bbox_to_anchor=(1.02, 1), loc='upper left')
#             plt.tight_layout()
#             summary_plot_path = os.path.join(results_dir, 'overall_summary_plot.png')
#             plt.savefig(summary_plot_path)
#             plt.close()
#             print(f"✅ Saved overall summary plot to {summary_plot_path}")
#         except Exception as e:
#             print(f"\n⚠️ Could not generate summary plot. Error: {e}")

#         # --- Plot Overall Average Learning Curve (Unchanged) ---
#         print("\n--- Generating Overall Average Learning Curve ---")
#         try:
#             rewards_df = pd.DataFrame(training_rewards_list).T
#             mean_rewards = rewards_df.mean(axis=1)
#             std_rewards = rewards_df.std(axis=1)

#             plt.figure(figsize=(12, 6))
#             plt.plot(mean_rewards, label='Mean Episode Reward', color='blue')
#             plt.fill_between(
#                 rewards_df.index,
#                 mean_rewards - std_rewards,
#                 mean_rewards + std_rewards,
#                 color='blue',
#                 alpha=0.2,
#                 label='Standard Deviation'
#             )
#             plt.title('Overall Average Learning Curve (Closed-Loop)')
#             plt.xlabel('Episode')
#             plt.ylabel('Cumulative Reward')
#             plt.legend()
#             plt.grid(True)
#             overall_lc_path = os.path.join(results_dir, 'overall_learning_curve.png')
#             plt.savefig(overall_lc_path)
#             plt.close()
#             print(f"✅ Saved overall average learning curve to {overall_lc_path}")
#         except Exception as e:
#             print(f"\n⚠️ Could not generate overall learning curve. Error: {e}")

#     else:
#         print("\n⚠️ No results were generated to summarize.")









import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import multiprocessing as mp

# --- Local Imports ---
from agents.ddpg_agent import DDPGAgent
from utils.replay_buffer import ReplayBuffer
from utils.state_management_closed_loop import StateRewardManager
from utils.safety2_closed_loop import SafetyLayer
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario

def plot_cvga(df, results_dir, agent_name):
    """
    Generates and saves a Control Variability Grid Analysis (CVGA) plot.
    """
    print(f"\n--- Generating CVGA Plot for {agent_name} ---")
    plt.figure(figsize=(8, 8))
    
    plt.scatter(df['Min Glucose'], df['Max Glucose'], c='blue', alpha=0.7, zorder=3)

    plt.plot([0, 70], [180, 180], 'r-', lw=2)
    plt.plot([70, 70], [180, 500], 'r-', lw=2)
    plt.plot([0, 110], [400, 400], 'r-', lw=2, linestyle='--')
    plt.plot([110, 110], [400, 500], 'r-', lw=2, linestyle='--')
    plt.plot([30, 30], [0, 500], 'r-', lw=2, linestyle='--')
    plt.plot([0, 400], [40, 40], 'r-', lw=2, linestyle='--')
    
    plt.fill_between([70, 180], [70, 70], [180, 180], color='limegreen', alpha=0.3, zorder=1) # A Zone
    plt.fill_between([70, 180], [180, 180], [400, 400], color='gold', alpha=0.3, zorder=1) # Upper B
    plt.fill_between([30, 70], [70, 70], [180, 180], color='gold', alpha=0.3, zorder=1) # Lower B

    plt.xlabel("Minimum Glucose (mg/dL)")
    plt.ylabel("Maximum Glucose (mg/dL)")
    plt.title(f"Control Variability Grid Analysis (CVGA) - {agent_name}")
    plt.xlim(0, 400)
    plt.ylim(0, 500)
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    plt.text(125, 125, 'A', fontsize=20, ha='center')
    plt.text(125, 250, 'B (Upper)', fontsize=16, ha='center')
    plt.text(50, 125, 'B (Lower)', fontsize=16, ha='center')
    plt.text(250, 450, 'D (Upper)', fontsize=16, ha='center')
    plt.text(40, 300, 'C (Lower)', fontsize=16, ha='center')
    
    cvga_path = os.path.join(results_dir, 'overall_cvga_plot.png')
    plt.savefig(cvga_path)
    plt.close()
    print(f"✅ Saved CVGA plot to {cvga_path}")

def plot_agp(all_glucose_data, results_dir, agent_name):
    """
    Generates and saves an Ambulatory Glucose Profile (AGP) plot.
    """
    print(f"\n--- Generating AGP Plot for {agent_name} ---")
    if all_glucose_data.shape[0] < 2:
        print("⚠️ Not enough data for AGP plot.")
        return

    percentiles = np.percentile(all_glucose_data, [10, 25, 50, 75, 90], axis=0)
    p10, p25, p50, p75, p90 = percentiles
    
    time_axis = np.arange(all_glucose_data.shape[1]) * 5 / 60 # Time in hours

    plt.figure(figsize=(12, 7))
    
    plt.plot(time_axis, p50, color='blue', lw=2, label='Median')
    plt.fill_between(time_axis, p25, p75, color='blue', alpha=0.3, label='IQR (25-75%)')
    plt.fill_between(time_axis, p10, p90, color='blue', alpha=0.15, label='10-90% Range')
    
    plt.axhline(180, color='red', linestyle='--', label='Hyperglycemia (>180 mg/dL)')
    plt.axhline(70, color='orange', linestyle='--', label='Hypoglycemia (<70 mg/dL)')
    
    plt.xlabel("Time of Day (Hours)")
    plt.ylabel("Blood Glucose (mg/dL)")
    plt.title(f"Ambulatory Glucose Profile (AGP) - {agent_name}")
    plt.xlim(0, 24)
    plt.ylim(0, 400)
    plt.xticks(np.arange(0, 25, 3))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    agp_path = os.path.join(results_dir, 'overall_agp_plot.png')
    plt.savefig(agp_path)
    plt.close()
    print(f"✅ Saved AGP plot to {agp_path}")

def train_and_evaluate_patient(patient_name, gpu_id, seed, hyperparameters):
    # 1. SETUP AND CONFIGURATION
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Process for {patient_name} starting on device: {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)

    max_episodes = hyperparameters['max_episodes']
    lr = hyperparameters['lr']
    gamma_val = hyperparameters['gamma_val']
    tau = hyperparameters['tau']
    batch_size = hyperparameters['batch_size']
    replay_buffer_size = hyperparameters['replay_buffer_size']
    max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
    learning_starts = hyperparameters['learning_starts']
    ETA = hyperparameters['ETA']
    expl_noise = hyperparameters['expl_noise']

    AGENT_NAME = 'ddpg_personalized_exp_closed_loop'
    model_dir = f'./models/{AGENT_NAME}'
    results_dir = f'./results/{AGENT_NAME}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    actor_path = f'{model_dir}/actor_{patient_name.replace("#", "-")}.pth'
    
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=seed)
    clean_patient_name = patient_name.replace('#', '-')
    env_id = f'simglucose/ddpg-{clean_patient_name}-v0'
    
    try:
        register(id=env_id, entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
                 max_episode_steps=max_timesteps_per_episode,
                 kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario})
    except gymnasium.error.Error:
        pass

    env = gymnasium.make(env_id)
    env.action_space.seed(seed)
    
    state_dim = 3
    action_dim = 1
    i_max = float(env.action_space.high[0])
    max_action = 1.0

    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                      lr=lr, gamma=gamma_val, tau=tau, device=device)

    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # 2. TRAINING LOOP
    print(f"--- Starting DDPG Training for Patient: {patient_name} on {device} ---")
    total_timesteps_taken = 0
    training_rewards_history = []

    for i_episode in range(1, max_episodes + 1):
        obs_array, info = env.reset(seed=seed + i_episode)
        manager.reset()
        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        episode_reward = 0

        for t in range(max_timesteps_per_episode):
            if total_timesteps_taken < learning_starts:
                raw_action = np.random.uniform(low=-max_action, high=max_action, size=(action_dim,))
            else:
                action = agent.select_action(current_state)
                noise = np.random.normal(0, max_action * expl_noise, size=action_dim)
                raw_action = (action + noise).clip(-max_action, max_action)

            insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
            safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
            
            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated

            next_unnormalized_state = manager.get_full_state(next_obs_array[0])
            next_state = manager.get_normalized_state(next_unnormalized_state)
            reward = manager.get_reward(unnormalized_state)
            
            replay_buffer.push(current_state, raw_action, reward, next_state, done)
            current_state = next_state
            unnormalized_state = next_unnormalized_state
            episode_reward += reward
            total_timesteps_taken += 1

            if total_timesteps_taken > learning_starts and len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)

            if done:
                break
        
        training_rewards_history.append(episode_reward)
        if i_episode % 50 == 0:
            print(f"[{patient_name} on {device}] Episode {i_episode}/{max_episodes} | Reward: {episode_reward:.2f}")
    
    env.close()
    print(f"--- Training Finished for {patient_name} ---")
    torch.save(agent.actor.state_dict(), actor_path)
    print(f"Saved trained model to {actor_path}")

    # Plot individual learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(training_rewards_history, label='Episode Reward', alpha=0.6)
    moving_avg = pd.Series(training_rewards_history).rolling(window=20, min_periods=1).mean()
    plt.plot(moving_avg, label='Moving Average (20 episodes)', color='red', linewidth=2)
    plt.title(f'Learning Curve for {patient_name} (DDPG)')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    learning_curve_path = f'{results_dir}/learning_curve_{clean_patient_name}.png'
    plt.savefig(learning_curve_path)
    plt.close()
    print(f"Saved learning curve plot to {learning_curve_path}")

    # 3. EVALUATION LOOP
    print(f"\n--- Starting Evaluation for {patient_name} ---")
    meal_times = [7 * 60, 12 * 60, 18 * 60]
    meal_carbs = [45, 70, 80]
    eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
    
    try:
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
    except gymnasium.error.Error:
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
    
    eval_agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                      lr=lr, gamma=gamma_val, tau=tau, device=device)
    eval_agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    eval_agent.actor.eval()

    manager.reset()
    obs_array, info = eval_env.reset()
    unnormalized_state = manager.get_full_state(obs_array[0])
    current_state = manager.get_normalized_state(unnormalized_state)
    glucose_history = [obs_array[0]]
    eval_insulin_history = []

    for t in range(max_timesteps_per_episode):
        with torch.no_grad():
            raw_action = eval_agent.select_action(current_state)

        insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
        safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
        eval_insulin_history.append(safe_action[0])
        obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)
        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history.append(obs_array[0])
        if terminated or truncated:
            break
            
    eval_env.close()

    # Plot evaluation results
    time_axis_minutes = np.arange(len(glucose_history)) * 5
    if len(eval_insulin_history) != len(glucose_history):
       eval_insulin_history.append(0)

    fig, ax1 = plt.subplots(figsize=(15, 7))
    color = 'tab:blue'
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Blood Glucose (mg/dL)', color=color)
    ax1.plot(time_axis_minutes, glucose_history, color=color, label='Blood Glucose')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(which='major', axis='y', linestyle='--', alpha=0.7)
    ax1.axhline(y=180, color='r', linestyle=':', label='Hyper Threshold (180)')
    ax1.axhline(y=70, color='orange', linestyle=':', label='Hypo Threshold (70)')
    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Insulin Dose (U / 5min)', color=color)
    ax2.bar(time_axis_minutes, eval_insulin_history, width=5, color=color, alpha=0.6, label='Insulin Dose')
    ax2.tick_params(axis='y', labelcolor=color)
    meal_labels_seen = set()
    for meal_time, meal_amount in zip(meal_times, meal_carbs):
        label = f'Meal ({meal_amount}g)'
        if label not in meal_labels_seen:
            ax1.axvline(x=meal_time, color='black', linestyle='--', label=label)
            meal_labels_seen.add(label)
        else:
            ax1.axvline(x=meal_time, color='black', linestyle='--')
    fig.suptitle(f'DDPG Performance for {patient_name}', fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')
    plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved evaluation plot to {plot_path}")
    
    # Calculate evaluation metrics
    glucose_history = np.array(glucose_history)
    time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
    time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
    time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
    mean_glucose = np.mean(glucose_history)
    min_glucose = np.min(glucose_history)
    max_glucose = np.max(glucose_history)

    return {
        "eval_metrics": {
            "Patient": patient_name,
            "Mean Glucose (mg/dL)": mean_glucose,
            "Time in Range (%)": time_in_range,
            "Time Hypo (%)": time_hypo,
            "Time Hyper (%)": time_hyper,
            "Min Glucose": min_glucose,
            "Max Glucose": max_glucose,
        },
        "training_rewards": training_rewards_history
    }

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    hyperparameters = {
        'max_episodes': 200, 
        'lr': 1e-4,
        'gamma_val': 0.99, 
        'tau': 0.005,
        'batch_size': 256, 
        'replay_buffer_size': 1000000,
        'max_timesteps_per_episode': 288, 
        'learning_starts': 1500, 
        'ETA': 4.0,
        'expl_noise': 0.1
    }
    
    AGENT_NAME = 'ddpg_personalized_exp_closed_loop'
    results_dir = f'./results/{AGENT_NAME}'
    model_dir = f'./models/{AGENT_NAME}'
    os.makedirs(results_dir, exist_ok=True)

    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
    tasks = []
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("This script requires at least one GPU.")

    for i, patient_name in enumerate(adult_patients):
        gpu_id = i % num_gpus
        tasks.append((patient_name, gpu_id, 42 + i, hyperparameters))

    print(f"\nStarting parallel DDPG training for {len(tasks)} patients on {num_gpus} GPUs...")
    num_processes = min(len(tasks), mp.cpu_count()) 
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(train_and_evaluate_patient, tasks)

    print("\n--- ALL DDPG TRAINING PROCESSES COMPLETE ---")

    # 4. AGGREGATE, DISPLAY, AND PLOT FINAL RESULTS
    if results:
        eval_results = [res['eval_metrics'] for res in results if res and res.get('eval_metrics')]
        training_rewards_list = [res['training_rewards'] for res in results if res and res.get('training_rewards')]

        if not eval_results:
            print("\n⚠️ No valid evaluation results were generated to summarize.")
            exit()
            
        results_df = pd.DataFrame(eval_results)
        
        average_metrics = results_df.select_dtypes(include=np.number).mean()
        std_dev_metrics = results_df.select_dtypes(include=np.number).std()

        print("\n" + "="*70)
        print(f"--- OVERALL {AGENT_NAME.upper()} EVALUATION SUMMARY ---")
        print("="*70)
        print("\n--- Performance per Patient ---")
        print(results_df.to_string(index=False))
        
        print("\n\n--- Average Performance Across All Patients ---")
        print(f"Mean Glucose (mg/dL): {average_metrics['Mean Glucose (mg/dL)']:.2f} ± {std_dev_metrics['Mean Glucose (mg/dL)']:.2f}")
        print(f"Time in Range (%):    {average_metrics['Time in Range (%)']:.2f} ± {std_dev_metrics['Time in Range (%)']:.2f}")
        print(f"Time Hypo (%):        {average_metrics['Time Hypo (%)']:.2f} ± {std_dev_metrics['Time Hypo (%)']:.2f}")
        print(f"Time Hyper (%):       {average_metrics['Time Hyper (%)']:.2f} ± {std_dev_metrics['Time Hyper (%)']:.2f}")
        
        summary_path = os.path.join(results_dir, 'overall_summary.csv')
        results_df.to_csv(summary_path, index=False, float_format='%.2f')
        print(f"\n✅ Saved overall summary table to {summary_path}")
        
        plot_cvga(results_df, results_dir, AGENT_NAME.upper())
        # (Other overall plots would go here)
        
    # 5. EXTENDED EVALUATION FOR AGP PLOTS
    print("\n" + "="*50)
    print("--- STARTING EXTENDED EVALUATION FOR AGP ---")
    print("="*50)
    
    all_days_glucose_data = []
    N_DAYS = 30
    
    for patient_name in adult_patients:
        print(f"Running {N_DAYS}-day simulation for {patient_name}...")
        
        clean_patient_name = patient_name.replace('#', '-')
        actor_path = f'{model_dir}/actor_{clean_patient_name}.pth'
        if not os.path.exists(actor_path):
            print(f"  Model for {patient_name} not found, skipping.")
            continue
            
        eval_agent = DDPGAgent(state_dim=3, action_dim=1, max_action=1.0, lr=hyperparameters['lr'], 
                               gamma=hyperparameters['gamma_val'], tau=hyperparameters['tau'], device='cpu')
        eval_agent.actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
        eval_agent.actor.eval()
        
        patient_days_data = []
        for day in range(N_DAYS):
            day_seed = abs(hash(f"{patient_name}_{day}")) % (2**32)
            scenario = scgen.RandomScenario(start_time=datetime.now(), seed=day_seed)
            env_id = f'simglucose/agp-{clean_patient_name}-v0'
            try:
                register(id=env_id, entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
                         max_episode_steps=288, kwargs={"patient_name": patient_name, "custom_scenario": scenario})
            except gymnasium.error.Error:
                pass
            
            env = gymnasium.make(env_id)
            manager = StateRewardManager(3)
            
            obs, _ = env.reset()
            glucose_history = [obs[0]]
            done = False
            
            while not done:
                unnormalized_state = manager.get_full_state(obs[0])
                current_state = manager.get_normalized_state(unnormalized_state)
                with torch.no_grad():
                    raw_action = eval_agent.select_action(current_state)
                
                insulin_dose = float(env.action_space.high[0]) * np.exp(hyperparameters['ETA'] * (raw_action - 1.0))
                safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
                
                obs, _, terminated, truncated, _ = env.step(safe_action)
                done = terminated or truncated
                glucose_history.append(obs[0])
            
            if len(glucose_history) >= 289:
                patient_days_data.append(glucose_history[:289])
            env.close()

    if patient_days_data:
        all_days_glucose_data.extend(patient_days_data)

    if all_days_glucose_data:
        plot_agp(np.array(all_days_glucose_data), results_dir, AGENT_NAME.upper())

    print("\n==============================================")
    print("All analysis and plotting is complete!")
    print("==============================================")