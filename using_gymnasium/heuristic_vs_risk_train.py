# import gymnasium
# from gymnasium.envs.registration import register
# import numpy as np
# import torch
# from datetime import datetime
# import os
# import random
# import pandas as pd
# import multiprocessing as mp

# # --- Local Imports ---
# # NOTE: We are using a standard Gaussian SAC agent for a fair comparison of reward functions.
# # Ensure you have a standard 'sac_agent.py' file.
# from agents.sac_agent import SACAgent
# from utils.replay_buffer import ReplayBuffer
# from utils.state_management3 import StateRewardManager
# from utils.safety2 import SafetyLayer
# import simglucose.simulation.scenario_gen as scgen
# from simglucose.simulation.scenario import CustomScenario

# def train_and_evaluate_patient(patient_name, gpu_id, seed, hyperparameters, reward_type):
#     """
#     This function contains the entire training and evaluation loop for a single patient.
#     The 'reward_type' argument ('heuristic' or 'risk') determines which reward function to use.
#     """
#     # Force a non-interactive backend for matplotlib to ensure safety in multiprocessing
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt

#     # 1. SETUP AND CONFIGURATION
#     # =================================================================
#     device = torch.device(f"cuda:{gpu_id}")
#     print(f"Process for {patient_name} (Reward: {reward_type.upper()}) starting on device: {device}")

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
#     alpha = hyperparameters['alpha']
#     batch_size = hyperparameters['batch_size']
#     n_latent_var = hyperparameters['n_latent_var']
#     replay_buffer_size = hyperparameters['replay_buffer_size']
#     max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
#     learning_starts = hyperparameters['learning_starts']
#     ETA = hyperparameters['ETA']

#     # Create reward-specific directories for models and results
#     AGENT_NAME = f'sac_gaussian_{reward_type}_reward'
#     model_dir = f'./models/{AGENT_NAME}'
#     results_dir = f'./results/{AGENT_NAME}'
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(results_dir, exist_ok=True)
#     actor_path = f'{model_dir}/actor_{patient_name.replace("#", "-")}.pth'
    
#     now = datetime.now()
#     start_time = datetime.combine(now.date(), datetime.min.time())
#     meal_scenario = scgen.RandomScenario(start_time=start_time, seed=seed)
#     clean_patient_name = patient_name.replace('#', '-')
#     env_id = f'simglucose/{clean_patient_name}-v0'
    
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
    
#     state_dim = 4
#     action_dim = 1
#     i_max = float(env.action_space.high[0])

#     agent = SACAgent(
#         state_dim=state_dim, action_dim=action_dim, max_action=1.0,
#         n_latent_var=n_latent_var, lr=lr, gamma=gamma_val, tau=tau,
#         alpha=alpha, device=device
#     )

#     manager = StateRewardManager(state_dim)
#     safety_layer = SafetyLayer()
#     replay_buffer = ReplayBuffer(replay_buffer_size)

#     # 2. TRAINING LOOP
#     # =================================================================
#     print(f"--- Starting Training for Patient: {patient_name} (Reward: {reward_type.upper()}) ---")
#     total_timesteps_taken = 0
#     training_rewards_history = []

#     for i_episode in range(1, max_episodes + 1):
#         obs_array, info = env.reset(seed=seed + i_episode)
#         episode_scenario = info.get('scenario')
#         manager.reset()

#         current_sim_time = env.unwrapped.env.env.time
#         upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
#         unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
#         current_state = manager.get_normalized_state(unnormalized_state)
#         episode_reward = 0

#         for t in range(max_timesteps_per_episode):
#             if total_timesteps_taken < learning_starts:
#                 raw_action = np.random.uniform(low=-1.0, high=1.0, size=(action_dim,))
#             else:
#                 raw_action = agent.select_action(current_state)

#             insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
#             safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
            
#             manager.insulin_history.append(safe_action[0])
#             next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
#             done = terminated or truncated

#             current_sim_time = env.unwrapped.env.env.time
#             upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
#             next_unnormalized_state = manager.get_full_state(next_obs_array[0], upcoming_carbs)
#             next_state = manager.get_normalized_state(next_unnormalized_state)
            
#             # *** KEY CHANGE: Use the specified reward_type ***
#             reward = manager.get_reward(next_unnormalized_state, reward_type=reward_type)
            
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
#             print(f"[{patient_name} (Reward: {reward_type.upper()})] Episode {i_episode}/{max_episodes} | Reward: {episode_reward:.2f}")
    
#     env.close()
#     print(f"--- Training Finished for {patient_name} (Reward: {reward_type.upper()}) ---")
#     torch.save(agent.actor.state_dict(), actor_path)
#     print(f"Saved trained model to {actor_path}")

#     # 3. GRAPHING AND EVALUATION (Code remains the same, but uses reward-specific paths)
#     # =================================================================
#     # Plotting individual learning curve
#     plt.figure(figsize=(12, 6))
#     plt.plot(training_rewards_history, label='Episode Reward', alpha=0.6)
#     moving_avg = pd.Series(training_rewards_history).rolling(window=20, min_periods=1).mean()
#     plt.plot(moving_avg, label='Moving Average (20 episodes)', color='red', linewidth=2)
#     plt.title(f'Learning Curve for {patient_name} (Reward: {reward_type.upper()})')
#     plt.xlabel('Episode')
#     plt.ylabel('Cumulative Reward')
#     plt.legend()
#     plt.grid(True)
#     learning_curve_path = f'{results_dir}/learning_curve_{clean_patient_name}.png'
#     plt.savefig(learning_curve_path)
#     plt.close()
#     print(f"Saved learning curve plot to {learning_curve_path}")

#     # Evaluation Loop
#     print(f"\n--- Starting Evaluation for {patient_name} (Reward: {reward_type.upper()}) ---")
#     meal_times = [7 * 60, 12 * 60, 18 * 60]
#     meal_carbs = [45, 70, 80]
#     eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
#     eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
    
#     eval_agent = SACAgent(
#         state_dim=state_dim, action_dim=action_dim, max_action=1.0,
#         n_latent_var=n_latent_var, lr=lr, gamma=gamma_val, tau=tau,
#         alpha=alpha, device=device
#     )
#     eval_agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
#     eval_agent.actor.eval()
#     manager.reset()
#     obs_array, info = eval_env.reset()
#     current_sim_time = eval_env.unwrapped.env.env.time
#     upcoming_carbs = eval_scenario.get_action(current_sim_time).meal if eval_scenario else 0
#     unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
#     current_state = manager.get_normalized_state(unnormalized_state)
#     glucose_history = [obs_array[0]]
#     eval_insulin_history = []

#     for t in range(max_timesteps_per_episode):
#         with torch.no_grad():
#             raw_action = eval_agent.select_action(current_state)
#         insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
#         safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
#         eval_insulin_history.append(safe_action[0])
#         obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)
#         current_sim_time = eval_env.unwrapped.env.env.time
#         upcoming_carbs = eval_scenario.get_action(current_sim_time).meal if eval_scenario else 0
#         unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
#         current_state = manager.get_normalized_state(unnormalized_state)
#         glucose_history.append(obs_array[0])
#         if terminated or truncated:
#             break
#     eval_env.close()

#     # Evaluation Plotting
#     time_axis_minutes = np.arange(len(glucose_history)) * 5
#     if len(eval_insulin_history) != len(glucose_history):
#        eval_insulin_history.append(0)
#     fig, ax1 = plt.subplots(figsize=(15, 7))
#     # ... (plotting code is identical, but will save to the reward-specific folder)
#     fig.suptitle(f'Performance for {patient_name} (Reward: {reward_type.upper()})', fontsize=16, weight='bold')
#     # ...
#     plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
#     plt.savefig(plot_path)
#     plt.close()
#     print(f"Saved evaluation plot to {plot_path}")
    
#     # Metric calculation
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

#     # Define a list of reward functions to test
#     REWARD_TYPES_TO_TEST = ['heuristic', 'hybrid']

#     for reward_type in REWARD_TYPES_TO_TEST:
#         print("\n" + "#"*70)
#         print(f"###### STARTING EXPERIMENT FOR REWARD TYPE: {reward_type.upper()} ######")
#         print("#"*70 + "\n")

#         hyperparameters = {
#             'max_episodes': 200, 'lr': 2e-4, 'gamma_val': 0.99, 'tau': 0.005, 'alpha': 0.2,
#             'batch_size': 256, 'n_latent_var': 256, 'replay_buffer_size': 1000000,
#             'max_timesteps_per_episode': 288, 'learning_starts': 1500, 'ETA': 4.0
#         }
        
#         results_dir = f'./results/sac_gaussian_{reward_type}_reward'
#         os.makedirs(results_dir, exist_ok=True)

#         adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
#         tasks = []
#         num_gpus = torch.cuda.device_count()
#         if num_gpus == 0:
#             raise ValueError("This script requires at least one GPU.")

#         for i, patient_name in enumerate(adult_patients):
#             gpu_id = i % num_gpus
#             tasks.append((patient_name, gpu_id, 42 + i, hyperparameters, reward_type))

#         print(f"Starting parallel training for {len(tasks)} patients on {num_gpus} GPUs...")
        
#         num_processes = min(len(tasks), mp.cpu_count()) 
#         with mp.Pool(processes=num_processes) as pool:
#             results = pool.starmap(train_and_evaluate_patient, tasks)

#         print(f"\n--- ALL TRAINING PROCESSES FOR {reward_type.upper()} REWARD COMPLETE ---")
        
#         # 4. AGGREGATE, DISPLAY, AND PLOT FINAL RESULTS
#         # =================================================================
#         if results:
#             eval_results = [res['eval_metrics'] for res in results]
#             training_rewards_list = [res['training_rewards'] for res in results]

#             results_df = pd.DataFrame(eval_results)
#             column_order = [
#                 "Patient", "Mean Glucose (mg/dL)", "Time in Range (%)",
#                 "Time Hypo (%)", "Time Hyper (%)"
#             ]
#             results_df = results_df[column_order]
#             average_metrics = results_df.select_dtypes(include=np.number).mean()
            
#             print("\n" + "="*70)
#             print(f"--- OVERALL EVALUATION SUMMARY (REWARD: {reward_type.upper()}) ---")
#             # ... (rest of summary printing is the same)
            
#             # This part will now run for each reward type, saving results to the correct folder
#             # ... (rest of plotting and saving logic is the same)
#             print("="*70)
#             print("\n--- Performance per Patient ---")
#             print(results_df.to_string(index=False))
#             print("\n\n--- Average Performance Across All Patients ---")
#             print(f"Mean Glucose (mg/dL): {average_metrics['Mean Glucose (mg/dL)']:.2f}")
#             print(f"Time in Range (%):    {average_metrics['Time in Range (%)']:.2f}")
#             print(f"Time Hypo (%):        {average_metrics['Time Hypo (%)']:.2f}")
#             print(f"Time Hyper (%):       {average_metrics['Time Hyper (%)']:.2f}")
#             summary_path = os.path.join(results_dir, 'overall_summary.csv')
#             results_df.to_csv(summary_path, index=False, float_format='%.2f')
#             print(f"\n✅ Saved overall summary table to {summary_path}")
#             print("\n" + "="*70)

#             # --- Display Overall Summary Plot (same as before) ---
#             # ... (code for the stacked bar chart is the same)
#             print("\n--- Generating Overall Summary Plot ---")
#             try:
#                 plot_data = results_df.set_index('Patient')
#                 time_metrics_df = plot_data[['Time in Range (%)', 'Time Hypo (%)', 'Time Hyper (%)']]
#                 ax = time_metrics_df.plot(
#                     kind='bar', stacked=True, figsize=(16, 9),
#                     color={'Time in Range (%)': 'green', 'Time Hypo (%)': 'orange', 'Time Hyper (%)': 'red'},
#                     edgecolor='black', width=0.8
#                 )
#                 plt.title('Glycemic Control Summary Across All Patients', fontsize=18, weight='bold')
#                 plt.ylabel('Percentage of Time (%)', fontsize=14)
#                 plt.xlabel('Patient ID', fontsize=14)
#                 plt.xticks(rotation=45, ha='right', fontsize=12)
#                 plt.yticks(fontsize=12)
#                 plt.ylim(0, 100)
#                 plt.grid(axis='y', linestyle='--', alpha=0.7)
#                 plt.axhline(y=70, color='blue', linestyle='--', label='70% Time in Range Target')
#                 plt.legend(title='Glycemic Range', bbox_to_anchor=(1.02, 1), loc='upper left')
#                 plt.tight_layout()
#                 summary_plot_path = os.path.join(results_dir, 'overall_summary_plot.png')
#                 plt.savefig(summary_plot_path)
#                 plt.close()
#                 print(f"✅ Saved overall summary plot to {summary_plot_path}")
#             except Exception as e:
#                 print(f"\n⚠️ Could not generate summary plot. Error: {e}")

#             # ############################################################### #
#             # ########### NEW: PLOT OVERALL AVERAGE LEARNING CURVE ########## #
#             # ############################################################### #
#             print("\n--- Generating Overall Average Learning Curve ---")
#             try:
#                 # Create a DataFrame from the list of rewards, transposing it
#                 rewards_df = pd.DataFrame(training_rewards_list).T
                
#                 # Calculate mean and standard deviation across patients for each episode
#                 mean_rewards = rewards_df.mean(axis=1)
#                 std_rewards = rewards_df.std(axis=1)

#                 plt.figure(figsize=(12, 6))
#                 plt.plot(mean_rewards, label='Mean Episode Reward', color='blue')
#                 # Fill the area between mean +/- one standard deviation
#                 plt.fill_between(
#                     rewards_df.index,
#                     mean_rewards - std_rewards,
#                     mean_rewards + std_rewards,
#                     color='blue',
#                     alpha=0.2,
#                     label='Standard Deviation'
#                 )
#                 plt.title('Overall Average Learning Curve Across All Patients')
#                 plt.xlabel('Episode')
#                 plt.ylabel('Cumulative Reward')
#                 plt.legend()
#                 plt.grid(True)

#                 overall_lc_path = os.path.join(results_dir, 'overall_learning_curve.png')
#                 plt.savefig(overall_lc_path)
#                 plt.close()
#                 print(f"✅ Saved overall average learning curve to {overall_lc_path}")

#             except Exception as e:
#                 print(f"\n⚠️ Could not generate overall learning curve. Error: {e}")

#         else:
#             print("\n⚠️ No results were generated to summarize.")






'''closed looop '''
import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import os
import random
import pandas as pd
import multiprocessing as mp

# --- Local Imports ---
from agents.sac_agent import SACAgent as SACAgentGaussian
from utils.replay_buffer import ReplayBuffer
from utils.state_management4 import StateRewardManager
from utils.safety2 import SafetyLayer
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario

def train_and_evaluate_patient(patient_name, gpu_id, seed, hyperparameters):
    """
    Worker function to train/evaluate a single patient.
    The `use_meal_announcements` hyperparameter determines the experiment mode.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1. SETUP AND CONFIGURATION
    # =================================================================
    device = torch.device(f"cuda:{gpu_id}")
    
    # Unpack hyperparameters
    use_meal_announcements = hyperparameters['use_meal_announcements']
    experiment_type = 'MealsAnnounced' if use_meal_announcements else 'ClosedLoop'
    print(f"Process for {patient_name} ({experiment_type}) starting on device: {device}")

    # Seed everything for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)

    # General hyperparameters
    max_episodes = hyperparameters['max_episodes']
    lr, gamma_val, tau, alpha = hyperparameters['lr'], hyperparameters['gamma_val'], hyperparameters['tau'], hyperparameters['alpha']
    batch_size, n_latent_var = hyperparameters['batch_size'], hyperparameters['n_latent_var']
    replay_buffer_size, learning_starts = hyperparameters['replay_buffer_size'], hyperparameters['learning_starts']
    ETA, train_freq, reward_type = hyperparameters['ETA'], hyperparameters['train_freq'], hyperparameters['reward_type']

    # Set up directories based on the experiment type
    AGENT_NAME = f'sac_optimized_{experiment_type}'
    model_dir = f'./models/{AGENT_NAME}'
    results_dir = f'./results/{AGENT_NAME}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    actor_path = f'{model_dir}/actor_{patient_name.replace("#", "-")}.pth'
    
    # Environment setup
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=seed)
    clean_patient_name = patient_name.replace('#', '-')
    env_id = f'simglucose/{clean_patient_name}-v0'
    
    try:
        register(id=env_id, entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
                 max_episode_steps=hyperparameters['max_timesteps_per_episode'],
                 kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario})
    except gymnasium.error.Error: pass

    env = gymnasium.make(env_id)
    env.action_space.seed(seed)
    
    # Initialize the manager with the correct mode
    manager = StateRewardManager(use_meal_announcements=use_meal_announcements)
    state_dim = manager.state_dim
    action_dim = 1
    i_max = float(env.action_space.high[0])

    agent = SACAgentGaussian(state_dim=state_dim, action_dim=action_dim, max_action=1.0, n_latent_var=n_latent_var,
                             lr=lr, gamma=gamma_val, tau=tau, alpha=alpha, device=device)
    safety_layer, replay_buffer = SafetyLayer(), ReplayBuffer(replay_buffer_size)

    # 2. TRAINING LOOP
    # =================================================================
    print(f"--- Starting Training: {patient_name} ({experiment_type}) ---")
    total_timesteps_taken = 0
    training_rewards_history = []
    
    for i_episode in range(1, max_episodes + 1):
        obs_array, info = env.reset(seed=seed + i_episode)
        manager.reset()
        episode_reward = 0
        
        # Correctly get state based on whether meals are announced
        upcoming_carbs = info.get('scenario').get_action(env.unwrapped.env.env.time).meal if use_meal_announcements and info.get('scenario') else 0
        unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs=upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)

        for t in range(hyperparameters['max_timesteps_per_episode']):
            raw_action = agent.select_action(current_state) if total_timesteps_taken >= learning_starts else np.random.uniform(-1., 1., size=(action_dim,))
            insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
            safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
            
            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated
            
            # Get next state correctly
            upcoming_carbs = info.get('scenario').get_action(env.unwrapped.env.env.time).meal if use_meal_announcements and info.get('scenario') else 0
            next_unnormalized_state = manager.get_full_state(next_obs_array[0], upcoming_carbs=upcoming_carbs)
            reward = manager.get_reward(next_unnormalized_state, reward_type=reward_type)
            next_state = manager.get_normalized_state(next_unnormalized_state)
            
            replay_buffer.push(current_state, raw_action, reward, next_state, done)
            current_state, unnormalized_state = next_state, next_unnormalized_state
            episode_reward += reward
            total_timesteps_taken += 1

            if total_timesteps_taken > learning_starts and total_timesteps_taken % train_freq == 0:
                for _ in range(train_freq):
                    if len(replay_buffer) > batch_size:
                        agent.update(replay_buffer, batch_size)
            if done: break
        
        training_rewards_history.append(episode_reward)
        if i_episode % 50 == 0:
            print(f"[{patient_name} ({experiment_type})] Ep {i_episode}/{max_episodes} | Reward: {episode_reward:.2f}")

    env.close()
    torch.save(agent.actor.state_dict(), actor_path)
    print(f"--- Training Finished for {patient_name} ({experiment_type}). Model saved. ---")
    
    # 3. PLOT LEARNING CURVE
    plt.figure(figsize=(12, 6))
    plt.plot(pd.Series(training_rewards_history).rolling(20).mean(), label='Moving Avg (20 ep)', color='red')
    plt.plot(training_rewards_history, label='Episode Reward', alpha=0.4)
    plt.title(f'Learning Curve: {patient_name} ({experiment_type})')
    plt.xlabel('Episode'); plt.ylabel('Cumulative Reward'); plt.legend(); plt.grid(True)
    learning_curve_path = f'{results_dir}/learning_curve_{clean_patient_name}.png'
    plt.savefig(learning_curve_path)
    plt.close()
    print(f"Saved learning curve plot to {learning_curve_path}")


    # 4. EVALUATION
    print(f"--- Starting Evaluation: {patient_name} ({experiment_type}) ---")
    meal_times, meal_carbs = [7*60, 12*60, 18*60], [45, 70, 80]
    eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
    eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
    eval_agent = SACAgentGaussian(state_dim=state_dim, action_dim=action_dim, max_action=1.0, n_latent_var=n_latent_var, lr=lr, gamma=gamma_val, tau=tau, alpha=alpha, device=device)
    eval_agent.actor.load_state_dict(torch.load(actor_path, map_location=device)); eval_agent.actor.eval()
    manager.reset()
    obs_array, _ = eval_env.reset()
    glucose_history, insulin_history = [obs_array[0]], []
    
    upcoming_carbs_eval = eval_scenario.get_action(eval_env.unwrapped.env.env.time).meal if use_meal_announcements else 0
    unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs=upcoming_carbs_eval)
    current_state = manager.get_normalized_state(unnormalized_state)

    for _ in range(hyperparameters['max_timesteps_per_episode']):
        with torch.no_grad(): raw_action = eval_agent.select_action(current_state)
        insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
        safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
        insulin_history.append(safe_action[0])
        obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)
        
        upcoming_carbs_eval = eval_scenario.get_action(eval_env.unwrapped.env.env.time).meal if use_meal_announcements else 0
        unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs=upcoming_carbs_eval)
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history.append(obs_array[0])
        if terminated or truncated: break
    eval_env.close()

    # 5. PLOT EVALUATION GRAPH
    time_axis_minutes = np.arange(len(glucose_history)) * 5
    if len(insulin_history) != len(glucose_history):
       insulin_history.append(0)

    fig, ax1 = plt.subplots(figsize=(15, 7))
    color = 'tab:blue'
    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Blood Glucose (mg/dL)', color=color, fontsize=12)
    ax1.plot(time_axis_minutes, glucose_history, color=color, label='Blood Glucose')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(which='major', axis='y', linestyle='--', alpha=0.7)
    ax1.axhline(y=180, color='r', linestyle=':', label='Hyper Threshold (180)')
    ax1.axhline(y=70, color='orange', linestyle=':', label='Hypo Threshold (70)')
    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Insulin Dose (U / 5min)', color=color, fontsize=12)
    ax2.bar(time_axis_minutes, insulin_history, width=5, color=color, alpha=0.6, label='Insulin Dose')
    ax2.tick_params(axis='y', labelcolor=color)
    meal_labels_seen = set()
    for meal_time, meal_amount in zip(meal_times, meal_carbs):
        label = f'Meal ({meal_amount}g)'
        if label not in meal_labels_seen:
            ax1.axvline(x=meal_time, color='black', linestyle='--', label=label)
            meal_labels_seen.add(label)
        else:
            ax1.axvline(x=meal_time, color='black', linestyle='--')
    fig.suptitle(f'Performance: {patient_name} ({experiment_type})', fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')
    plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved evaluation plot to {plot_path}")

    # 6. RETURN METRICS
    glucose_arr = np.array(glucose_history)
    metrics = {"Patient": patient_name,
               "Mean Glucose (mg/dL)": np.mean(glucose_arr),
               "Time in Range (%)": np.sum((glucose_arr >= 70) & (glucose_arr <= 180)) / len(glucose_arr) * 100,
               "Time Hypo (%)": np.sum(glucose_arr < 70) / len(glucose_arr) * 100,
               "Time Hyper (%)": np.sum(glucose_arr > 180) / len(glucose_arr) * 100}
    return {"eval_metrics": metrics, "training_rewards": training_rewards_history}

if __name__ == '__main__':
    try: mp.set_start_method('spawn')
    except RuntimeError: pass

    # Define the two experiment modes to compare
    EXPERIMENT_MODES = [True, False] # True: Meals Announced, False: Closed Loop

    for use_meals in EXPERIMENT_MODES:
        experiment_name = 'MealsAnnounced' if use_meals else 'ClosedLoop'
        print("\n" + "#"*70 + f"\n###### STARTING EXPERIMENT: {experiment_name.upper()} ######\n" + "#"*70 + "\n")

        hyperparameters = {
            'max_episodes': 200, 'lr': 2e-4, 'gamma_val': 0.99, 'tau': 0.005, 'alpha': 0.2,
            'batch_size': 1024, 'n_latent_var': 256, 'replay_buffer_size': 1000000,
            'max_timesteps_per_episode': 288, 'learning_starts': 5000, 'ETA': 4.0,
            'train_freq': 4, 'reward_type': 'hybrid',
            'use_meal_announcements': use_meals # Pass mode to workers
        }
        
        adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
        tasks, num_gpus = [], torch.cuda.device_count()
        if num_gpus == 0: raise ValueError("This script requires at least one GPU.")

        for i, patient_name in enumerate(adult_patients):
            tasks.append((patient_name, i % num_gpus, 42 + i, hyperparameters))

        # Launch processes
        num_processes = min(len(tasks), num_gpus * 3) # 3 workers per GPU
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(train_and_evaluate_patient, tasks)

        print(f"\n--- ALL PROCESSES FOR {experiment_name.upper()} COMPLETE ---\n")

        # 7. AGGREGATE AND PLOT FINAL RESULTS
        if results:
            results_dir = f'./results/sac_optimized_{experiment_name}'
            eval_results = [res['eval_metrics'] for res in results]
            training_rewards_list = [res['training_rewards'] for res in results]
            
            # Create and save summary table
            results_df = pd.DataFrame(eval_results)
            column_order = ["Patient", "Mean Glucose (mg/dL)", "Time in Range (%)", "Time Hypo (%)", "Time Hyper (%)"]
            results_df = results_df[column_order]
            average_metrics = results_df.select_dtypes(include=np.number).mean()
            
            print("\n" + "="*70)
            print(f"--- OVERALL SUMMARY: {experiment_name.upper()} ---")
            print("="*70)
            print("\n--- Performance per Patient ---")
            print(results_df.to_string(index=False))
            print("\n\n--- Average Performance Across All Patients ---")
            print(f"Mean Glucose (mg/dL): {average_metrics['Mean Glucose (mg/dL)']:.2f}")
            print(f"Time in Range (%):    {average_metrics['Time in Range (%)']:.2f}")
            print(f"Time Hypo (%):        {average_metrics['Time Hypo (%)']:.2f}")
            print(f"Time Hyper (%):       {average_metrics['Time Hyper (%)']:.2f}")
            summary_path = os.path.join(results_dir, 'overall_summary.csv')
            results_df.to_csv(summary_path, index=False, float_format='%.2f')
            print(f"\n✅ Saved overall summary table to {summary_path}")
            
            # Create and save overall summary plot
            try:
                plot_data = results_df.set_index('Patient')
                time_metrics_df = plot_data[['Time in Range (%)', 'Time Hypo (%)', 'Time Hyper (%)']]
                time_metrics_df.plot(kind='bar', stacked=True, figsize=(16, 9),
                                     color={'Time in Range (%)': 'green', 'Time Hypo (%)': 'orange', 'Time Hyper (%)': 'red'},
                                     edgecolor='black', width=0.8)
                plt.title(f'Glycemic Control Summary ({experiment_name})', fontsize=18, weight='bold')
                plt.ylabel('Percentage of Time (%)', fontsize=14)
                plt.xlabel('Patient ID', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 100); plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.axhline(y=70, color='blue', linestyle='--', label='70% TIR Target')
                plt.legend(title='Glycemic Range', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout()
                summary_plot_path = os.path.join(results_dir, 'overall_summary_plot.png')
                plt.savefig(summary_plot_path)
                plt.close()
                print(f"✅ Saved overall summary plot to {summary_plot_path}")
            except Exception as e:
                print(f"\n⚠️ Could not generate summary plot. Error: {e}")

            # Create and save overall average learning curve
            try:
                rewards_df = pd.DataFrame(training_rewards_list).T
                mean_rewards = rewards_df.mean(axis=1)
                std_rewards = rewards_df.std(axis=1)
                plt.figure(figsize=(12, 6))
                plt.plot(mean_rewards, label='Mean Episode Reward', color='blue')
                plt.fill_between(rewards_df.index, mean_rewards - std_rewards, mean_rewards + std_rewards,
                                 color='blue', alpha=0.2, label='Standard Deviation')
                plt.title(f'Overall Average Learning Curve ({experiment_name})')
                plt.xlabel('Episode'); plt.ylabel('Cumulative Reward'); plt.legend(); plt.grid(True)
                overall_lc_path = os.path.join(results_dir, 'overall_learning_curve.png')
                plt.savefig(overall_lc_path)
                plt.close()
                print(f"✅ Saved overall average learning curve to {overall_lc_path}")
            except Exception as e:
                print(f"\n⚠️ Could not generate overall learning curve. Error: {e}")

        else:
            print(f"\n⚠️ No results were generated for {experiment_name} to summarize.")

