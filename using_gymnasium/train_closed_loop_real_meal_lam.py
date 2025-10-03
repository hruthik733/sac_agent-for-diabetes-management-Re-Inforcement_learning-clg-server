# train_closed_loop_lam.py
# Description: Main training script for the SAC agent using a Linear Action Mapping.
# This version includes all individual and aggregate performance plots.
'''got final plotting error override in env registering at evaltion; below code got error'''

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
from agents.sac_agent_mdn import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.state_management_closed_loop import StateRewardManager
from utils.safety2_closed_loop import SafetyLayer
from utils.realistic_scenario import RealisticMealScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario

def train_and_evaluate_patient(patient_name, gpu_id, seed, hyperparameters):
    """
    This function contains the entire training and evaluation loop for a single patient
    in a closed-loop setting, using a stable linear action mapping.
    """
    # 1. SETUP AND CONFIGURATION
    # =================================================================
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Process for {patient_name} starting on device: {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)

    # Unpack hyperparameters
    max_episodes = hyperparameters['max_episodes']
    lr = hyperparameters['lr']
    gamma_val = hyperparameters['gamma_val']
    tau = hyperparameters['tau']
    alpha = hyperparameters['alpha']
    batch_size = hyperparameters['batch_size']
    n_latent_var = hyperparameters['n_latent_var']
    replay_buffer_size = hyperparameters['replay_buffer_size']
    max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
    learning_starts = hyperparameters['learning_starts']
    MAX_ACTION_U_PER_5MIN = hyperparameters['MAX_ACTION_U_PER_5MIN']

    AGENT_NAME = 'sac_linear_action_mapping_exp'
    model_dir = f'./models/{AGENT_NAME}'
    results_dir = f'./results/{AGENT_NAME}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    actor_path = f'{model_dir}/actor_{patient_name.replace("#", "-")}.pth'
    
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    patient_for_scenario = T1DPatient.withName(patient_name)
    meal_scenario = RealisticMealScenario(start_time=start_time, patient=patient_for_scenario, seed=seed)

    clean_patient_name = patient_name.replace('#', '-')
    env_id = f'simglucose/sac-linear-{clean_patient_name}-v0'
    
    try:
        register(
            id=env_id,
            entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
            max_episode_steps=max_timesteps_per_episode,
            kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario}
        )
    except gymnasium.error.Error:
        pass

    env = gymnasium.make(env_id)
    env.action_space.seed(seed)
    
    state_dim = 3
    action_dim = 1
    i_max = float(env.action_space.high[0])

    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim, max_action=1.0,
        n_latent_var=n_latent_var, lr=lr, gamma=gamma_val, tau=tau,
        alpha=alpha, device=device
    )

    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer(max_allowed_dose=i_max)
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # 2. TRAINING LOOP
    # =================================================================
    print(f"--- Starting Training for Patient: {patient_name} on {device} ---")
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
                raw_action = np.random.uniform(low=-1.0, high=1.0, size=(action_dim,))
            else:
                raw_action = agent.select_action(current_state)

            proposed_action = (raw_action[0] + 1.0) / 2.0 * MAX_ACTION_U_PER_5MIN
            safe_action = safety_layer.apply(proposed_action, unnormalized_state)
            
            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated

            next_unnormalized_state = manager.get_full_state(next_obs_array[0])
            next_state = manager.get_normalized_state(next_unnormalized_state)
            reward = manager.get_reward(next_unnormalized_state)
            
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
        if i_episode % 25 == 0:
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
    plt.title(f'Learning Curve for {patient_name}')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    learning_curve_path = f'{results_dir}/learning_curve_{clean_patient_name}.png'
    plt.savefig(learning_curve_path)
    plt.close()

    # 3. EVALUATION LOOP
    # =================================================================
    print(f"\n--- Starting Evaluation for {patient_name} ---")
    meal_times = [7 * 60, 12 * 60, 18 * 60]
    meal_carbs = [45, 70, 80]
    eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
    
    register(id=env_id, entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps_per_episode, kwargs={"patient_name": patient_name, "custom_scenario": eval_scenario}, override=True)
    eval_env = gymnasium.make(env_id)

    eval_agent = SACAgent(state_dim=state_dim, action_dim=action_dim, max_action=1.0, n_latent_var=n_latent_var, lr=lr, gamma=gamma_val, tau=tau, alpha=alpha, device=device)
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

        proposed_action = (raw_action[0] + 1.0) / 2.0 * MAX_ACTION_U_PER_5MIN
        safe_action = safety_layer.apply(proposed_action, unnormalized_state)
        
        eval_insulin_history.append(safe_action[0])
        obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)

        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history.append(obs_array[0])

        if terminated or truncated:
            break
            
    eval_env.close()

    # --- RESTORED PLOTTING BLOCK FOR INDIVIDUAL EVALUATION ---
    # =================================================================
    time_axis_minutes = np.arange(len(glucose_history)) * 5
    if len(eval_insulin_history) < len(glucose_history):
       eval_insulin_history.extend([0] * (len(glucose_history) - len(eval_insulin_history)))

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
    ax2.bar(time_axis_minutes[:len(eval_insulin_history)], eval_insulin_history, width=5, color=color, alpha=0.6, label='Insulin Dose')
    ax2.tick_params(axis='y', labelcolor=color)
    
    meal_labels_seen = set()
    for meal_time, meal_amount in zip(meal_times, meal_carbs):
        label = f'Meal ({meal_amount}g)'
        if label not in meal_labels_seen:
            ax1.axvline(x=meal_time, color='black', linestyle='--', label=label)
            meal_labels_seen.add(label)
        else:
            ax1.axvline(x=meal_time, color='black', linestyle='--')
            
    fig.suptitle(f'Evaluation for {patient_name} (Linear Action Mapping)', fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')
    plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved evaluation plot to {plot_path}")
    # =================================================================

    # Calculate evaluation metrics
    glucose_history = np.array(glucose_history)
    time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
    time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
    time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
    mean_glucose = np.mean(glucose_history)

    return {
        "eval_metrics": { "Patient": patient_name, "Mean Glucose (mg/dL)": mean_glucose, "Time in Range (%)": time_in_range, "Time Hypo (%)": time_hypo, "Time Hyper (%)": time_hyper },
        "training_rewards": training_rewards_history
    }


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    hyperparameters = { 'max_episodes': 250, 'lr': 3e-4, 'gamma_val': 0.99, 'tau': 0.005, 'alpha': 0.2, 'batch_size': 256, 'n_latent_var': 256, 'replay_buffer_size': 1000000, 'max_timesteps_per_episode': 288, 'learning_starts': 1000, 'MAX_ACTION_U_PER_5MIN': 2.0 }
    
    AGENT_NAME = 'sac_linear_action_mapping_exp'
    results_dir = f'./results/{AGENT_NAME}'
    os.makedirs(results_dir, exist_ok=True)

    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
    tasks = []
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("WARNING: No GPUs found. Running on CPU, which will be very slow.")
        num_gpus = 1

    for i, patient_name in enumerate(adult_patients):
        gpu_id = 0 if torch.cuda.device_count() == 0 else i % num_gpus
        tasks.append((patient_name, gpu_id, 42 + i, hyperparameters))

    print("\n" + "="*50 + f"\nStarting parallel training for {len(tasks)} patients...\nUsing Linear Action Mapping with MAX_ACTION = {hyperparameters['MAX_ACTION_U_PER_5MIN']} U\n" + "="*50)
    
    num_processes = min(len(tasks), mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(train_and_evaluate_patient, tasks)

    print("\n" + "="*50 + "\n--- ALL TRAINING PROCESSES COMPLETE ---\n" + "="*50)

    # 4. AGGREGATE, DISPLAY, AND PLOT FINAL RESULTS
    # =================================================================
    if results:
        eval_results = [res['eval_metrics'] for res in results if res and res.get('eval_metrics')]
        training_rewards_list = [res['training_rewards'] for res in results if res and res.get('training_rewards')]

        if not eval_results:
            print("\n⚠️ No valid evaluation results were generated to summarize.")
        else:
            results_df = pd.DataFrame(eval_results)
            results_df = results_df[["Patient", "Mean Glucose (mg/dL)", "Time in Range (%)", "Time Hypo (%)", "Time Hyper (%)"]]
            
            average_metrics = results_df.select_dtypes(include=np.number).mean()
            std_dev_metrics = results_df.select_dtypes(include=np.number).std()

            print("\n" + "="*70 + f"\n--- OVERALL {AGENT_NAME.upper()} EVALUATION SUMMARY ---\n" + "="*70)
            print("\n--- Performance per Patient ---\n" + results_df.to_string(index=False))
            print("\n\n--- Average Performance Across All Patients ---")
            print(f"Mean Glucose (mg/dL): {average_metrics['Mean Glucose (mg/dL)']:.2f} ± {std_dev_metrics['Mean Glucose (mg/dL)']:.2f}")
            print(f"Time in Range (%):    {average_metrics['Time in Range (%)']:.2f} ± {std_dev_metrics['Time in Range (%)']:.2f}")
            print(f"Time Hypo (%):        {average_metrics['Time Hypo (%)']:.2f} ± {std_dev_metrics['Time Hypo (%)']:.2f}")
            print(f"Time Hyper (%):       {average_metrics['Time Hyper (%)']:.2f} ± {std_dev_metrics['Time Hyper (%)']:.2f}")
            
            summary_path = os.path.join(results_dir, 'overall_summary.csv')
            results_df.to_csv(summary_path, index=False, float_format='%.2f')
            print(f"\n✅ Saved overall summary table to {summary_path}\n" + "="*70)

            # --- PLOTTING BLOCK: OVERALL SUMMARY PLOT ---
            print("\n--- Generating Overall Summary Plot ---")
            try:
                plot_data = results_df.set_index('Patient')
                time_metrics_df = plot_data[['Time in Range (%)', 'Time Hypo (%)', 'Time Hyper (%)']]
                ax = time_metrics_df.plot(kind='bar', stacked=True, figsize=(16, 9), color={'Time in Range (%)': 'green', 'Time Hypo (%)': 'orange', 'Time Hyper (%)': 'red'}, edgecolor='black', width=0.8)
                plt.title(f'Glycemic Control Summary ({AGENT_NAME.upper()})', fontsize=18, weight='bold')
                plt.ylabel('Percentage of Time (%)', fontsize=14)
                plt.xlabel('Patient ID', fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.yticks(fontsize=12)
                plt.ylim(0, 100)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.axhline(y=70, color='blue', linestyle='--', label='70% Time in Range Target')
                plt.legend(title='Glycemic Range', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout()
                summary_plot_path = os.path.join(results_dir, 'overall_summary_plot.png')
                plt.savefig(summary_plot_path)
                plt.close()
                print(f"✅ Saved overall summary plot to {summary_plot_path}")
            except Exception as e:
                print(f"\n⚠️ Could not generate summary plot. Error: {e}")

            # --- PLOTTING BLOCK: OVERALL AVERAGE LEARNING CURVE ---
            print("\n--- Generating Overall Average Learning Curve ---")
            try:
                rewards_df = pd.DataFrame(training_rewards_list).T
                mean_rewards = rewards_df.mean(axis=1)
                std_rewards = rewards_df.std(axis=1)

                plt.figure(figsize=(12, 6))
                plt.plot(mean_rewards, label='Mean Episode Reward', color='blue')
                plt.fill_between(rewards_df.index, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.2, label='Standard Deviation')
                plt.title(f'Overall Average Learning Curve ({AGENT_NAME.upper()})')
                plt.xlabel('Episode')
                plt.ylabel('Cumulative Reward')
                plt.legend()
                plt.grid(True)
                overall_lc_path = os.path.join(results_dir, 'overall_learning_curve.png')
                plt.savefig(overall_lc_path)
                plt.close()
                print(f"✅ Saved overall average learning curve to {overall_lc_path}")
            except Exception as e:
                print(f"\n⚠️ Could not generate overall learning curve. Error: {e}")

    else:
        print("\n⚠️ No results were generated to summarize.")