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
from agents.ensemble_agent import EnsembleAgent
from agents.sac_agent_mdn import SACAgent
from agents.td3_agent import TD3Agent
from utils.replay_buffer import ReplayBuffer
from utils.state_management_closed_loop_ensemble import StateRewardManager
from utils.safety2_closed_loop import SafetyLayer
from utils.realistic_scenario import RealisticMealScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario

def train_and_evaluate_patient(patient_name, gpu_id, seed, hyperparameters):
    """
    Training and evaluation loop for ensemble SAC-TD3 agent
    """
    # 1. SETUP AND CONFIGURATION
    # =================================================================
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Process for {patient_name} starting on device: {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    ETA = hyperparameters['ETA']
    policy_delay = hyperparameters['policy_delay']
    ensemble_weights = hyperparameters['ensemble_weights']

    AGENT_NAME = 'ensemble_sac_td3_exp'
    model_dir = f'./models/{AGENT_NAME}'
    results_dir = f'./results/{AGENT_NAME}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    ensemble_path = f'{model_dir}/ensemble_{patient_name.replace("#", "-")}.pth'
    
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    # Create patient and meal scenario
    patient_for_scenario = T1DPatient.withName(patient_name)
    meal_scenario = RealisticMealScenario(start_time=start_time, patient=patient_for_scenario, seed=seed)

    clean_patient_name = patient_name.replace('#', '-')
    env_id = f'simglucose/ensemble-{clean_patient_name}-v0'
    
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

    # Initialize ensemble agent
    ensemble_agent = EnsembleAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        n_latent_var=n_latent_var,
        lr=lr,
        gamma=gamma_val,
        tau=tau,
        alpha=alpha,
        policy_delay=policy_delay,
        ensemble_weights=ensemble_weights,
        device=device
    )

    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # 2. TRAINING LOOP
    # =================================================================
    print(f"--- Starting Ensemble Training for Patient: {patient_name} on {device} ---")
    total_timesteps_taken = 0
    training_rewards_history = []
    sac_usage_count = 0
    td3_usage_count = 0

    for i_episode in range(1, max_episodes + 1):
        obs_array, info = env.reset(seed=seed + i_episode)
        manager.reset()

        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        episode_reward = 0

        for t in range(max_timesteps_per_episode):
            if total_timesteps_taken < learning_starts:
                raw_action = np.random.uniform(low=-1.0, high=1.0, size=(action_dim,))
                selected_agent = 'random'
            else:
                raw_action, selected_agent = ensemble_agent.select_action(current_state)
                if selected_agent == 'sac':
                    sac_usage_count += 1
                elif selected_agent == 'td3':
                    td3_usage_count += 1

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
                ensemble_agent.update(replay_buffer, batch_size)

            if done:
                break
        
        training_rewards_history.append(episode_reward)
        if i_episode % 50 == 0:
            print(f"[{patient_name}] Episode {i_episode}/{max_episodes} | Reward: {episode_reward:.2f} | SAC: {sac_usage_count} TD3: {td3_usage_count}")
    
    env.close()
    print(f"--- Training Finished for {patient_name} ---")
    print(f"Final Agent Usage - SAC: {sac_usage_count}, TD3: {td3_usage_count}")
    
    # Save ensemble model
    ensemble_agent.save(ensemble_path)
    print(f"Saved ensemble model to {ensemble_path}")

    # Plot learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(training_rewards_history, label='Episode Reward', alpha=0.6)
    moving_avg = pd.Series(training_rewards_history).rolling(window=20, min_periods=1).mean()
    plt.plot(moving_avg, label='Moving Average (20 episodes)', color='red', linewidth=2)
    plt.title(f'Ensemble Learning Curve for {patient_name}')
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
    
    try:
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
    except gymnasium.error.Error:
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
    
    # Load ensemble for evaluation
    eval_ensemble = EnsembleAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        n_latent_var=n_latent_var,
        lr=lr,
        gamma=gamma_val,
        tau=tau,
        alpha=alpha,
        policy_delay=policy_delay,
        ensemble_weights=ensemble_weights,
        device=device
    )
    eval_ensemble.load(ensemble_path, device)
    eval_ensemble.set_eval_mode()

    manager.reset()
    obs_array, info = eval_env.reset()
    
    unnormalized_state = manager.get_full_state(obs_array[0])
    current_state = manager.get_normalized_state(unnormalized_state)
    
    glucose_history = [obs_array[0]]
    eval_insulin_history = []
    agent_selections = []

    for t in range(max_timesteps_per_episode):
        with torch.no_grad():
            raw_action, selected_agent = eval_ensemble.select_action(current_state)
            agent_selections.append(selected_agent)

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
    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Blood Glucose (mg/dL)', color=color, fontsize=12)
    ax1.plot(time_axis_minutes, glucose_history, color=color, label='Blood Glucose', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(which='major', axis='y', linestyle='--', alpha=0.7)
    ax1.axhline(y=180, color='r', linestyle=':', label='Hyper Threshold (180)', linewidth=2)
    ax1.axhline(y=70, color='orange', linestyle=':', label='Hypo Threshold (70)', linewidth=2)
    ax1.fill_between(time_axis_minutes, 70, 180, alpha=0.1, color='green', label='Target Range')
    
    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Insulin Dose (U / 5min)', color=color, fontsize=12)
    ax2.bar(time_axis_minutes, eval_insulin_history, width=5, color=color, alpha=0.6, label='Insulin Dose')
    ax2.tick_params(axis='y', labelcolor=color)
    
    meal_labels_seen = set()
    for meal_time, meal_amount in zip(meal_times, meal_carbs):
        label = f'Meal ({meal_amount}g)'
        if label not in meal_labels_seen:
            ax1.axvline(x=meal_time, color='black', linestyle='--', label=label, linewidth=1.5)
            meal_labels_seen.add(label)
        else:
            ax1.axvline(x=meal_time, color='black', linestyle='--', linewidth=1.5)
    
    fig.suptitle(f'Ensemble SAC-TD3 Performance: {patient_name}', fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=10)
    plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Calculate evaluation metrics
    glucose_history = np.array(glucose_history)
    time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
    time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
    time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
    mean_glucose = np.mean(glucose_history)
    glucose_std = np.std(glucose_history)
    
    # Agent usage statistics
    sac_usage_pct = (agent_selections.count('sac') / len(agent_selections)) * 100
    td3_usage_pct = (agent_selections.count('td3') / len(agent_selections)) * 100

    print(f"\n{'='*60}")
    print(f"Evaluation Results for {patient_name}:")
    print(f"  Time in Range: {time_in_range:.2f}%")
    print(f"  Time Hypo: {time_hypo:.2f}%")
    print(f"  Time Hyper: {time_hyper:.2f}%")
    print(f"  Mean Glucose: {mean_glucose:.2f} mg/dL")
    print(f"  Glucose Std: {glucose_std:.2f} mg/dL")
    print(f"  SAC Usage: {sac_usage_pct:.1f}%")
    print(f"  TD3 Usage: {td3_usage_pct:.1f}%")
    print(f"{'='*60}\n")

    return {
        "eval_metrics": {
            "Patient": patient_name,
            "Mean Glucose (mg/dL)": mean_glucose,
            "Glucose Std (mg/dL)": glucose_std,
            "Time in Range (%)": time_in_range,
            "Time Hypo (%)": time_hypo,
            "Time Hyper (%)": time_hyper,
            "SAC Usage (%)": sac_usage_pct,
            "TD3 Usage (%)": td3_usage_pct,
        },
        "training_rewards": training_rewards_history
    }

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    hyperparameters = {
        'max_episodes': 250,
        'lr': 2e-4,
        'gamma_val': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'batch_size': 256,
        'n_latent_var': 256,
        'replay_buffer_size': 1000000,
        'max_timesteps_per_episode': 288,
        'learning_starts': 1500,
        'ETA': 4.0,
        'policy_delay': 2,
        'ensemble_weights': {'sac': 0.6, 'td3': 0.4}  # Adjustable weights
    }
    
    AGENT_NAME = 'ensemble_sac_td3_exp'
    results_dir = f'./results/{AGENT_NAME}'
    os.makedirs(results_dir, exist_ok=True)

    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
    tasks = []
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("This script requires at least one GPU.")

    for i, patient_name in enumerate(adult_patients):
        gpu_id = i % num_gpus
        tasks.append((patient_name, gpu_id, 42 + i, hyperparameters))

    print("\n" + "="*70)
    print(f"Starting Ensemble SAC-TD3 Training for {len(tasks)} patients on {num_gpus} GPUs")
    print("="*70)
    
    num_processes = min(len(tasks), mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(train_and_evaluate_patient, tasks)

    print("\n" + "="*70)
    print("--- ALL ENSEMBLE TRAINING PROCESSES COMPLETE ---")
    print("="*70)

    # 4. AGGREGATE AND DISPLAY RESULTS
    # =================================================================
    if results:
        eval_results = [res['eval_metrics'] for res in results if res and res.get('eval_metrics')]
        training_rewards_list = [res['training_rewards'] for res in results if res and res.get('training_rewards')]

        if eval_results:
            results_df = pd.DataFrame(eval_results)
            column_order = [
                "Patient", "Mean Glucose (mg/dL)", "Glucose Std (mg/dL)",
                "Time in Range (%)", "Time Hypo (%)", "Time Hyper (%)",
                "SAC Usage (%)", "TD3 Usage (%)"
            ]
            results_df = results_df[column_order]
            
            average_metrics = results_df.select_dtypes(include=np.number).mean()
            std_dev_metrics = results_df.select_dtypes(include=np.number).std()

            print("\n" + "="*80)
            print(f"--- ENSEMBLE SAC-TD3 EVALUATION SUMMARY ---")
            print("="*80)
            print("\n--- Performance per Patient ---")
            print(results_df.to_string(index=False))
            
            print("\n\n--- Average Performance Across All Patients ---")
            print(f"Mean Glucose (mg/dL):  {average_metrics['Mean Glucose (mg/dL)']:.2f} ± {std_dev_metrics['Mean Glucose (mg/dL)']:.2f}")
            print(f"Glucose Std (mg/dL):   {average_metrics['Glucose Std (mg/dL)']:.2f} ± {std_dev_metrics['Glucose Std (mg/dL)']:.2f}")
            print(f"Time in Range (%):     {average_metrics['Time in Range (%)']:.2f} ± {std_dev_metrics['Time in Range (%)']:.2f}")
            print(f"Time Hypo (%):         {average_metrics['Time Hypo (%)']:.2f} ± {std_dev_metrics['Time Hypo (%)']:.2f}")
            print(f"Time Hyper (%):        {average_metrics['Time Hyper (%)']:.2f} ± {std_dev_metrics['Time Hyper (%)']:.2f}")
            print(f"Average SAC Usage:     {average_metrics['SAC Usage (%)']:.2f}%")
            print(f"Average TD3 Usage:     {average_metrics['TD3 Usage (%)']:.2f}%")
            
            summary_path = os.path.join(results_dir, 'ensemble_summary.csv')
            results_df.to_csv(summary_path, index=False, float_format='%.2f')
            print(f"\n✅ Saved summary to {summary_path}")
            print("="*80)

            # Summary plot
            plot_data = results_df.set_index('Patient')
            time_metrics_df = plot_data[['Time in Range (%)', 'Time Hypo (%)', 'Time Hyper (%)']]
            ax = time_metrics_df.plot(
                kind='bar', stacked=True, figsize=(16, 9),
                color={'Time in Range (%)': 'green', 'Time Hypo (%)': 'orange', 'Time Hyper (%)': 'red'},
                edgecolor='black', width=0.8
            )
            plt.title('Ensemble SAC-TD3 Glycemic Control Summary', fontsize=18, weight='bold')
            plt.ylabel('Percentage of Time (%)', fontsize=14)
            plt.xlabel('Patient ID', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.ylim(0, 100)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.axhline(y=70, color='blue', linestyle='--', linewidth=2, label='70% TIR Target')
            plt.legend(title='Glycemic Range', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            summary_plot_path = os.path.join(results_dir, 'ensemble_summary_plot.png')
            plt.savefig(summary_plot_path, dpi=300)
            plt.close()
            print(f"✅ Saved summary plot to {summary_plot_path}")

            # Learning curve
            rewards_df = pd.DataFrame(training_rewards_list).T
            mean_rewards = rewards_df.mean(axis=1)
            std_rewards = rewards_df.std(axis=1)

            plt.figure(figsize=(12, 6))
            plt.plot(mean_rewards, label='Mean Episode Reward', color='blue', linewidth=2)
            plt.fill_between(
                rewards_df.index,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                color='blue',
                alpha=0.2,
                label='Standard Deviation'
            )
            plt.title('Ensemble SAC-TD3 Average Learning Curve', fontsize=16, weight='bold')
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Cumulative Reward', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            overall_lc_path = os.path.join(results_dir, 'ensemble_learning_curve.png')
            plt.savefig(overall_lc_path, dpi=300)
            plt.close()
            print(f"✅ Saved learning curve to {overall_lc_path}\n")