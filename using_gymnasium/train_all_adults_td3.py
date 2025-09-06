import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from tqdm import tqdm

# --- Local Imports from our modules ---
from agents.td3_agent import TD3Agent
from utils.simple_replay_buffer import ReplayBuffer # Using the simple, non-vectorized buffer
from utils.safety import SafetyLayer
from utils.new_state_management import StateRewardManager
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario

def main():
    # --- 1. Set a master seed for reproducibility ---
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --- 2. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Hyperparameters ---
    max_episodes = 5000
    batch_size = 512
    learning_starts = 2500
    state_dim = 12
    
    lr = 3e-4
    gamma_val = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    n_latent_var = 256
    expl_noise_std = 0.1
    max_timesteps_per_episode = 288
    replay_buffer_size = 1_000_000

    # --- 4. Define the patient cohort ---
    PATIENT_COHORT = [f'adult#{i:03d}' for i in range(1, 11)]

    # --- 5. Setup Directories ---
    model_dir = './models/td3_adult_cohort_single'
    results_dir = './results/td3_adult_cohort_single'
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    actor_path = f'{model_dir}/actor_adult_cohort.pth'

    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    # --- 6. Environment Registration and Creation ---
    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=SEED)
    ENV_ID = 'simglucose/adult-cohort-v0'
    try:
        register(id=ENV_ID, entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps_per_episode, kwargs={"patient_name": PATIENT_COHORT, "custom_scenario": meal_scenario})
    except gymnasium.error.Error:
        print(f"Environment {ENV_ID} already registered.")

    # ! Use a single environment, NOT make_vec
    env = gymnasium.make(ENV_ID)
    
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --- 7. Initialization ---
    agent = TD3Agent(state_dim, action_dim, max_action, n_latent_var, lr, gamma_val, tau, policy_noise, noise_clip, policy_freq, device)
    manager = StateRewardManager(state_dim, cohort_name='adult')
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(replay_buffer_size, device)
    total_timesteps = 0

    # --- 8. Serial Training Loop ---
    print("--- Starting Single Environment Cohort Training ---")
    for i_episode in range(1, max_episodes + 1):
        obs_array, info = env.reset(seed=SEED + i_episode)
        patient_name = info.get('patient_name', 'Unknown')
        
        manager.reset()
        
        core_env = env.unwrapped.env.env
        episode_scenario = env.unwrapped.env.custom_scenario
        current_sim_time = core_env.time
        upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
        unnormalized_state = manager.get_full_state(obs_array, current_sim_time, upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        
        episode_reward = 0
        glucose_history = [unnormalized_state[0]]

        for t in range(max_timesteps_per_episode):
            total_timesteps += 1
            if total_timesteps < learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.select_action(current_state)
                noise = np.random.normal(0, max_action * expl_noise_std, size=action_dim)
                action = (action + noise).clip(env.action_space.low, env.action_space.high)
            
            action = action.flatten()
            safe_action = safety_layer.apply(action, unnormalized_state)
            clipped_action = np.clip(safe_action, env.action_space.low, env.action_space.high)
            
            manager.insulin_history.append(clipped_action.item())
            next_obs_array, _, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated

            core_env = env.unwrapped.env.env
            current_sim_time = core_env.time
            episode_scenario = env.unwrapped.env.custom_scenario
            upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
            next_unnormalized_state = manager.get_full_state(next_obs_array, current_sim_time, upcoming_carbs)
            next_state = manager.get_normalized_state(next_unnormalized_state)

            reward = manager.get_reward(unnormalized_state)
            replay_buffer.push(current_state, clipped_action, reward, next_state, done)
            
            current_state = next_state
            unnormalized_state = next_unnormalized_state
            episode_reward += reward
            glucose_history.append(unnormalized_state[0])

            if total_timesteps >= learning_starts:
                agent.update(replay_buffer, batch_size)

            if done:
                break
        
        tir = np.sum((np.array(glucose_history) >= 70) & (np.array(glucose_history) <= 180)) / len(glucose_history) * 100
        print(f"Episode {i_episode} | Patient: {patient_name} | Length: {t+1} | Reward: {episode_reward:.2f} | TIR: {tir:.2f}%")

    env.close()
    
    print("--- Adult Cohort Training Finished ---")
    agent.save(actor_path)
    print(f"Saved trained cohort model to {actor_path}")

    # --- 9. Systematic Evaluation ---
    # (The evaluation code is already serial and correct)

    # --- 9. Systematic Evaluation ---
    print("\n--- Starting Systematic Evaluation on Adult Cohort ---")
    all_patient_results = []
    eval_scenario = CustomScenario(start_time=start_time, scenario=[(7 * 60, 45), (12 * 60, 70), (18 * 60, 80)])
    
    eval_agent = TD3Agent(state_dim, action_dim, max_action, n_latent_var, lr, gamma_val, tau, policy_noise, noise_clip, policy_freq, device)
    eval_agent.load(actor_path)
    eval_agent.actor.eval()

    for patient_name in PATIENT_COHORT:
        print(f"--- Evaluating on Patient: {patient_name} ---")
        eval_env = gymnasium.make(ENV_ID, custom_scenario=eval_scenario, patient_name=patient_name)
        manager = StateRewardManager(state_dim, cohort_name='adult')
        obs_array, info = eval_env.reset()
        
        core_env = eval_env.unwrapped.env.env
        # ! FIX: Access scenario from the correct object
        episode_scenario = eval_env.unwrapped.env.custom_scenario
        current_sim_time = core_env.time
        upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
        unnormalized_state = manager.get_full_state(obs_array, current_sim_time, upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history = [unnormalized_state[0]]
        
        for t in range(max_timesteps_per_episode):
            with torch.no_grad():
                action = eval_agent.select_action(current_state)

            safe_action = safety_layer.apply(action, unnormalized_state)
            clipped_action = np.clip(safe_action, eval_env.action_space.low, eval_env.action_space.high)
            manager.insulin_history.append(clipped_action.item())
            obs_array, _, terminated, truncated, _ = eval_env.step(clipped_action)
            
            core_env = eval_env.unwrapped.env.env
            # ! FIX: Access scenario from the correct object
            episode_scenario = eval_env.unwrapped.env.custom_scenario
            current_sim_time = core_env.time
            upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
            unnormalized_state = manager.get_full_state(obs_array, current_sim_time, upcoming_carbs)
            current_state = manager.get_normalized_state(unnormalized_state)
            glucose_history.append(unnormalized_state[0])
            if terminated or truncated:
                break
        
        eval_env.close()

        glucose_history = np.array(glucose_history)
        time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
        time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
        time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
        mean_glucose = np.mean(glucose_history)
        all_patient_results.append({
            "Patient": patient_name, "Mean Glucose (mg/dL)": mean_glucose,
            "Time in Range (%)": time_in_range, "Time Hypo (%)": time_hypo,
            "Time Hyper (%)": time_hyper
        })

    # --- 10. Final Summary ---
    print("\n\n========================================================")
    print("---    ADULT COHORT OVERALL PERFORMANCE SUMMARY    ---")
    print("========================================================")
    results_df = pd.DataFrame(all_patient_results)
    results_df.set_index('Patient', inplace=True)
    print("\n--- Detailed Results Per Patient ---")
    print(results_df)
    average_performance = results_df.mean()
    print("\n--- Average Performance Across Adult Patients ---")
    print(average_performance.to_string())
    
    summary_csv_path = f'{results_dir}/cohort_evaluation_summary.csv'
    results_df.to_csv(summary_csv_path)
    print(f"\nSaved detailed summary results to {summary_csv_path}")

if __name__ == '__main__':
    main()