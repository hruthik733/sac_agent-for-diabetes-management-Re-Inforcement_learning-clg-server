# train_all_adults.py

import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
import pandas as pd

from agents.sac_agent import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.safety import SafetyLayer
from utils.state_management import StateRewardManager
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario

def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Print GPU info
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU available. Exiting.")
        return

    max_episodes = 2000
    lr = 3e-4
    gamma_val = 0.99
    tau = 0.005
    alpha = 0.2
    batch_size = 256
    n_latent_var = 256
    replay_buffer_size = 1000000
    max_timesteps_per_episode = 288
    learning_starts = 1000

    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]

    all_patient_results = []
    for patient_name in adult_patients:
        print(f"\n{'='*50}\n--- Starting Training for {patient_name} ---\n{'='*50}")

        AGENT_NAME = 'sac'
        model_dir = f'./models/{AGENT_NAME}'
        results_dir = f'./results/{AGENT_NAME}'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        actor_path = f'{model_dir}/actor_{patient_name.replace("#", "-")}.pth'

        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())

        meal_scenario = scgen.RandomScenario(start_time=start_time, seed=SEED)
        clean_patient_name = patient_name.replace('#', '-')
        env_id = f'simglucose/{clean_patient_name}-v0'

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
        env.action_space.seed(SEED)

        state_dim = 4
        action_dim = 1

        agent = SACAgent(env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
        manager = StateRewardManager(state_dim)
        safety_layer = SafetyLayer()
        replay_buffer = ReplayBuffer(replay_buffer_size)

        episode_lengths, final_glucose_levels = [], []
        total_timesteps_taken = 0

        for i_episode in range(1, max_episodes + 1):
            obs_array, info = env.reset(seed=SEED + i_episode)
            episode_scenario = info.get('scenario')
            manager.reset()

            current_sim_time = env.unwrapped.env.env.time
            upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0

            unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
            current_state = manager.get_normalized_state(unnormalized_state)
            episode_reward = 0

            for t in range(max_timesteps_per_episode):
                if total_timesteps_taken < learning_starts:
                    action = np.array([np.random.uniform(0, 0.5)])
                else:
                    action = agent.select_action(current_state)

                safe_action = safety_layer.apply(action, unnormalized_state)
                clipped_action = np.clip(safe_action, 0, 5.0)

                manager.insulin_history.append(clipped_action[0])
                next_obs_array, _, terminated, truncated, _ = env.step(clipped_action)
                done = terminated or truncated

                current_sim_time = env.unwrapped.env.env.time
                upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
                next_unnormalized_state = manager.get_full_state(next_obs_array[0], upcoming_carbs)
                next_state = manager.get_normalized_state(next_unnormalized_state)

                reward = manager.get_reward(unnormalized_state)
                replay_buffer.push(current_state, clipped_action, reward, next_state, done)

                current_state = next_state
                unnormalized_state = next_unnormalized_state
                episode_reward += reward
                total_timesteps_taken += 1

                if total_timesteps_taken > learning_starts and len(replay_buffer) > batch_size:
                    agent.update(replay_buffer, batch_size)

                if done:
                    break

            episode_lengths.append(t + 1)
            final_glucose_levels.append(unnormalized_state[0])

            if i_episode % 50 == 0:
                print(f"Episode {i_episode}/{max_episodes} | Length: {t+1} | Reward: {episode_reward:.2f}")

        torch.save(agent.actor.state_dict(), actor_path)
        print(f"Saved model to {actor_path}")

        eval_scenario = CustomScenario(start_time=start_time, scenario=[(7*60, 45), (12*60, 70), (18*60, 80)])
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)

        eval_agent = SACAgent(eval_env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
        eval_agent.actor.load_state_dict(torch.load(actor_path))

        manager.reset()
        obs_array, info = eval_env.reset()
        episode_scenario = info.get('scenario')

        current_sim_time = eval_env.unwrapped.env.env.time
        upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
        unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history = [obs_array[0]]

        for t in range(max_timesteps_per_episode):
            action = eval_agent.select_action(current_state)
            safe_action = safety_layer.apply(action, unnormalized_state)
            clipped_action = np.clip(safe_action, 0, 5.0)

            manager.insulin_history.append(clipped_action[0])
            obs_array, _, terminated, truncated, _ = eval_env.step(clipped_action)

            current_sim_time = eval_env.unwrapped.env.env.time
            upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
            unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)

            current_state = manager.get_normalized_state(unnormalized_state)
            glucose_history.append(obs_array[0])
            if terminated or truncated:
                break

        eval_env.close()

        glucose_history = np.array(glucose_history)
        time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
        time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
        time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
        mean_glucose = np.mean(glucose_history)

        print(f"Mean Glucose: {mean_glucose:.2f} mg/dL")
        print(f"Time in Range: {time_in_range:.2f}% | Hypo: {time_hypo:.2f}% | Hyper: {time_hyper:.2f}%")

        patient_summary = {
            "Patient": patient_name,
            "Mean Glucose (mg/dL)": mean_glucose,
            "Time in Range (%)": time_in_range,
            "Time Hypo (%)": time_hypo,
            "Time Hyper (%)": time_hyper,
        }
        all_patient_results.append(patient_summary)

        plt.figure(figsize=(15, 6))
        plt.plot(glucose_history, label='SAC Agent')
        plt.axhline(y=180, color='r', linestyle=':', label='Hyperglycemia Threshold')
        plt.axhline(y=70, color='orange', linestyle=':', label='Hypoglycemia Threshold')
        plt.axhline(y=140, color='g', linestyle='-', label='Target')
        plt.title(f'SAC Agent Performance for {patient_name}')
        plt.xlabel('Time (minutes)'); plt.ylabel('Blood Glucose (mg/dL)')
        plt.legend(); plt.grid(True)

        plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved evaluation plot to {plot_path}")

    results_df = pd.DataFrame(all_patient_results)
    results_df.set_index('Patient', inplace=True)

    print("\n--- Results Per Patient ---")
    print(results_df)

    print("\n--- Average Performance ---")
    print(results_df.mean().to_string())

if __name__ == '__main__':
    main()
