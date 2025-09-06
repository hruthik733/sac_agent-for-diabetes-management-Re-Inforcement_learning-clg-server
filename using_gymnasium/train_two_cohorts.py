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
from agents.sac_agent import SACAgent
from utils.replay_buffer import VectorizedReplayBuffer
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
    # ! WORKER COUNT INCREASED
    num_envs = 16
    total_timesteps = 1_000_000
    batch_size = 4096
    actor_lr = 3e-5
    critic_lr = 3e-4
    gamma_val = 0.99
    tau = 0.005
    n_latent_var = 512
    replay_buffer_size = total_timesteps
    learning_starts = batch_size * 4

    # --- 4. Define the two separate patient cohorts ---
    cohorts = {
        'adult': [f'adult#{i:03d}' for i in range(1, 11)],
        'adolescent': [f'adolescent#{i:03d}' for i in range(1, 11)]
    }

    # --- 5. Main loop to train a model for each cohort ---
    for cohort_name, patient_list in cohorts.items():
        print(f"\n\n===========================================================")
        print(f"   STARTING TRAINING FOR {cohort_name.upper()} COHORT")
        print(f"===========================================================")

        # --- Setup Directories for the current cohort ---
        model_dir = f'./models/sac_{cohort_name}_cohort'
        results_dir = f'./results/sac_{cohort_name}_cohort'
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        actor_path = f'{model_dir}/actor_{cohort_name}_cohort.pth'

        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())

        # --- Environment Registration for the current cohort ---
        meal_scenario = scgen.RandomScenario(start_time=start_time, seed=SEED)
        ENV_ID = f'simglucose/{cohort_name}-cohort-v0'
        try:
            register(
                id=ENV_ID,
                entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
                kwargs={"patient_name": patient_list, "custom_scenario": meal_scenario}
            )
        except gymnasium.error.Error:
            print(f"Environment {ENV_ID} already registered. Skipping.")

        envs = gymnasium.make_vec(ENV_ID, num_envs=num_envs, vectorization_mode="sync")

        state_dim = 4
        action_dim = envs.action_space.shape[1]
        max_action = float(envs.action_space.high[0])

        # --- Initialization for the current cohort ---
        agent = SACAgent(state_dim, action_dim, max_action, n_latent_var, actor_lr, critic_lr, gamma_val, tau, device)
        managers = [StateRewardManager(state_dim, cohort_name=cohort_name) for _ in range(num_envs)]
        safety_layer = SafetyLayer()
        replay_buffer = VectorizedReplayBuffer(num_envs, replay_buffer_size // num_envs, state_dim, action_dim, device)

        # --- Vectorized Training Loop for the current cohort ---
        print(f"--- Starting Vectorized Training with {num_envs} workers ---")
        obs_array, _ = envs.reset(seed=SEED)
        unnormalized_states = np.zeros((num_envs, state_dim))
        for i in range(num_envs):
            managers[i].reset()
            core_env = envs.unwrapped.envs[i].env.env
            episode_scenario = core_env.scenario
            current_sim_time = core_env.time
            upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
            unnormalized_states[i] = managers[i].get_full_state(obs_array[i], upcoming_carbs)
        
        current_states = np.array([managers[i].get_normalized_state(unnormalized_states[i]) for i in range(num_envs)])

        for step in tqdm(range(total_timesteps // num_envs)):
            if step * num_envs < learning_starts:
                actions = envs.action_space.sample()
            else:
                actions = agent.select_action(current_states)

            safe_actions = np.array([safety_layer.apply(actions[i], unnormalized_states[i]) for i in range(num_envs)])
            clipped_actions = np.clip(safe_actions, envs.action_space.low, envs.action_space.high)
            next_obs_array, _, terminated, truncated, infos = envs.step(clipped_actions)
            next_unnormalized_states = np.zeros((num_envs, state_dim))
            custom_rewards = np.zeros(num_envs)
            
            for i in range(num_envs):
                managers[i].insulin_history.append(clipped_actions[i][0])
                custom_rewards[i] = managers[i].get_reward(unnormalized_states[i])
                core_env = envs.unwrapped.envs[i].env.env
                episode_scenario = core_env.scenario
                current_sim_time = core_env.time
                upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
                next_unnormalized_states[i] = managers[i].get_full_state(next_obs_array[i], upcoming_carbs)

            next_states = np.array([managers[i].get_normalized_state(next_unnormalized_states[i]) for i in range(num_envs)])
            dones = np.logical_or(terminated, truncated)
            replay_buffer.push(current_states, clipped_actions, custom_rewards, next_states, dones)
            current_states = next_states
            unnormalized_states = next_unnormalized_states
            
            if '_final_info' in infos:
                for i, done in enumerate(dones):
                    if done:
                        managers[i].reset()
            
            if step * num_envs >= learning_starts:
                agent.update(replay_buffer, batch_size)

        envs.close()
        print(f"--- {cohort_name.capitalize()} Cohort Training Finished ---")
        torch.save(agent.actor.state_dict(), actor_path)
        print(f"Saved trained cohort model to {actor_path}")

        # --- Systematic Evaluation for the current cohort ---
        print(f"\n--- Starting Systematic Evaluation on {cohort_name.capitalize()} Cohort ---")
        all_patient_results = []
        eval_scenario = CustomScenario(start_time=start_time, scenario=[(7 * 60, 45), (12 * 60, 70), (18 * 60, 80)])
        
        eval_agent = SACAgent(state_dim, action_dim, max_action, n_latent_var, actor_lr, critic_lr, gamma_val, tau, device)
        eval_agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
        eval_agent.actor.eval()

        for patient_name in patient_list:
            print(f"--- Evaluating on Patient: {patient_name} ---")
            eval_env = gymnasium.make(ENV_ID, custom_scenario=eval_scenario, patient_name=patient_name)
            manager = StateRewardManager(state_dim, cohort_name=cohort_name)
            obs_array, info = eval_env.reset()
            episode_scenario = info.get('scenario')
            core_env = eval_env.unwrapped.env.env
            current_sim_time = core_env.time
            upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
            unnormalized_state = manager.get_full_state(obs_array, upcoming_carbs)
            current_state = manager.get_normalized_state(unnormalized_state)
            glucose_history = [obs_array[0]]
            
            for t in range(288):
                with torch.no_grad():
                    action = eval_agent.select_action(current_state)
                safe_action = safety_layer.apply(action, unnormalized_state)
                clipped_action = np.clip(safe_action, eval_env.action_space.low, eval_env.action_space.high)
                manager.insulin_history.append(clipped_action.item())
                obs_array, _, terminated, truncated, _ = eval_env.step(clipped_action)
                core_env = eval_env.unwrapped.env.env
                current_sim_time = core_env.time
                upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
                unnormalized_state = manager.get_full_state(obs_array, upcoming_carbs)
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
            all_patient_results.append({
                "Patient": patient_name, "Mean Glucose (mg/dL)": mean_glucose,
                "Time in Range (%)": time_in_range, "Time Hypo (%)": time_hypo,
                "Time Hyper (%)": time_hyper
            })

        # --- Final Summary for the current cohort ---
        print(f"\n\n========================================================")
        print(f"---    {cohort_name.upper()} COHORT OVERALL PERFORMANCE SUMMARY    ---")
        print(f"========================================================")
        results_df = pd.DataFrame(all_patient_results)
        results_df.set_index('Patient', inplace=True)
        print("\n--- Detailed Results Per Patient ---")
        print(results_df)
        average_performance = results_df.mean()
        print(f"\n--- Average Performance Across {cohort_name.capitalize()} Patients ---")
        print(average_performance.to_string())
        
        summary_csv_path = f'{results_dir}/cohort_evaluation_summary.csv'
        results_df.to_csv(summary_csv_path)
        print(f"\nSaved detailed summary results to {summary_csv_path}")

if __name__ == '__main__':
    main()















# import gymnasium
# from gymnasium.envs.registration import register
# import numpy as np
# import torch
# from datetime import datetime
# import matplotlib.pyplot as plt
# import os
# import random
# import pandas as pd
# from tqdm import tqdm

# from agents.sac_agent import SACAgent
# from utils.replay_buffer import VectorizedReplayBuffer
# from utils.safety import SafetyLayer
# from utils.new_state_management import StateRewardManager
# import simglucose.simulation.scenario_gen as scgen
# from simglucose.simulation.scenario import CustomScenario

# def main():
#     # --- 1. Set a master seed for reproducibility ---
#     SEED = 42
#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(SEED)

#     # --- 2. Device Configuration ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # --- 3. Hyperparameters ---
#     num_envs = 16
#     total_timesteps = 1_000_000
#     batch_size = 4096
#     actor_lr = 3e-5
#     critic_lr = 3e-4
#     gamma_val = 0.99
#     tau = 0.005
#     n_latent_var = 512
#     replay_buffer_size = total_timesteps
#     learning_starts = batch_size * 4
#     log_interval_episodes = 10
#     max_timesteps_per_episode = 288
#     state_dim = 12

#     cohorts = {
#         'adult': [f'adult#{i:03d}' for i in range(1, 11)],
#         'adolescent': [f'adolescent#{i:03d}' for i in range(1, 11)]
#     }

#     for cohort_name, patient_list in cohorts.items():
#         print(f"\n\n===========================================================")
#         print(f"   STARTING TRAINING FOR {cohort_name.upper()} COHORT")
#         print(f"===========================================================")

#         model_dir = f'./models/sac_{cohort_name}_cohort'
#         results_dir = f'./results/sac_{cohort_name}_cohort'
#         if not os.path.exists(model_dir): os.makedirs(model_dir)
#         if not os.path.exists(results_dir): os.makedirs(results_dir)
#         actor_path = f'{model_dir}/actor_{cohort_name}_cohort.pth'

#         now = datetime.now()
#         start_time = datetime.combine(now.date(), datetime.min.time())

#         meal_scenario = scgen.RandomScenario(start_time=start_time, seed=SEED)
#         ENV_ID = f'simglucose/{cohort_name}-cohort-v0'
#         try:
#             register(
#                 id=ENV_ID,
#                 entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
#                 max_episode_steps=max_timesteps_per_episode,
#                 kwargs={"patient_name": patient_list, "custom_scenario": meal_scenario}
#             )
#         except gymnasium.error.Error:
#             print(f"Environment {ENV_ID} already registered. Skipping.")

#         envs = gymnasium.make_vec(ENV_ID, num_envs=num_envs, vectorization_mode="sync")

#         action_dim = envs.action_space.shape[1]
#         max_action = float(envs.action_space.high[0])

#         agent = SACAgent(state_dim, action_dim, max_action, n_latent_var, actor_lr, critic_lr, gamma_val, tau, device)
#         managers = [StateRewardManager(state_dim, cohort_name=cohort_name) for _ in range(num_envs)]
#         safety_layer = SafetyLayer()
#         replay_buffer = VectorizedReplayBuffer(num_envs, replay_buffer_size // num_envs, state_dim, action_dim, device)

#         print(f"--- Starting Vectorized Training with {num_envs} workers ---")
#         obs_array, _ = envs.reset(seed=SEED)
        
#         episode_rewards = [0.0] * num_envs
#         episode_lengths = [0] * num_envs
#         episode_glucose_histories = [[] for _ in range(num_envs)]
#         completed_episodes_count = 0
        
#         unnormalized_states = np.zeros((num_envs, state_dim))
#         for i in range(num_envs):
#             managers[i].reset()
#             core_env = envs.unwrapped.envs[i].env.env
#             # ! FIX: Access scenario from the correct wrapper
#             episode_scenario = envs.unwrapped.envs[i].env._custom_scenario
#             current_sim_time = core_env.time
#             upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
#             unnormalized_states[i] = managers[i].get_full_state(obs_array[i], current_sim_time, upcoming_carbs)
#             episode_glucose_histories[i].append(unnormalized_states[i][0])

#         current_states = np.array([managers[i].get_normalized_state(unnormalized_states[i]) for i in range(num_envs)])

#         for step in tqdm(range(total_timesteps // num_envs)):
#             if step * num_envs < learning_starts:
#                 actions = envs.action_space.sample()
#             else:
#                 actions = agent.select_action(current_states)

#             safe_actions = np.array([safety_layer.apply(actions[i], unnormalized_states[i]) for i in range(num_envs)])
#             clipped_actions = np.clip(safe_actions, envs.action_space.low, envs.action_space.high)
#             next_obs_array, _, terminated, truncated, infos = envs.step(clipped_actions)
            
#             next_unnormalized_states = np.zeros((num_envs, state_dim))
#             custom_rewards = np.zeros(num_envs)
            
#             for i in range(num_envs):
#                 managers[i].insulin_history.append(clipped_actions[i][0])
#                 custom_rewards[i] = managers[i].get_reward(unnormalized_states[i])
#                 core_env = envs.unwrapped.envs[i].env.env
#                 # ! FIX: Access scenario from the correct wrapper
#                 episode_scenario = envs.unwrapped.envs[i].env._custom_scenario
#                 current_sim_time = core_env.time
#                 upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
#                 next_unnormalized_states[i] = managers[i].get_full_state(next_obs_array[i], current_sim_time, upcoming_carbs)
                
#                 episode_rewards[i] += custom_rewards[i]
#                 episode_lengths[i] += 1
#                 episode_glucose_histories[i].append(next_unnormalized_states[i][0])

#             next_states = np.array([managers[i].get_normalized_state(next_unnormalized_states[i]) for i in range(num_envs)])
#             dones = np.logical_or(terminated, truncated)
#             replay_buffer.push(current_states, clipped_actions, custom_rewards, next_states, dones)
            
#             current_states = next_states
#             unnormalized_states = next_unnormalized_states
            
#             if '_final_info' in infos:
#                 for i, done in enumerate(dones):
#                     if done:
#                         completed_episodes_count += 1
#                         glucose_history = np.array(episode_glucose_histories[i])
#                         tir = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100 if len(glucose_history) > 0 else 0
#                         patient_name = infos['final_info'][i].get('patient_name', 'Unknown')
                        
#                         if completed_episodes_count % log_interval_episodes == 0:
#                             tqdm.write(f"\nEpisode {completed_episodes_count} | Patient: {patient_name} | Length: {episode_lengths[i]} | Reward: {episode_rewards[i]:.2f} | TIR: {tir:.2f}%")

#                         episode_rewards[i] = 0.0
#                         episode_lengths[i] = 0
#                         episode_glucose_histories[i] = []
#                         managers[i].reset()
            
#             if step * num_envs >= learning_starts:
#                 agent.update(replay_buffer, batch_size)

#         envs.close()
        
#         print(f"--- {cohort_name.capitalize()} Cohort Training Finished ---")
#         agent.save(actor_path)
#         print(f"Saved trained cohort model to {actor_path}")

#         # --- Systematic Evaluation for the current cohort ---
#         print(f"\n--- Starting Systematic Evaluation on {cohort_name.capitalize()} Cohort ---")
#         all_patient_results = []
#         eval_scenario = CustomScenario(start_time=start_time, scenario=[(7 * 60, 45), (12 * 60, 70), (18 * 60, 80)])
        
#         eval_agent = SACAgent(state_dim, action_dim, max_action, n_latent_var, actor_lr, critic_lr, gamma_val, tau, device)
#         eval_agent.load(actor_path)
#         eval_agent.actor.eval()

#         for patient_name in patient_list:
#             print(f"--- Evaluating on Patient: {patient_name} ---")
#             eval_env = gymnasium.make(ENV_ID, custom_scenario=eval_scenario, patient_name=patient_name)
#             manager = StateRewardManager(state_dim, cohort_name=cohort_name)
#             obs_array, info = eval_env.reset()
#             # ! FIX: Access scenario from the correct wrapper
#             episode_scenario = eval_env.unwrapped.env._custom_scenario
#             core_env = eval_env.unwrapped.env.env
#             current_sim_time = core_env.time
#             upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
#             unnormalized_state = manager.get_full_state(obs_array, current_sim_time, upcoming_carbs)
#             current_state = manager.get_normalized_state(unnormalized_state)
#             glucose_history = [unnormalized_state[0]]
            
#             for t in range(288):
#                 with torch.no_grad():
#                     action = eval_agent.select_action(current_state)
#                 safe_action = safety_layer.apply(action, unnormalized_state)
#                 clipped_action = np.clip(safe_action, eval_env.action_space.low, eval_env.action_space.high)
#                 manager.insulin_history.append(clipped_action.item())
#                 obs_array, _, terminated, truncated, _ = eval_env.step(clipped_action)
#                 core_env = eval_env.unwrapped.env.env
#                 # ! FIX: Access scenario from the correct wrapper
#                 episode_scenario = eval_env.unwrapped.env._custom_scenario
#                 current_sim_time = core_env.time
#                 upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
#                 unnormalized_state = manager.get_full_state(obs_array, current_sim_time, upcoming_carbs)
#                 current_state = manager.get_normalized_state(unnormalized_state)
#                 glucose_history.append(unnormalized_state[0])
#                 if terminated or truncated:
#                     break
            
#             eval_env.close()

#             glucose_history = np.array(glucose_history)
#             time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
#             time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
#             time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
#             mean_glucose = np.mean(glucose_history)
#             all_patient_results.append({
#                 "Patient": patient_name, "Mean Glucose (mg/dL)": mean_glucose,
#                 "Time in Range (%)": time_in_range, "Time Hypo (%)": time_hypo,
#                 "Time Hyper (%)": time_hyper
#             })

#         # --- Final Summary for the current cohort ---
#         print(f"\n\n========================================================")
#         print(f"---    {cohort_name.upper()} COHORT OVERALL PERFORMANCE SUMMARY    ---")
#         print(f"========================================================")
#         results_df = pd.DataFrame(all_patient_results)
#         results_df.set_index('Patient', inplace=True)
#         print("\n--- Detailed Results Per Patient ---")
#         print(results_df)
#         average_performance = results_df.mean()
#         print(f"\n--- Average Performance Across {cohort_name.capitalize()} Patients ---")
#         print(average_performance.to_string())
        
#         summary_csv_path = f'{results_dir}/cohort_evaluation_summary.csv'
#         results_df.to_csv(summary_csv_path)
#         print(f"\nSaved detailed summary results to {summary_csv_path}")

# if __name__ == '__main__':
#     main()