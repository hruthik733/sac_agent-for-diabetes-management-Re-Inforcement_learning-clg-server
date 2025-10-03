# train_parallel.py

import os
import gymnasium
from gymnasium.envs.registration import register
from datetime import datetime
import multiprocessing as mp
import stable_baselines3

# Correct import for the scenario generator
import simglucose.simulation.scenario_gen as scgen
from simglucose.envs.simglucose_gym_env import T1DSimGymnaisumEnv


def train_worker(model_class_name, patient_list, gpu_id):
    """
    This function is the core training job. It will be run in a separate process.
    It takes the model name (e.g., 'SAC'), a list of patients, and the GPU ID to use.
    """
    # CRITICAL: Set the visible GPU for this specific process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"--- [{model_class_name}] Worker process started on GPU {gpu_id} for {len(patient_list)} patients ---")

    model_class = getattr(stable_baselines3, model_class_name)
    
    for patient_name in patient_list:
        print(f"--- [{model_class_name} on GPU {gpu_id}] Starting Training for {patient_name} ---")

        # --- Environment Setup ---
        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())
        seed = abs(hash(f"{model_class_name}_{patient_name}")) % (2**32)
        
        # MODIFIED: Correctly use the imported 'scgen' module
        training_scenario = scgen.RandomScenario(start_time=start_time, seed=seed)

        env_id = f'simglucose/{model_class_name.lower()}-{patient_name.replace("#", "-")}-v0'
        try:
            register(
                id=env_id,
                entry_point='simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv',
                kwargs={'patient_name': patient_name, 'custom_scenario': training_scenario}
            )
        except gymnasium.error.Error:
            pass
        
        env = gymnasium.make(env_id, max_episode_steps=288)

        # --- Model and Log Directories ---
        model_dir = f'./models/{model_class_name.lower()}_baseline'
        log_dir = f'./logs/{model_class_name.lower()}_baseline'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # --- Instantiate and Train the Model ---
        if model_class_name == 'SAC':
            model = model_class("MlpPolicy", env, verbose=0, tensorboard_log=log_dir,
                                learning_starts=1500, batch_size=256, gamma=0.99,
                                tau=0.005, device="cuda")
        elif model_class_name == 'PPO':
             model = model_class("MlpPolicy", env, verbose=0, tensorboard_log=log_dir,
                                gamma=0.99, device="cuda")
        else:
            raise ValueError(f"Model class {model_class_name} not supported.")

        training_timesteps = 200 * 288 
        model.learn(total_timesteps=training_timesteps, log_interval=10, tb_log_name=patient_name)

        save_path = os.path.join(model_dir, f"model_{patient_name.replace('#', '-')}")
        model.save(save_path)
        print(f"--- [{model_class_name} on GPU {gpu_id}] Model for {patient_name} saved to {save_path} ---")
        env.close()

    print(f"--- [{model_class_name}] Worker on GPU {gpu_id} finished. ---")


if __name__ == '__main__':
    # It's important to set the start method to 'spawn' for CUDA applications
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    all_patients = [f'adult#{i:03d}' for i in range(1, 11)]
    
    # Define the jobs to run
    jobs = [
        ('SAC', all_patients, 0),
        ('PPO', all_patients, 1)
    ]

    processes = []
    for model_name, patient_list, gpu_id in jobs:
        process = mp.Process(target=train_worker, args=(model_name, patient_list, gpu_id))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("\n=======================================================")
    print("All parallel training jobs have completed successfully!")
    print("=======================================================")