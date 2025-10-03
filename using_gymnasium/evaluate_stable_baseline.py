# evaluate_stable_baselines.py

import os
import gymnasium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
from datetime import datetime
import stable_baselines3

from simglucose.simulation.scenario import CustomScenario
from simglucose.envs.simglucose_gym_env import T1DSimGymnaisumEnv

def evaluate_model(model_class_name):
    """
    Loads and evaluates all trained models for a given algorithm (e.g., 'SAC' or 'PPO').
    """
    # Dynamically set paths based on the model class name
    model_dir = f'./models/{model_class_name.lower()}_baseline'
    results_dir = f'./results/{model_class_name.lower()}_baseline'
    os.makedirs(results_dir, exist_ok=True)

    # == 1. SET UP THE EVALUATION SCENARIO (MUST BE IDENTICAL TO OTHER EVALUATIONS) ==
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    meal_times = [7 * 60, 12 * 60, 18 * 60]
    meal_carbs = [45, 70, 80]
    eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
    
    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
    all_patient_results = []

    # Dynamically get the model class from the stable_baselines3 library
    model_class = getattr(stable_baselines3, model_class_name)

    # == 2. LOOP THROUGH PATIENTS, LOAD MODEL, AND EVALUATE ==
    for patient_name in adult_patients:
        print(f"--- Evaluating Stable-Baseline {model_class_name} for {patient_name} ---")

        # Load the trained model using the correct path and filename
        model_path = os.path.join(model_dir, f"model_{patient_name.replace('#', '-')}.zip")
        if not os.path.exists(model_path):
            print(f"Model for {patient_name} not found at {model_path}. Skipping.")
            continue
        model = model_class.load(model_path)

        # Create the evaluation environment with the correct ID
        env_id = f'simglucose/{model_class_name.lower()}-{patient_name.replace("#", "-")}-v0'
        try:
            register(id=env_id, entry_point=T1DSimGymnaisumEnv, 
                     kwargs={'patient_name': patient_name, 'custom_scenario': eval_scenario})
        except gymnasium.error.Error:
            pass
        env = gymnasium.make(env_id, max_episode_steps=288)

        # Run the evaluation loop
        obs, info = env.reset()
        # BUG FIX: The first observation is the first element of `obs`, not in `info`.
        glucose_history = [obs[0]]
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # The new glucose value is in the returned observation
            glucose_history.append(obs[0])
        
        env.close()

        # == 3. CALCULATE AND STORE METRICS ==
        glucose_history = np.array(glucose_history)
        time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
        time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
        time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
        mean_glucose = np.mean(glucose_history)

        all_patient_results.append({
            "Patient": patient_name,
            "Mean Glucose (mg/dL)": mean_glucose,
            "Time in Range (%)": time_in_range,
            "Time Hypo (%)": time_hypo,
            "Time Hyper (%)": time_hyper,
        })

    # == 4. AGGREGATE, DISPLAY, AND PLOT FINAL RESULTS ==
    if not all_patient_results:
        print(f"No results found for {model_class_name}. Make sure models are trained and paths are correct.")
        return
        
    results_df = pd.DataFrame(all_patient_results)
    print("\n" + "="*70)
    print(f"--- OVERALL STABLE-BASELINE {model_class_name} SUMMARY ---")
    print("="*70)
    print("\n--- Performance per Patient ---")
    print(results_df.to_string(index=False))

    average_metrics = results_df.select_dtypes(include=np.number).mean()
    print("\n\n--- Average Performance Across All Patients ---")
    print(f"Mean Glucose (mg/dL): {average_metrics['Mean Glucose (mg/dL)']:.2f}")
    print(f"Time in Range (%):    {average_metrics['Time in Range (%)']:.2f}")
    print(f"Time Hypo (%):        {average_metrics['Time Hypo (%)']:.2f}")
    print(f"Time Hyper (%):       {average_metrics['Time Hyper (%)']:.2f}")

    summary_path = os.path.join(results_dir, f'{model_class_name.lower()}_summary.csv')
    results_df.to_csv(summary_path, index=False, float_format='%.2f')
    print(f"\n✅ Saved {model_class_name} summary table to {summary_path}")
    
    # Generate and save the summary plot
    plot_data = results_df.set_index('Patient')
    time_metrics_df = plot_data[['Time in Range (%)', 'Time Hypo (%)', 'Time Hyper (%)']]
    ax = time_metrics_df.plot(
        kind='bar', stacked=True, figsize=(16, 9),
        color={'Time in Range (%)': 'green', 'Time Hypo (%)': 'orange', 'Time Hyper (%)': 'red'},
        edgecolor='black', width=0.8)
    plt.title(f'Glycemic Control Summary ({model_class_name} Baseline)', fontsize=18, weight='bold')
    plt.ylabel('Percentage of Time (%)', fontsize=14)
    plt.xlabel('Patient ID', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=70, color='blue', linestyle='--', label='70% Time in Range Target')
    plt.legend(title='Glycemic Range', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    summary_plot_path = os.path.join(results_dir, f'{model_class_name.lower()}_summary_plot.png')
    plt.savefig(summary_plot_path)
    plt.close()
    print(f"✅ Saved {model_class_name} summary plot to {summary_plot_path}\n")


if __name__ == '__main__':
    # Evaluate all the models you have trained
    evaluate_model("SAC")
    evaluate_model("PPO")
    
    print("=======================================================")
    print("All baseline model evaluations are complete!")
    print("=======================================================")