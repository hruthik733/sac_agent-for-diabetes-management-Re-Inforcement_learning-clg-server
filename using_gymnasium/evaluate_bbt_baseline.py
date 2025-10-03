import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from simglucose.envs.simglucose_gym_env import T1DSimGymnaisumEnv
from simglucose.simulation.scenario import CustomScenario

def run_bbt_evaluation():
    """
    Main function to run a generic BBT baseline evaluation for all adult patients.
    """
    results_dir = './results/bbt_baseline'
    os.makedirs(results_dir, exist_ok=True)

    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    meal_times = [7 * 60, 12 * 60, 18 * 60]
    meal_carbs = [45, 70, 80]
    eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))

    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
    all_patient_results = []
    
    target_glucose = 120

    for patient_name in adult_patients:
        print(f"--- Evaluating BBT for {patient_name} ---")

        env = T1DSimGymnaisumEnv(patient_name=patient_name, custom_scenario=eval_scenario)
        
        core_env = env.unwrapped.env.env
        patient = core_env.patient
        
        # Get the personalized basal rate from the patient file
        basal_rate_U_per_hour = patient._params['u2ss']
        
        # WORKAROUND: Use generic, hard-coded CR and CF values
        # The patient data file is missing personalized CR and CF.
        icr = 15.0  # Generic Carb Ratio (g/U)
        isf = 50.0  # Generic Correction Factor (mg/dL/U)
        
        basal_per_5min = basal_rate_U_per_hour / 12

        obs, info = env.reset()
        glucose_history = [obs[0]]
        
        for t in range(288):
            current_glucose = glucose_history[-1]
            bolus_dose = 0.0

            meal_carbs_at_step = core_env.scenario.get_action(core_env.time).meal
            
            if meal_carbs_at_step > 0:
                carb_bolus = meal_carbs_at_step / icr
                
                correction_bolus = 0.0
                if current_glucose > target_glucose:
                    correction_bolus = (current_glucose - target_glucose) / isf
                
                bolus_dose = carb_bolus + correction_bolus

            action = np.array([basal_per_5min + bolus_dose])
            
            obs, reward, terminated, truncated, info = env.step(action)
            glucose_history.append(obs[0])
            
            if terminated or truncated:
                break
        
        env.close()

        # --- 3. CALCULATE AND STORE METRICS ---
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

    # --- 4. AGGREGATE, DISPLAY, AND PLOT FINAL RESULTS ---
    results_df = pd.DataFrame(all_patient_results)
    
    print("\n" + "="*70)
    print("--- OVERALL GENERIC BBT BASELINE SUMMARY ---")
    print("="*70)
    print("\n--- Performance per Patient ---")
    print(results_df.to_string(index=False))

    average_metrics = results_df.select_dtypes(include=np.number).mean()
    print("\n\n--- Average Performance Across All Patients ---")
    print(f"Mean Glucose (mg/dL): {average_metrics['Mean Glucose (mg/dL)']:.2f}")
    print(f"Time in Range (%):    {average_metrics['Time in Range (%)']:.2f}")
    print(f"Time Hypo (%):        {average_metrics['Time Hypo (%)']:.2f}")
    print(f"Time Hyper (%):       {average_metrics['Time Hyper (%)']:.2f}")

    summary_path = os.path.join(results_dir, 'bbt_summary.csv')
    results_df.to_csv(summary_path, index=False, float_format='%.2f')
    print(f"\n✅ Saved BBT summary table to {summary_path}")

    # Generate and save the summary plot
    plot_data = results_df.set_index('Patient')
    time_metrics_df = plot_data[['Time in Range (%)', 'Time Hypo (%)', 'Time Hyper (%)']]
    ax = time_metrics_df.plot(
        kind='bar', stacked=True, figsize=(16, 9),
        color={'Time in Range (%)': 'green', 'Time Hypo (%)': 'orange', 'Time Hyper (%)': 'red'},
        edgecolor='black', width=0.8)
    plt.title('Glycemic Control Summary (Generic BBT Baseline)', fontsize=18, weight='bold')
    plt.ylabel('Percentage of Time (%)', fontsize=14)
    plt.xlabel('Patient ID', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=70, color='blue', linestyle='--', label='70% Time in Range Target')
    plt.legend(title='Glycemic Range', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    summary_plot_path = os.path.join(results_dir, 'bbt_summary_plot.png')
    plt.savefig(summary_plot_path)
    plt.close()
    print(f"✅ Saved BBT summary plot to {summary_plot_path}")

if __name__ == '__main__':
    run_bbt_evaluation()