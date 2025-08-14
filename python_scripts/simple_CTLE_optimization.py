import os
import ocean_creation
import matplotlib.pyplot as plt
import numpy as np
import csv
import optuna
import subprocess

import time
import argparse

# --- Define Domain ---
fW_RANGE = (1e-6, 10e-6)
current_RANGE = (0.5e-3, 2.5e-3)
ind_RANGE = (0, 3e-9)
Rd_RANGE = (10, 1500)
Cs_RANGE = (1e-15, 1000e-15)
Rs_RANGE = (10, 1500)

def run_ocean_simulation(simulation_filename):
    start_time = time.time()
    # Expand the Ocean script path
    script_path = os.path.expanduser(simulation_filename)
    delete_filepath_1 = os.path.expanduser(f"~/workarea_GF22_FDX_EXT/ECE740/tb_channel_CTLE_ML_{args.run_id}/maestro/history/*")
    delete_filepath_2 = os.path.expanduser(f"~/simulation/ECE740/tb_channel_CTLE_ML_{args.run_id}/maestro/history/*")
    # Compose the shell command with a here-document
    if args.del_his == 1:
        print(f"Deleting history files in {delete_filepath_1}.")
        print(f"Deleting history files in {delete_filepath_2}.")
        commands = f"""
        rm {delete_filepath_1}
        rm {delete_filepath_2}
        ocean <<EOF
        load "{script_path}"
        exit()
        EOF
        """
    else:
        commands = f"""
        ocean <<EOF
        load "{script_path}"
        exit()
        EOF
        """
    # load "home/y864zhou/workarea_GF22_FDX_EXT/ocean_scripts/oceanScript_56G_good_example_1_test.ocn"
    # Run the command
    process = subprocess.run(commands, shell=True, capture_output=True, text=True)
    print("Captured STDOUT:", process.stdout)
    print("Captured STDERR:", process.stderr)
    output_log_path = "~/workarea_GF22_FDX_EXT/python_scripts/output_log.txt"
    error_log_path = "~/workarea_GF22_FDX_EXT/python_scripts/error_log.txt"
    with open(os.path.expanduser(output_log_path), 'w') as f:
        f.write(process.stdout)
    with open(os.path.expanduser(error_log_path), 'w') as f:
        f.write(process.stderr)
    ocean_runtime = time.time() - start_time
    return ocean_runtime

def function_CTLE(
    input_list: list[float],
) -> float:
    """
    # input_list = [fW, current, ind, Rd, Cs, Rs]
    # Output from the Cadence simulation
    """
    
    # Create the design parameter ocean script 
    design_param_directory = "~/workarea_GF22_FDX_EXT/ocean_scripts/"
    design_param_filename = f"design_variables_tb_channel_CTLE_ML_Bayesian_{args.run_id}.ocn"
    ocean_creation.create_design_param_ocn(design_param_directory, design_param_filename, input_list)

    # Run the ocean script
    simulation_filename = f"~/workarea_GF22_FDX_EXT/ocean_scripts/oceanScript_56G_good_example_1_test_{args.run_id}.ocn"
    ocean_runtime = run_ocean_simulation(simulation_filename)
    print(f"Ocean runtime: {ocean_runtime}")
    # Check if the output file is generated, copy the output file with different name, delete the output file
    output_dir = os.path.expanduser(f"~/workarea_GF22_FDX_EXT/CSV_results/BO_result_run_{args.run_id}")  # <- Change this
    original_file = f"test.csv"
    # --- Basic Setup ---
    base_name = "myresult"
    extension = ".csv"
    counter = 0 # Default start with 0
    # --- Generate Filename ---
    # Format the filename using the counter
    user_defined_name = f"{base_name}_{counter}{extension}"
    user_defined_path = os.path.join(output_dir, user_defined_name)
    # --- Loop to Find a Unique Name ---
    # Keep incrementing the counter until a filename is found that doesn't exist
    while os.path.exists(user_defined_path):
        counter += 1
        user_defined_name = f"{base_name}_{counter}{extension}" # Always up to date
        user_defined_path = os.path.join(output_dir, user_defined_name)
    timeout_seconds = 180  # wait for up to 3 minutes
    check_interval_seconds = 10
    success = ocean_creation.wait_for_file_and_rename(
        output_location=output_dir,
        original_name=original_file,
        new_name=user_defined_name,
        timeout=timeout_seconds,
        check_interval=check_interval_seconds
    )
    if success:
        print("File renamed successfully.")
    else:
        print("File operation failed or timed out.")
    # Collect the generated results from the output file
    new_path = os.path.join(output_dir, user_defined_name)
    output_dict = ocean_creation.read_csv_data(new_path)
    stage_1_eye_max_height = output_dict.get("eye_maxHeight Vout_1 56G")
    stage_1_eye_max_width = output_dict.get("eye_maxWidth Vout_1 56G")
    stage_2_eye_max_height = output_dict.get("eye_maxHeight Vout_2 56G")
    stage_2_eye_max_width = output_dict.get("eye_maxWidth Vout_2 56G")
    stage_1_dc_attenuation = output_dict.get("stage 1 0.1G attenuation")
    stage_2_dc_attenuation = output_dict.get("stage 2 0.1G attenuation")
    stage_1_28G_attenuation = output_dict.get("stage 1 28G attenuation")
    stage_2_28G_attenuation = output_dict.get("stage 2 28G attenuation")
    print(f"stage_1_eye_max_height = {stage_1_eye_max_height}.")
    print(f"stage_1_eye_max_width = {stage_1_eye_max_width}.")
    print(f"stage_2_eye_max_height = {stage_2_eye_max_height}.")
    print(f"stage_2_eye_max_width = {stage_2_eye_max_width}.")
    print(f"stage_1_dc_attenuation = {stage_1_dc_attenuation}.")
    print(f"stage_2_dc_attenuation = {stage_2_dc_attenuation}.")
    print(f"stage_1_28G_attenuation = {stage_1_28G_attenuation}.")
    print(f"stage_2_28G_attenuation = {stage_2_28G_attenuation}.")
    # Return the reward
    reward = (args.eye_coef_1 * stage_1_eye_max_height * stage_1_eye_max_width + 
              args.eye_coef_2 * stage_2_eye_max_height * stage_2_eye_max_width - 
              args.dc_coef * stage_1_dc_attenuation * stage_2_dc_attenuation - 
              args.ac_coef * stage_1_28G_attenuation * stage_2_28G_attenuation)
    print(f"Reward = {reward}\n")
    with open(os.path.join(output_dir, 'CTLE_x_input_history.txt'), 'a') as file:
        file.write('====================================================\n')
        file.write(user_defined_name)
        file.write("\n")
        file.write(f"Ocean runtime is: {ocean_runtime}\n")
        file.write(f"Swept inputs: fW = {input_list[0]}, " +
                                   f"current = {input_list[1]}, "+
                                   f"ind = {input_list[2]}, "+
                                   f"Rd = {input_list[3]}, "+
                                   f"Cs = {input_list[4]}, "+
                                   f"Rs = {input_list[5]},\n")
        file.write(f"stage_1_eye_max_height = {stage_1_eye_max_height}\n")
        file.write(f"stage_1_eye_max_width = {stage_1_eye_max_width}\n")
        file.write(f"stage_2_eye_max_height = {stage_2_eye_max_height}\n")
        file.write(f"stage_2_eye_max_width = {stage_2_eye_max_width}\n")
        file.write(f"stage_1_dc_attenuation = {stage_1_dc_attenuation}\n")
        file.write(f"stage_2_dc_attenuation = {stage_2_dc_attenuation}\n")
        file.write(f"stage_1_28G_attenuation = {stage_1_28G_attenuation}\n")
        file.write(f"stage_2_28G_attenuation = {stage_2_28G_attenuation}\n")
        file.write(f"Reward = {reward}\n")
    return -reward

def objective_CTLE(trial: optuna.trial.Trial) -> float:
    """Simple Optuna objective function for CTLE optimization."""
    fW = trial.suggest_float("x0", fW_RANGE[0], fW_RANGE[1])
    current = trial.suggest_float("x1", current_RANGE[0], current_RANGE[1])
    ind = trial.suggest_float("x2", ind_RANGE[0], ind_RANGE[1])
    Rd = trial.suggest_float("x3", Rd_RANGE[0], Rd_RANGE[1])
    Cs = trial.suggest_float("x5", Cs_RANGE[0], Cs_RANGE[1])
    Rs = trial.suggest_float("x4", Rs_RANGE[0], Rs_RANGE[1])

    input_list = [fW, current, ind, Rd, Cs, Rs]
    return function_CTLE(input_list)

# --- Main Optimization Logic ---
samplers = {
    "Base": optuna.samplers.BaseSampler,
    "Grid": optuna.samplers.GridSampler,
    "Random": optuna.samplers.RandomSampler,
    "TPE": optuna.samplers.TPESampler,
    "CmaEs": optuna.samplers.CmaEsSampler,
    "GPS": optuna.samplers.GPSampler,
    "PartialFixed": optuna.samplers.PartialFixedSampler,
    "NSGAII": optuna.samplers.NSGAIISampler,
    "NSGAIII": optuna.samplers.NSGAIIISampler,
    "QMCS": optuna.samplers.QMCSampler,
    "BruteForce": optuna.samplers.BruteForceSampler,
    "nsgaii": optuna.samplers.nsgaii
}


def track_optimization_CTLE(max_trials: int) -> list[float]:
    """
    Run CTLE optimization and track the running best value at each trial.
    
    Args:
        max_trials: Maximum number of trials to run
        
    Returns:
        List of running best values (one for each trial)
    """
    objective = objective_CTLE
    study = optuna.create_study(direction="minimize", sampler=samplers["GPS"]())
    # study = optuna.create_study(direction="minimize", sampler=samplers["Grid"](search_space_10d))
    # collected_results = []
    # def collect_data(): # inner function
    #     collected_results.append(study.best_value)
    
    # study.optimize(objective, n_trials=max_trials, callbacks=[collect_data])

    collected_results = []
    for trial in range(max_trials):
        trial = study.ask()
        value = objective(trial)
        # collected_results.append(value)
        study.tell(trial, value) # Finish the trial created with ask.
        collected_results.append(study.best_value)

    return collected_results  # Placeholder for actual implementation


def run_multiple_optimizations_CTLE(max_trials: int, num_reps: int) -> list[list[float]]:
    """
    Run CTLE optimization multiple times and collect running best values.
    
    Args:
        max_trials: Maximum number of trials per run
        num_reps: Number of repetitions
        
    Returns:
        List of lists, where each sublist contains running best values for one run
    """
    all_runs = []
    for rep in range(num_reps):
        print(f"Running CTLE optimization repetition {rep + 1}/{num_reps}")
        
        # TODO: Add rest of the implementation.
        one_run = []
        # TODO: Add rest of the implementation.
        one_run =  track_optimization_CTLE(max_trials)
        all_runs.append(one_run)
    return all_runs


def plot_convergence(title: str, all_runs: list[list[float]]) -> None:
    """
    Plot the convergence of optimization runs showing mean and standard deviation.
    
    Args:
        title: Title for the plot
        all_runs: List of lists, where each sublist contains running best values for one run
    """
    runs_array = np.array(all_runs)
    
    # Calculate mean and standard deviation across runs
    mean_values = np.mean(runs_array, axis=0)
    std_values = np.std(runs_array, axis=0)
    trials = np.arange(1, len(mean_values) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot individual runs with low alpha
    for run in all_runs:
        plt.plot(trials, run, alpha=0.3, color='lightblue', linewidth=0.5)
    
    # Plot mean line
    plt.plot(trials, mean_values, color='blue', linewidth=2, label='Mean')
    
    # Plot confidence interval (mean ± std)
    plt.fill_between(trials, 
                     mean_values - std_values, 
                     mean_values + std_values, 
                     alpha=0.3, color='blue', label='±1 Std Dev')
    
    plt.xlabel('Number of Trials')
    plt.ylabel('Best Objective Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = title.replace(" ", "_") + ".png"
    output_dir = os.path.expanduser(f"~/workarea_GF22_FDX_EXT/CSV_results/BO_result_run_{args.run_id}")
    file_path = os.path.join(output_dir, filename)
    plt.savefig(file_path)
    print(f"Plot saved as {filename}")

    # Save the mean and std values as a CSV file
    csv_filename = title.replace(" ", "_") + ".csv"
    csv_file_path = os.path.join(output_dir, csv_filename)
    with open(csv_file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Trial', 'Mean', 'Std'])
        for i, (mean, std) in enumerate(zip(mean_values, std_values), start=1):
            writer.writerow([i, mean, std])
    print(f"CSV saved as {csv_filename}")


def main():
    """
    Main function to demonstrate convergence tracking and plotting.
    """    
    print("Exploring optimization convergence in Optuna...")

    # Setup parser
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=0, help="Run id (for parallel run purpose)")
    parser.add_argument("--eye_coef_1", type=float, default=1.0, help="Reward coefficient for stage 1 eye info")
    parser.add_argument("--eye_coef_2", type=float, default=1.5, help="Reward coefficient for stage 2 eye info")
    parser.add_argument("--dc_coef", type=float, default=10, help="Reward coefficient for ac info")
    parser.add_argument("--ac_coef", type=float, default=0.1, help="Reward coefficient for ac info")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials for 1 rep")
    parser.add_argument("--reps", type=int, default=1, help="Number of reps for better results")
    parser.add_argument("--del_his", type=int, default=1, help="Delete maestro history files")
    args = parser.parse_args()
    # python ./python_scripts/simple_CTLE_optimization.py --run_id 1 --trials 5
    start_time = time.time()
    # Run CTLE optimization multiple times
    print("\n=== CTLE Function Optimization ===")
    runs_CTLE = run_multiple_optimizations_CTLE(max_trials=args.trials, num_reps=args.reps)
    plot_convergence("CTLE Optuna Optimization Convergence", runs_CTLE)

    # Print final statistics
    final_CTLE = [run[-1] for run in runs_CTLE]
    end_time = time.time()
    print(f"\n=== Final Results Summary ===")
    print(f"CTLE - Final best values:")
    print(f"  Mean: {np.mean(final_CTLE):.6f}")
    print(f"  Std:  {np.std(final_CTLE):.6f}")
    print(f"  Best: {np.min(final_CTLE):.6f}")
    print(f"  Total runtime is: {end_time-start_time}")

if __name__ == "__main__":
    main()
