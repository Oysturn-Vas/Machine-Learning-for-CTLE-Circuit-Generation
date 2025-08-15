import os
import numpy as np
import subprocess
import time
import shutil
import csv
import io
import pprint

ENG_SUFFIXES = {
    -15: 'f',
    -12: 'p',
    -9: 'n',
    -6: 'u',
    -3: 'm',
     3: 'k',
     6: 'M',
     9: 'G'
}

def to_engineering_notation(val):
    """Convert float to engineering string with SI suffix."""
    if val == 0:
        return "0"
    # Scale the value to the nearest lower multiple of 3 (base 10). e.g. 5.7 → 3, 1200 → 3, 0.002 → -3.
    exponent = int(np.floor(np.log10(abs(val)) // 3 * 3)) 
    if exponent in ENG_SUFFIXES:
        scaled = val / (10 ** exponent)
        return f"{scaled:.6g}{ENG_SUFFIXES[exponent]}" # .6g mean up to 6 significant digits.
    else:
        return f"{val:.6g}"

# input_list = [fW, current, ind, Rd, Cs, Rs]
def create_design_param_ocn(directory_name, filename, design_params):
    """
    Writes an OCEAN script file with design variables.

    Parameters:
        filename (str): Name of the file to write.
        design_params (list of float): List of 6 float values in the order:
            [fW, current, ind, Rd, Cs, Rs]
    """

    if len(design_params) != 6:
        raise ValueError("Exactly 6 design parameters must be provided.")

    # Convert values to strings with appropriate units
    fW_str     = f'"{to_engineering_notation(design_params[0])}"'
    current_str= f'"{to_engineering_notation(design_params[1])}"'
    ind_str    = f'"{to_engineering_notation(design_params[2])}"'
    Rd_str     = f'"{to_engineering_notation(design_params[3])}"'
    Cs_str     = f'"{to_engineering_notation(design_params[4])}"'
    Rs_str     = f'"{to_engineering_notation(design_params[5])}"'

    content = f"""desVar(	  "wireopt" 19	)
desVar(	  "bitperiod" "1/56G"	) ; Signal input frequency
desVar(	  "V_one" 0.9	)
desVar(	  "V_zero" 0.5	)
desVar(	  "Vin_AC" 1	)
desVar(	  "Vin_DC" 0.7	)
desVar(	  "L" 20n	)
desVar(	  "Rsource" 0	)
desVar(	  "Rload" 50	)
; Actual varying parameters
desVar(	  "fW" {fW_str}	) 
desVar(	  "current" {current_str}	)
desVar(	  "ind" {ind_str}	)
desVar(	  "Rd" {Rd_str}	)
desVar(	  "Cs" {Cs_str}	)
desVar(	  "Rs" {Rs_str}	)
"""
    directory = os.path.expanduser(directory_name)
    os.makedirs(directory, exist_ok=True)
    # Full path for the output file
    full_path = os.path.join(directory, filename)

    with open(full_path, 'w') as f:
        f.write(content)

    print(f"File written to: {full_path}")

# def run_ocean_simulation(simulation_filename):
#     start_time = time.time()
#     # Expand the Ocean script path
#     script_path = os.path.expanduser(simulation_filename)
#     # delete_filepath = os.path.expanduser(f"~/workarea_GF22_FDX_EXT/ECE740/tb_channel_CTLE_ML_{args.run_id}")
#     # Compose the shell command with a here-document
#     commands = f"""
#     ocean <<EOF
#     load "{script_path}"
#     exit()

#     EOF
#     """
#     # load "home/y864zhou/workarea_GF22_FDX_EXT/ocean_scripts/oceanScript_56G_good_example_1_test.ocn"
#     # Run the command
#     process = subprocess.run(commands, shell=True, capture_output=True, text=True)
#     print("Captured STDOUT:", process.stdout)
#     print("Captured STDERR:", process.stderr)
#     output_log_path = "~/workarea_GF22_FDX_EXT/python_scripts/output_log.txt"
#     error_log_path = "~/workarea_GF22_FDX_EXT/python_scripts/error_log.txt"
#     with open(os.path.expanduser(output_log_path), 'w') as f:
#         f.write(process.stdout)
#     with open(os.path.expanduser(error_log_path), 'w') as f:
#         f.write(process.stderr)
#     ocean_runtime = time.time() - start_time
#     return ocean_runtime

def wait_for_file_and_rename(output_location, original_name, new_name, timeout=60, check_interval=10):
    """
    Waits for a file to appear in a directory and renames it if found.
    
    Parameters:
    - output_location: str - directory path where the file is expected
    - original_name: str - name of the output file to wait for (e.g. "test.csv")
    - new_name: str - new name for the file (e.g. "results_backup.csv")
    - timeout: int - max time to wait in seconds
    - check_interval: int - how often to check in seconds
    """
    original_path = os.path.join(output_location, original_name)
    new_path = os.path.join(output_location, new_name)

    print(f"Waiting for '{original_name}' to appear in '{output_location}'...")

    start_time = time.time()

    while time.time() - start_time < timeout:
        if os.path.exists(original_path):
            print(f"File found: {original_path}")

            # Copy to new name
            shutil.copy2(original_path, new_path)
            print(f"Copied to: {new_path}")

            # Delete original
            os.remove(original_path)
            print(f"Deleted original file: {original_path}")
            return True

        time.sleep(check_interval)
        print(f"Sleeped for {check_interval} seconds")

    print("Timeout reached. File not found.")
    return False

#=========================================================

def read_csv_data(output_csv):
    # Assume your CSV data is in a string.
    # If reading from a file, replace 'io.StringIO(csv_data)' with 'open("your_file.csv", "r")'
    csv_data = open(output_csv, "r")
    # The keywords you want to find
    keywords_to_find = {
        "eye_maxHeight Vout_1 56G",
        "eye_maxWidth Vout_1 56G",
        "eye_maxHeight Vout_2 56G",
        "eye_maxWidth Vout_2 56G",
        "stage 1 0.1G attenuation",
        "stage 2 0.1G attenuation",
        "stage 1 28G attenuation",
        "stage 2 28G attenuation",
    }
    # A dictionary to store the results
    extracted_results = {}
    # Use io.StringIO to read the string data like a file
    csv_reader = csv.reader(csv_data)
    # --- Main Logic ---
    # Loop through each row in the CSV
    for row in csv_reader:
        # The keyword is in the second column (index 1)
        if len(row) > 2:
            keyword_in_row = row[1]
            if keyword_in_row in keywords_to_find:
                # The value is in the third column (index 2)
                value_str = row[2]
                # Convert the value from a string to a float and store it
                extracted_results[keyword_in_row] = float(value_str)

    # --- Assign to Variables and Print Results ---
    # Use .get() to safely access the dictionary values
    #eye_max_height = extracted_results.get("eye_maxHeight Vout_1 56G")
    #eye_max_width = extracted_results.get("eye_maxWidth Vout_1 56G")

    print(f"✅ Extraction complete!")
    #print(f"eye_maxHeight Vout_1 56G = {eye_max_height}")
    #print(f"eye_maxWidth Vout_1 56G  = {eye_max_width}")
    # Return the directory
    return extracted_results

def main():
    """
    Main function to demonstrate convergence tracking and plotting.
    """    
    # Example usage:
    input_list = [1e-6, 2.5e-3, 4.01e-12, 730.3, 1e-12, 740.8]
    design_param_directory = "~/workarea_GF22_FDX_EXT/ocean_scripts/"
    design_param_filename = "design_variables_tb_channel_CTLE_ML_created.ocn"
    create_design_param_ocn(design_param_directory, design_param_filename, input_list)
    simulation_filename = "~/workarea_GF22_FDX_EXT/ocean_scripts/oceanScript_56G_good_example_1_test.ocn"
    ocean_runtime = run_ocean_simulation(simulation_filename)
    print(f"Ocean runtime: {ocean_runtime}")
    # print(f"\n=== Final Results Summary ===")
    # print(f"1D Function - Final best values:")
    # print(f"  Mean: {np.mean(final_1d):.6f}")
    # print(f"  Std:  {np.std(final_1d):.6f}")
    # print(f"  Best: {np.min(final_1d):.6f}")
    output_dir = os.path.expanduser("~/workarea_GF22_FDX_EXT/CSV_results/")  # <- Change this
    original_file = "test.csv"
    user_defined_name = "my_results_0.csv"
    timeout_seconds = 180  # wait for up to 3 minutes
    check_interval_seconds = 10
    success = wait_for_file_and_rename(
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
    new_path = os.path.join(output_dir, user_defined_name)
    output_dict = read_csv_data(new_path)
    print("\n--- Formatted with pprint ---")
    pprint.pprint(output_dict)
    # for key, value in extracted_results.items():
    #     print(f"'{key}': {value}")

if __name__ == "__main__":
    main()
