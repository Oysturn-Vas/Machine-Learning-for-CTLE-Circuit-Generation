import pandas as pd
import os


def combine_csv_files(folder_path, output_file_name):
    """
    Combines data from multiple CSV files in a specified folder into one.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
        output_file_name (str): The name of the new CSV file to save the combined data to.
    """
    all_data = (
        pd.DataFrame()
    )  # Initialize an empty DataFrame to store all combined data

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")
            try:
                # Read the CSV file
                # You might need to specify encoding if you encounter errors (e.g., encoding='utf-8')
                df = pd.read_csv(file_path)
                all_data = pd.concat([all_data, df], ignore_index=True)
            except Exception as e:
                print(f"Could not read '{filename}': {e}")

    if not all_data.empty:
        # Save the combined data to a new CSV file
        # You might want to specify encoding here too (e.g., encoding='utf-8')
        output_file_path = os.path.join(os.path.dirname(folder_path), output_file_name)
        all_data.to_csv(output_file_path, index=False)
        print(f"\nSuccessfully combined data into: {output_file_path}")
    else:
        print("\nNo data was combined. Please check folder path and file types (.csv).")


# --- Configuration ---
# IMPORTANT: Replace 'path/to/your/csv_files' with the actual path to your folder
csv_files_folder = r"C:\Users\oystu\Desktop\Waterloo\2. Spring 2025\ECE 720 - ML for Chip Design\Project\Project\Dataset"  # Example: r"C:\Users\YourName\Documents\my_csv_files" or "/Users/YourName/Documents/my_csv_files"
output_csv_name = "Combined_ProcessedData.csv"

# Run the function
combine_csv_files(csv_files_folder, output_csv_name)
