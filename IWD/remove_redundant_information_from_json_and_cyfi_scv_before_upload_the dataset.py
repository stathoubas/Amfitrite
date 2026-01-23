# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 11:12:48 2026

@author: K. Pikounis
"""

import os
import json
import pandas as pd
from tqdm import tqdm  # Library for progress bar, optional but recommended

def clean_dataset_files(root_folder):
    """
    Iterates through subfolders in root_folder.
    1. Updates metadata.json:
       - Removes 'max_case_score' from 'compatibility_analysis'
       - Adds 'compatibility_analysis' = {} if missing
    2. Updates cyfi_lattice_predictions.csv:
       - Removes 'date' column
    """
    
    if not os.path.exists(root_folder):
        print(f"Error: Root folder not found at {root_folder}")
        return

    # Get list of all subdirectories
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    print(f"Found {len(subfolders)} subfolders. Starting processing...")

    count_json = 0
    count_csv = 0

    # Iterate with a progress bar
    for folder_path in tqdm(subfolders, desc="Processing folders"):
        
        # ---------------------------------------------------------
        # 1. Process metadata.json
        # ---------------------------------------------------------
        json_path = os.path.join(folder_path, "metadata.json")
        
        if os.path.exists(json_path):
            try:
                # Read
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                modified = False
                
                # b) Check/Insert 'compatibility_analysis'
                if "compatibility_analysis" not in data:
                    data["compatibility_analysis"] = {}
                    modified = True
                else:
                    # a) Remove 'max_case_score' if key exists and is a dictionary
                    if isinstance(data["compatibility_analysis"], dict):
                        if "max_case_score" in data["compatibility_analysis"]:
                            data["compatibility_analysis"].pop("max_case_score")
                            modified = True

                # Overwrite file if changes were made (or just always to ensure consistency)
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=4)
                
                count_json += 1
                
            except Exception as e:
                print(f"[Error JSON] {os.path.basename(folder_path)}: {e}")

        # ---------------------------------------------------------
        # 2. Process cyfi_lattice_predictions.csv
        # ---------------------------------------------------------
        csv_path = os.path.join(folder_path, "cyfi_lattice_predictions.csv")
        
        if os.path.exists(csv_path):
            try:
                # Read
                df = pd.read_csv(csv_path)
                
                # Remove column 'date' if it exists
                if 'date' in df.columns:
                    df.drop(columns=['date'], inplace=True)
                    
                    # Overwrite file
                    df.to_csv(csv_path, index=False)
                    count_csv += 1
                    
            except Exception as e:
                print(f"[Error CSV] {os.path.basename(folder_path)}: {e}")

    print("-" * 50)
    print("Processing Complete.")
    print(f"Processed {count_json} JSON files.")
    print(f"Processed {count_csv} CSV files.")

# --- EXECUTION ---
if __name__ == "__main__":
    # Replace this with your actual root path
    ROOT_DIRECTORY = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\Dataset_v1"
    
    clean_dataset_files(ROOT_DIRECTORY)