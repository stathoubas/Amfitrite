# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 14:07:12 2025

@author: K. Pikounis

test_5_extra_data.py
Script to process extra data cases using utils_v5_extra_data.
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Import functions from your utils script
from utils_v5_extra_data import (
    find_satelite_images,
    select_item,
    extract_and_save_tile,
    predict_using_cyfi_pipeline,
    update_uids_and_summary
)

def load_processed_uids(filepath):
    """
    Loads already processed UIDs from the Excel log to allow resuming.
    """
    if os.path.exists(filepath):
        try:
            df = pd.read_excel(filepath)
            if 'uid' in df.columns:
                return set(df['uid'].astype(str))
        except Exception as e:
            print(f"Warning: Could not read processed log {filepath}: {e}")
    return set()

def log_problematic(uid, case, reason, filepath):
    """
    Logs failed cases to a CSV file.
    """
    file_exists = os.path.exists(filepath)
    df = pd.DataFrame([{
        'uid': uid,
        'case': case,
        'reason': reason
    }])
    # Append to CSV
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)
    print(f"xx Logged problem for {uid}: {reason}")

# --- CONFIGURATION ---
# Input file (Excel or CSV)
INPUT_FILE = "/vol/Amfitrite/sat_data/extra_data_no_low_cases/cases_not_low_lat_lon_part4.xlsx"

# Output directories and logs
OUTPUT_ROOT_FOLDER = "/vol/Amfitrite/sat_data/extra_data_no_low_cases"
PROCESSED_LOG = "/vol/Amfitrite/sat_data/extra_data_no_low_cases/processed_uids_part4.xlsx"
SUMMARY_LOG = "/vol/Amfitrite/sat_data/extra_data_no_low_cases/Master_summary_part4.xlsx"
PROBLEMATIC_LOG = "/vol/Amfitrite/sat_data/extra_data_no_low_cases/problematic_uids_part4.csv"

# Parameters
SEARCH_BUFFER_METERS = 3000
BOX_SIDE_PIXELS = 365  # 365 pixels * 10m = 3650m box
CLOUD_CUTOFF = 7.5


def main():
    print("--- Starting Extra Data Processing ---")
    
    # 1. Load Input Data
    if not os.path.exists(INPUT_FILE):
        # Fallback to CSV if Excel not found (based on your uploaded files)
        csv_fallback = "cases_not_low_lat_lon.xlsx - Sheet1.csv"
        if os.path.exists(csv_fallback):
            print(f"Input Excel not found, using CSV: {csv_fallback}")
            df_cases = pd.read_csv(csv_fallback)
        else:
            print(f"Error: Input file {INPUT_FILE} not found.")
            return
    else:
        df_cases = pd.read_excel(INPUT_FILE)

    total_cases = len(df_cases)
    total_cases_with_dates = df_cases.target_dates.str.split(",").str.len().sum()
    
    # 2. Load Processed State (Resume Feature)
    processed_uids = load_processed_uids(PROCESSED_LOG)
    print(f"Found {len(processed_uids)} already processed items.")
    
    # 3. Iterate Through Cases
    count_cases = 0
    cont_cases_with_dates = 0
    for idx, row in df_cases.iterrows():
        count_cases += 1
        case_id = str(row['case'])
        lat = row['lat']
        lon = row['lon']
        target_dates_str = str(row['target_dates'])
        ver = row["version"]
        
        # Split dates by comma
        if pd.isna(target_dates_str) or target_dates_str.lower() == 'nan':
            continue
            
        dates = [d.strip() for d in target_dates_str.split(',') if d.strip()]
        
        for date_str in dates:
            cont_cases_with_dates += 1
            # Create Unique ID: "case_date"
            uid = f"{case_id}_{date_str}"
            
            # CHECK RESUME
            if uid in processed_uids:
                print(f"Skipping {uid} (Done)")
                continue

            print(f"\n>>> Processing UID: {uid} | Case: {case_id} | Target Date: {date_str}")
            print(f">>> This is case {count_cases} from {total_cases}")
            print(f">>> Cosidering also dates this is case {cont_cases_with_dates} from {total_cases_with_dates}")

            # STEP 1: Find Satellite Images
            # Utils function handles the +/- 15 days window internally using date_str
            try:
                items_df = find_satelite_images(lat, lon, date_str, meter_buffer=SEARCH_BUFFER_METERS)
            except Exception as e:
                log_problematic(uid, case_id, f"Search Error: {e}", PROBLEMATIC_LOG)
                continue

            # STEP 2: Select Best Item
            # Returns (Success, ItemObject) based on our fixed utils
            success, selected_item = select_item(items_df)
            
            if not success:
                log_problematic(uid, case_id, "No images < 7.5% clouds", PROBLEMATIC_LOG)
                continue
                
            final_date = selected_item.datetime.strftime("%Y-%m-%d")
            print(f"   Selected Image Date: {final_date}")
            
            # STEP 3: Extract and Save Tile
            # Folder name format: case{id}_{date}
            out_folder_name = f"case{case_id}_{final_date}"
            out_folder_path = os.path.join(OUTPUT_ROOT_FOLDER, ver+"_extra", out_folder_name)

            # Setup Output
            if not os.path.exists(OUTPUT_ROOT_FOLDER):
                os.makedirs(OUTPUT_ROOT_FOLDER)
            
            extract_ok = extract_and_save_tile(
                item=selected_item,
                lat=lat,
                lon=lon,
                case_id=uid, # Passing case_id for metadata
                initial_pixel_size=BOX_SIDE_PIXELS,
                save_data=True,
                output_path=out_folder_path,
                print_images=False
            )
            
            if not extract_ok:
                log_problematic(uid, case_id, "Extraction/Clipping Failed", PROBLEMATIC_LOG)
                continue
                
            # STEP 4: Predict using CyFi
            # Returns (Success, New_Path)
            pred_ok, final_path = predict_using_cyfi_pipeline(out_folder_path, final_date)
            
            if pred_ok:
                # STEP 5: Update Logs
                # We pass the final path (which might have been renamed)
                update_uids_and_summary(final_path, PROCESSED_LOG, SUMMARY_LOG)
                
                # Update local set to avoid reprocessing if code loops
                processed_uids.add(uid)
            else:
                log_problematic(uid, case_id, "Prediction Failed / Low Valid Points", PROBLEMATIC_LOG)

if __name__ == "__main__":
    main()