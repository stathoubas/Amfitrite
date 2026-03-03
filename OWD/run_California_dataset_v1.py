# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:23:32 2026

@author: K. Pikounis
"""

import pandas as pd
import os
import argparse
import logging
import traceback
import csv
import json
import shutil

# Import functions from your utils script
from utils_v1 import (
    find_satelite_images,
    select_item,
    extract_and_save_tile,
    predict_using_cyfi_pipeline,
    run_single_folder,
    folder_summary
)

def setup_logger(output_root, base_name):
    """Sets up a text logger for detailed debugging."""
    log_file = os.path.join(output_root, f"execution_log_{base_name}.log")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    return log_file

def run_automated_pipeline(csv_path, output_root, model_paths):
    # 1. Setup Tracking and Logging
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
        
    log_txt_path = setup_logger(output_root, base_name)
    log_csv_path = os.path.join(output_root, f"summary_report_{base_name}.csv")
    
    logging.info(f"Starting pipeline. Reading input from: {csv_path}")
    
    # --- RESUME FUNCTIONALITY ---
    processed_cases = set()
    
    if os.path.exists(log_csv_path):
        logging.info(f"Found existing summary report: {log_csv_path}. Checking for processed cases...")
        try:
            existing_log_df = pd.read_csv(log_csv_path)
            processed_cases = set(existing_log_df['case_id'].astype(str))
            logging.info(f"Found {len(processed_cases)} previously processed cases. These will be skipped.")
            
            log_file_mode = 'a'
            write_header = False
        except Exception as e:
            logging.warning(f"Could not parse existing summary report. Starting fresh. Error: {e}")
            log_file_mode = 'w'
            write_header = True
    else:
        log_file_mode = 'w'
        write_header = True

    # Initialize the CSV report file with the NEW CLASSIFICATION COLUMNS
    csv_headers = [
        "case_id", "date", "status", "failed_stage", "error_message", "output_folder",
        "Overall_Class", "Max_HAB_Index", "HAB_Tiles_Sorted", "Max_Check_Index", "Check_Tiles_Sorted"
    ]
    with open(log_csv_path, mode=log_file_mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(csv_headers)
        
    # 2. Read the Input Data
    try:
        df = pd.read_csv(csv_path) if csv_path.endswith('.csv') else pd.read_excel(csv_path)
    except Exception as e:
        logging.error(f"Failed to read input file {csv_path}. Error: {e}")
        return

    # Constants
    SEARCH_BUFFER_METERS = 3840
    BOX_SIDE_PIXELS = 256  
    
    # 3. Process Row by Row
    for index, row in df.iterrows():
        # --- SAFE EXTRACTION FROM YOUR CSV FORMAT ---
        # Note: Adjust 'id_x', 'decimalLatitude', 'decimalLongitude', 'eventDate' if your new Excel differs!
        case_id = str(row.get('Event_count', f"Row_{index}"))
        
        if case_id in processed_cases:
            logging.info(f"Row {index+1}: Case {case_id} already processed. Skipping.")
            continue
        
        try:
            lat = float(row.get('decimalLatitude'))
            lon = float(row.get('decimalLongitude'))
        except (ValueError, TypeError):
            logging.error(f"Row {index} ({case_id}): Invalid or missing Lat/Lon. Skipping.")
            continue
            
        raw_date = str(row.get('Event_Date'))
        try:
            parsed_date = pd.to_datetime(raw_date)
            date_str = parsed_date.strftime("%Y-%m-%d")
        except Exception as e:
            logging.error(f"Row {index} ({case_id}): Invalid Date format '{raw_date}'. Skipping.")
            continue
            
        uid = f"{case_id}_{date_str}"
        logging.info(f"\n{'='*60}\nProcessing Case: {uid} (Row {index+1}/{len(df)})\n{'='*60}")
        
        # Tracking variables
        status = "SUCCESS"
        failed_stage = "None"
        error_msg = ""
        out_folder_path = ""
        
        # New classification variables
        overall_class = ""
        max_hab_idx = ""
        hab_tiles_sorted = ""
        max_chk_idx = ""
        chk_tiles_sorted = ""
        
        try:
            failed_stage = "1_Search_and_Select"
            items_df = find_satelite_images(lat, lon, date_str, meter_buffer=SEARCH_BUFFER_METERS)
            
            if items_df.empty:
                raise ValueError("No satellite items found.")

            # --- STAGE 1 & 2: THE 3-ATTEMPT SAFETY LOOP ---
            extraction_successful = False
            tried_indices = []
            
            for attempt in range(1, 4):
                failed_stage = f"1_Search_Attempt_{attempt}"
                logging.info(f"--- Extraction Attempt {attempt}/3 ---")
                
                # Apply cloud logic based on attempt
                if attempt == 1:
                    candidates = items_df[items_df.per_clouds < 7.5]
                else:
                    # After attempt 1, enforce > 0.5% clouds to avoid black glitches
                    candidates = items_df[(items_df.per_clouds > 0.5) & (items_df.per_clouds < 7.5)]
                
                # Filter out ones we already tried
                candidates = candidates[~candidates.index.isin(tried_indices)]
                
                if candidates.empty:
                    if attempt == 1:
                        raise ValueError("No items found with < 7.5% clouds.")
                    else:
                        raise ValueError("No items left with 0.5% - 7.5% clouds.")
                        
                # Pick the best available
                best_row = candidates.sort_values(by="per_clouds", ascending=True).iloc[0]
                best_item_index = best_row.name
                selected_item = best_row["item_obj"]
                tried_indices.append(best_item_index)
                
                final_date = selected_item.datetime.strftime("%Y-%m-%d")
                out_folder_name = f"case_{case_id}_{final_date}_item{best_item_index}"
                out_folder_path = os.path.join(output_root, out_folder_name)
                
                if not os.path.exists(out_folder_path):
                    os.makedirs(out_folder_path)

                failed_stage = f"2_Extract_Tiles_Attempt_{attempt}"
                extract_ok = extract_and_save_tile(
                    item=selected_item, lat=lat, lon=lon, case_id=uid,
                    initial_pixel_size=BOX_SIDE_PIXELS, save_data=True,
                    output_path=out_folder_path, print_images=True
                )
                
                if not extract_ok:
                    logging.warning(f"Tile extraction failed. Cleaning up folder and retrying...")
                    shutil.rmtree(out_folder_path, ignore_errors=True)
                    continue

                # --- NEW SAFETY CHECK: Count total water across all 9 tiles ---
                total_water_pixels = 0
                for i in range(9):
                    meta_path = os.path.join(out_folder_path, f"tile_{i}", "metadata.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            total_water_pixels += meta.get("water_pixels", 0)
                
                if total_water_pixels == 0:
                    logging.warning(f"Total black image detected (0 water pixels). Trashing folder and retrying...")
                    shutil.rmtree(out_folder_path, ignore_errors=True)
                    continue
                
                # If we get here, extraction worked and we have water!
                extraction_successful = True
                break
                
            if not extraction_successful:
                raise ValueError("Failed to extract a valid image with water pixels after 3 attempts.")

            # --- STAGE 3: CyFi Pipeline ---
            failed_stage = "3_CyFi_Pipeline"
            predict_using_cyfi_pipeline(out_folder_path, final_date)

            # --- STAGE 4: CNN Inference ---
            failed_stage = "4_CNN_Inference"
            run_single_folder(out_folder_path, model_paths)

            # --- STAGE 5: Summarize ---
            failed_stage = "5_Folder_Summary"
            df_summary = folder_summary(out_folder_path, add_images_to_excel=True)
            
            # --- STAGE 6: INLINE CLASSIFICATION & SCORING ---
            failed_stage = "6_Classification"
            
            if df_summary is not None and not df_summary.empty:
                # 1. Validity Filters
                valid_tiles = df_summary[
                    (df_summary['Total_pixels'] == 65536) & 
                    (df_summary['SCL_Water_Pixels'] >= 32000)
                ].copy()
                
                if valid_tiles.empty:
                    overall_class = "Filters removed all tiles"
                else:
                    hab_list = []
                    check_list = []
                    clean_count = 0
                    
                    for _, tile in valid_tiles.iterrows():
                        tile_name = tile['Tile_Folder']
                        
                        def is_yes(col_name):
                            return str(tile.get(col_name, '')).strip().upper() == 'YES'
                            
                        rdnet_yes = is_yes('rdnet_no_scl')
                        cnn_cols = ['rdnet_no_scl', 'convnext_scl', 'res18_scl', 'res18_no_scl']
                        yes_count = sum(1 for col in cnn_cols if is_yes(col))
                        
                        # Score CNN (Max 60)
                        cnn_score = 0
                        if is_yes('rdnet_no_scl'): cnn_score += 30
                        if is_yes('convnext_scl'): cnn_score += 10
                        if is_yes('res18_scl'): cnn_score += 10
                        if is_yes('res18_no_scl'): cnn_score += 10
                        
                        # Score CyFi (Max 40)
                        cyfi_high = float(tile.get('CyFi_High', 0))
                        cyfi_mod = float(tile.get('CyFi_Moderate', 0))
                        cyfi_mass = cyfi_high + (0.5 * cyfi_mod)
                        cyfi_score = (cyfi_mass / 676.0) * 40.0
                        
                        total_index = round(cnn_score + cyfi_score, 2)
                        
                        # Apply Classification Logic
                        is_hab = rdnet_yes and (yes_count >= 2) and (cyfi_high >= 20)
                        is_clean = (not rdnet_yes) and (yes_count <= 1) and (cyfi_high < 5) and (cyfi_mod < 15)
                        
                        if is_hab:
                            hab_list.append((tile_name, total_index))
                        elif is_clean:
                            clean_count += 1
                        else:
                            check_list.append((tile_name, total_index))
                            
                    # Sort by Index (Highest first)
                    hab_list.sort(key=lambda x: x[1], reverse=True)
                    check_list.sort(key=lambda x: x[1], reverse=True)
                    
                    if hab_list:
                        overall_class = "HAB Detected"
                        max_hab_idx = hab_list[0][1]
                        hab_tiles_sorted = " | ".join([f"{name} ({score})" for name, score in hab_list])
                    elif check_list:
                        overall_class = "To Be Checked"
                        max_chk_idx = check_list[0][1]
                        chk_tiles_sorted = " | ".join([f"{name} ({score})" for name, score in check_list])
                    elif clean_count > 0:
                        overall_class = "Clean"
            
            failed_stage = "None" 
            logging.info(f"Successfully completed {uid} | Class: {overall_class}")

        except Exception as e:
            status = "FAILED"
            error_msg = str(e).replace('\n', ' ')
            logging.error(f"Failed at {failed_stage} for {uid}. Error: {error_msg}")
            logging.debug(traceback.format_exc()) 
            
        finally:
            # Write everything immediately to CSV
            row_data = [
                case_id, date_str, status, failed_stage, error_msg, out_folder_path,
                overall_class, max_hab_idx, hab_tiles_sorted, max_chk_idx, chk_tiles_sorted
            ]
            with open(log_csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
                f.flush()

    logging.info(f"\nAll rows processed. Summary report saved to: {log_csv_path}")


if __name__ == "__main__":
    MODEL_PATHS = {
        "res18_scl":    "/vol/Amfit