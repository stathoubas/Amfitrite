# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 00:13:07 2026

@author: K. Pikounis
"""

import pandas as pd
import os
import argparse
import logging
import traceback
import csv
import planetary_computer as pc
from pystac_client import Client

# Import functions from your utils script
from utils_v1 import (
    extract_and_save_tile,
    predict_using_cyfi_pipeline,
    run_single_folder,
    folder_summary
)

# Initialize STAC client to fetch the specific item
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def setup_logger(output_root, base_name):
    """Sets up a text logger for detailed debugging."""
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
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

def fetch_specific_stac_item(item_id):
    """Fetches the exact STAC item object from Planetary Computer using its ID."""
    search = catalog.search(collections=["sentinel-2-l2a"], ids=[item_id])
    items = list(search.items())
    if not items:
        return None
    return items[0]

def run_heavy_pipeline(csv_path, output_root, model_paths):
    # 1. Setup Tracking and Logging
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
        
    log_txt_path = setup_logger(output_root, base_name)
    log_csv_path = os.path.join(output_root, f"summary_report_{base_name}.csv")
    
    logging.info(f"Starting Phase 2 Pipeline. Reading input from: {csv_path}")
    
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

    # Initialize the CSV report file
    csv_headers = [
        "case_id", "date", "status", "failed_stage", "error_message", "output_folder",
        "Overall_Class", "Max_HAB_Index", "HAB_Tiles_Sorted", "Max_Check_Index", "Check_Tiles_Sorted",
        "Index_per_Tile"
    ]
    with open(log_csv_path, mode=log_file_mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(csv_headers)
        
    # 2. Read the Input Data
    try:
        df = pd.read_csv(csv_path, low_memory=False) if csv_path.endswith('.csv') else pd.read_excel(csv_path)
    except Exception as e:
        logging.error(f"Failed to read input file {csv_path}. Error: {e}")
        return

    BOX_SIDE_PIXELS = 256  
    
    # Find Lat/Lon columns dynamically to avoid Excel truncation crashes
    lat_col = next((c for c in df.columns if 'decimalLat' in c or c == 'decimalLa' or c == 'decima'), 'decimalLatitude')
    lon_col = next((c for c in df.columns if 'decimalLon' in c or c == 'decimalLo'), 'decimalLongitude')
    
    # 3. Process Row by Row
    for index, row in df.iterrows():
        # --- ROBUST ID EXTRACTION ---
        # Look for HAB_ID, id_x, or generated_id from Phase 1. Fallback to Row index.
        raw_id = row.get('HAB_ID', row.get('id_x', row.get('generated_id', f"Row_{index}")))
        case_id = str(raw_id).replace(":", "_")
        
        # Ensure there is a valid satellite item assigned from Phase 1
        sat_item_id = str(row.get('sat_item', "")).strip()
        
        # Skip if missing STAC item or "nan" string
        if not sat_item_id or sat_item_id.lower() == 'nan':
            # Only log every 1000th skipped row to prevent log bloat
            if index % 1000 == 0:
                logging.debug(f"Row {index+1}: Skipped (No valid satellite item)")
            continue
            
        if case_id in processed_cases:
            logging.info(f"Row {index+1}: Case {case_id} already processed. Skipping.")
            continue
        
        try:
            lat = float(row.get(lat_col))
            lon = float(row.get(lon_col))
        except (ValueError, TypeError):
            logging.error(f"Row {index} ({case_id}): Invalid or missing Lat/Lon. Skipping.")
            continue
            
        # Extract Date (Phase 1 standardizes this in the eventDate column)
        raw_date = row.get('eventDate')
        try:
            parsed_date = pd.to_datetime(str(raw_date))
            date_str = parsed_date.strftime("%Y-%m-%d")
        except Exception as e:
            logging.error(f"Row {index} ({case_id}): Invalid Date format '{raw_date}'. Skipping.")
            continue
            
        uid = f"{case_id}_{date_str}"
        logging.info(f"\n{'='*60}\nProcessing Case: {uid} | Item: {sat_item_id} (Row {index+1}/{len(df)})\n{'='*60}")
        
        # Tracking variables
        status = "SUCCESS"
        failed_stage = "None"
        error_msg = ""
        
        # Name the output folder exactly after the ID
        out_folder_name = case_id
        out_folder_path = os.path.join(output_root, out_folder_name)
        
        # Classification variables
        overall_class = ""
        max_hab_idx = ""
        hab_tiles_sorted = ""
        max_chk_idx = ""
        chk_tiles_sorted = ""
        all_tiles = ""
        
        try:
            # --- STAGE 1: Fetch Pre-Determined Item ---
            failed_stage = "1_Fetch_Item"
            selected_item = fetch_specific_stac_item(sat_item_id)
            
            if not selected_item:
                raise ValueError(f"Could not retrieve item {sat_item_id} from Planetary Computer.")
                
            final_date = selected_item.datetime.strftime("%Y-%m-%d")
            
            if not os.path.exists(out_folder_path):
                os.makedirs(out_folder_path)

            # --- STAGE 2: Extract Tiles ---
            failed_stage = "2_Extract_Tiles"
            extract_ok = extract_and_save_tile(
                item=selected_item, lat=lat, lon=lon, case_id=uid,
                initial_pixel_size=BOX_SIDE_PIXELS, save_data=True,
                output_path=out_folder_path, print_images=True
            )
            
            if not extract_ok:
                raise ValueError("Tile extraction failed. Water thresholds or download bounds likely not met.")

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
                    indices_list = []
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
                        
                        indices_list.append((tile_name, total_index))
                        
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
                        
                    if indices_list:
                        all_tiles = " | ".join([f"{name} ({score})" for name, score in indices_list])
            
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
                overall_class, max_hab_idx, hab_tiles_sorted, max_chk_idx, chk_tiles_sorted,
                all_tiles
            ]
            with open(log_csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
                f.flush()

    logging.info(f"\nAll rows processed. Summary report saved to: {log_csv_path}")


if __name__ == "__main__":
    MODEL_PATHS = {
        "res18_scl":    "/vol/Amfitrite/CNNs_for_annotation/res18_scl/best_epoch_16.pth",
        "res18_no_scl": "/vol/Amfitrite/CNNs_for_annotation/res18_no_scl/best_epoch_26.pth",
        "convnext_scl": "/vol/Amfitrite/CNNs_for_annotation/convnext_scl/best_epoch_15.pth",
        "rdnet_no_scl": "/vol/Amfitrite/CNNs_for_annotation/rdnet_no_scl/best_epoch_35.pth"
    }
    
    parser = argparse.ArgumentParser(description="Phase 2: Heavy Execution Pipeline (Arctic Adaptation)")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV/Excel file")
    parser.add_argument("--output", type=str, required=True, help="Path to the root output folder")
    args = parser.parse_args()
    
    run_heavy_pipeline(args.input, args.output, MODEL_PATHS)