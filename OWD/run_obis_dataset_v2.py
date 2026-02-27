# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:43:02 2026

@author: K. Pikounis
added stop & resume functionality where it left off
"""

import pandas as pd
import os
import argparse
import logging
import traceback
import csv

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
    
    # Also print to console
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
    
    # If the summary report already exists, read it to find completed/failed IDs
    if os.path.exists(log_csv_path):
        logging.info(f"Found existing summary report: {log_csv_path}. Checking for processed cases...")
        try:
            existing_log_df = pd.read_csv(log_csv_path)
            # Add all previously attempted case_ids to the set so we don't retry broken ones
            processed_cases = set(existing_log_df['case_id'].astype(str))
            logging.info(f"Found {len(processed_cases)} previously processed cases. These will be skipped.")
            
            # Open in Append mode so we don't overwrite the header or previous results
            log_file_mode = 'a'
            write_header = False
        except Exception as e:
            logging.warning(f"Could not parse existing summary report. Starting fresh. Error: {e}")
            log_file_mode = 'w'
            write_header = True
    else:
        log_file_mode = 'w'
        write_header = True

    # Initialize the CSV report file (Write or Append)
    with open(log_csv_path, mode=log_file_mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["case_id", "date", "status", "failed_stage", "error_message", "output_folder"])
        
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
        # --- SAFE EXTRACTION FROM YOUR SPECIFIC CSV FORMAT ---
        raw_id = str(row.get('id_x', f"Row_{index}"))
        # Replace colons so Windows doesn't crash when creating folders
        case_id = raw_id.replace(":", "_") 
        
        # --- CHECK IF ALREADY PROCESSED ---
        if case_id in processed_cases:
            logging.info(f"Row {index+1}: Case {case_id} already processed. Skipping.")
            continue
        
        try:
            lat = float(row.get('decimalLatitude'))
            lon = float(row.get('decimalLongitude'))
        except (ValueError, TypeError):
            logging.error(f"Row {index} ({case_id}): Invalid or missing Lat/Lon. Skipping.")
            continue
            
        raw_date = str(row.get('eventDate'))
        try:
            # Convert "6/13/2017" into "2017-06-13"
            parsed_date = pd.to_datetime(raw_date)
            date_str = parsed_date.strftime("%Y-%m-%d")
        except Exception as e:
            logging.error(f"Row {index} ({case_id}): Invalid Date format '{raw_date}'. Skipping.")
            continue
            
        uid = f"{case_id}_{date_str}"
        logging.info(f"\n{'='*60}\nProcessing Case: {uid} (Row {index+1}/{len(df)})\n{'='*60}")
        
        status = "SUCCESS"
        failed_stage = "None"
        error_msg = ""
        out_folder_path = ""
        
        try:
            # --- STAGE 1: Search & Select ---
            failed_stage = "1_Search_and_Select"
            items_df = find_satelite_images(lat, lon, date_str, meter_buffer=SEARCH_BUFFER_METERS)
            proced, best_item_index, selected_item = select_item(items_df)
            
            if not proced:
                raise ValueError("No suitable satellite items found or cloud coverage too high.")
                
            final_date = selected_item.datetime.strftime("%Y-%m-%d")
            
            # Create safe folder name
            out_folder_name = f"case_{case_id}_{final_date}_item{best_item_index}"
            out_folder_path = os.path.join(output_root, out_folder_name)
            
            if not os.path.exists(out_folder_path):
                os.makedirs(out_folder_path)

            # --- STAGE 2: Extract Tiles ---
            failed_stage = "2_Extract_Tiles"
            extract_ok = extract_and_save_tile(
                item=selected_item,
                lat=lat,
                lon=lon,
                case_id=uid,
                initial_pixel_size=BOX_SIDE_PIXELS,
                save_data=True,
                output_path=out_folder_path,
                print_images=True
            )
            
            if not extract_ok:
                raise ValueError("Tile extraction failed (e.g., bounds error).")

            # --- STAGE 3: CyFi Pipeline ---
            failed_stage = "3_CyFi_Pipeline"
            final_paths = predict_using_cyfi_pipeline(out_folder_path, final_date)

            # --- STAGE 4: CNN Inference ---
            failed_stage = "4_CNN_Inference"
            run_single_folder(out_folder_path, model_paths)

            # --- STAGE 5: Summarize & Collage ---
            failed_stage = "5_Folder_Summary"
            df_summary = folder_summary(out_folder_path, add_images_to_excel=True)
            
            failed_stage = "None" 
            logging.info(f"Successfully completed {uid}")

        except Exception as e:
            status = "FAILED"
            error_msg = str(e).replace('\n', ' ')
            logging.error(f"Failed at {failed_stage} for {uid}. Error: {error_msg}")
            logging.debug(traceback.format_exc()) 
            
        finally:
            # Write the result to the CSV report immediately and flush buffer
            with open(log_csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([case_id, date_str, status, failed_stage, error_msg, out_folder_path])
                f.flush()
                
    logging.info(f"\nAll rows processed. Summary report saved to: {log_csv_path}")


if __name__ == "__main__":
    # Define Model Paths here
    MODEL_PATHS = {
        "res18_scl":    "/vol/Amfitrite/CNNs_for_annotation/res18_scl/best_epoch_16.pth",
        "res18_no_scl": "/vol/Amfitrite/CNNs_for_annotation/res18_no_scl/best_epoch_26.pth",
        "convnext_scl": "/vol/Amfitrite/CNNs_for_annotation/convnext_scl/best_epoch_15.pth",
        "rdnet_no_scl": "/vol/Amfitrite/CNNs_for_annotation/rdnet_no_scl/best_epoch_35.pthh"
    }
    
    # Set up Argparse so you can run it from the terminal easily
    parser = argparse.ArgumentParser(description="Automated HAB Satellite Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV/Excel file")
    parser.add_argument("--output", type=str, required=True, help="Path to the root output folder")
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_automated_pipeline(args.input, args.output, MODEL_PATHS)