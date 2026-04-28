# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:33:09 2026

@author: K. Pikounis
"""

import os
import shutil
import json
import pandas as pd
import numpy as np
import rasterio
import glob

def aggregate_dataset(input_csv, output_dir):
    print(f"Loading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Ensure the master output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    total_rows = len(df)
    processed = 0
    errors = 0
    
    print(f"Starting aggregation for {total_rows} tiles...\n" + "="*50)
    
    for index, row in df.iterrows():
        try:
            # 1. Parse row data safely
            row_id = str(row['ID'])
            orig_path = str(row['tile_folder_path'])
            
            # Convert to uppercase for robust comparison
            is_hab_str = str(row.get('tile_is_hab', '')).strip().upper()
            status_str = str(row.get('tile_status', '')).strip().lower()
            
            # 2. Determine folder suffix
            if is_hab_str == 'TRUE':
                suffix = "_HAB"
            else:
                if status_str == 'clouds':
                    suffix = "_clouds"
                elif status_str == 'land':
                    suffix = "_land"
                else:
                    suffix = "_nonHAB"
                    
            folder_name = f"{row_id}{suffix}"
            new_folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            # 3. Copy .tif, .tiff, and cyfi_prediction_map.png
            files_to_copy = []
            files_to_copy.extend(glob.glob(os.path.join(orig_path, "*.tif")))
            files_to_copy.extend(glob.glob(os.path.join(orig_path, "*.tiff")))
            
            png_path = os.path.join(orig_path, "cyfi_prediction_map.png")
            if os.path.exists(png_path):
                files_to_copy.append(png_path)
                
            for file_path in files_to_copy:
                shutil.copy(file_path, new_folder_path)
                
            # 4. Process Metadata and SCL pixels
            orig_json_path = os.path.join(orig_path, "metadata.json")
            if not os.path.exists(orig_json_path):
                print(f"  [Warning] Missing metadata.json for ID {row_id}")
                errors += 1
                continue
                
            with open(orig_json_path, 'r') as jf:
                old_meta = json.load(jf)
                
            # Open the freshly copied SCL image for pixel calculations
            new_scl_path = os.path.join(new_folder_path, "SCL_raw.tif")
            if not os.path.exists(new_scl_path):
                print(f"  [Warning] Missing SCL_raw.tif for ID {row_id}")
                errors += 1
                continue
                
            with rasterio.open(new_scl_path) as src:
                scl_data = src.read(1)
                
            # Sentinel-2 SCL Class 0 is "No Data". We treat > 0 as non-corrupted.
            non_corrupted_mask = (scl_data > 0)
            num_pixels = np.sum(non_corrupted_mask)
            
            water_pixels = np.sum(scl_data == 6)
            
            # Clouds: 3(Shadows), 8(Med Cloud), 9(High Cloud), 10(Cirrus)
            cloud_mask = np.isin(scl_data, [3, 8, 9, 10])
            cloud_pixels = np.sum(cloud_mask)
            
            if num_pixels > 0:
                per_clouds = round((cloud_pixels / num_pixels) * 100.0, 2)
            else:
                per_clouds = 0.0
                
            # 5. Create the new, clean JSON structure
            # Notice we capture keys exactly as they appear in your original JSON, 
            # including the trailing spaces on "Moderate counts " and "Low counts "
            new_meta = {
                "sat_item": old_meta.get("item_id", "N/A"),
                "date": old_meta.get("date", "N/A"),
                "center_lat": old_meta.get("center_lat", None),
                "center_lon": old_meta.get("center_lon", None),
                "num_pixels": int(num_pixels),
                "water_pixels": int(water_pixels),
                "per_clouds": per_clouds,
                "High counts": old_meta.get("High counts", 0),
                "Moderate counts": old_meta.get("Moderate counts ", 0), # Cleaned key name
                "Low counts": old_meta.get("Low counts ", 0)            # Cleaned key name
            }
            
            new_json_path = os.path.join(new_folder_path, "metadata.json")
            with open(new_json_path, 'w') as jf:
                json.dump(new_meta, jf, indent=4)
                
            processed += 1
            if processed % 500 == 0:
                print(f"  -> Aggregated {processed}/{total_rows} tiles...")
                
        except Exception as e:
            print(f"  [Error] Failed processing row {index} (ID: {row.get('ID')}): {e}")
            errors += 1

    print("="*50)
    print(f"Aggregation Complete!")
    print(f"Successfully processed: {processed}")
    print(f"Errors/Skipped: {errors}")
    print(f"Output Directory: {output_dir}")

if __name__ == "__main__":
    # UPDATE THESE PATHS
    INPUT_CSV_PATH = "/vol2/Amfitrite/OWD/processed_results/merged_2_and_imputed_v2.csv"
    OUTPUT_DIRECTORY = "/vol2/Amfitrite/OWD/dataset_v1"
    
    aggregate_dataset(INPUT_CSV_PATH, OUTPUT_DIRECTORY)