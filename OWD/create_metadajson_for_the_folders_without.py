# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 23:30:48 2026

@author: K. Pikounis
"""

import os
import json
import pandas as pd
import numpy as np
import rasterio

def generate_missing_metadata(output_directory, csv_path):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Ensure ID column is treated as a string for robust matching
    if 'ID' in df.columns:
        df['ID'] = df['ID'].astype(str)
    else:
        print("[!] Error: CSV does not contain an 'ID' column.")
        return

    processed_count = 0
    missing_scl_count = 0
    missing_in_csv_count = 0
    multiple_in_csv_count = 0

    print(f"Scanning directory: {output_directory}")
    print("=" * 60)

    for folder_name in os.listdir(output_directory):
        folder_path = os.path.join(output_directory, folder_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue
            
        json_path = os.path.join(folder_path, "metadata.json")
        
        # Only process folders missing the metadata.json
        if not os.path.exists(json_path):
            # Extract ID from the folder name (before the first '_')
            folder_id = folder_name.split('_')[0]
            
            # Find the matching row in the CSV
            matching_rows = df[df['ID'] == folder_id]
            
            if matching_rows.empty:
                print(f"  [Warning] ID '{folder_id}' not found in CSV. Skipping {folder_name}.")
                missing_in_csv_count += 1
                continue
            if len(matching_rows) > 1:
                print(f"  [Warning] ID '{folder_id}' found in CSV {len(matching_rows)} times. Skipping {folder_name}.")
                multiple_in_csv_count += 1
                continue
                
            # Take the first matching row
            row = matching_rows.iloc[0]
            
            # 1. Parse Data from CSV
            sat_item = str(row.get('sat_item', 'N/A'))
            center_lat = row.get('center_lat', None)
            center_lon = row.get('center_lon', None)
            high_counts = int(row.get('CyFi_High', 0))
            mod_counts = int(row.get('CyFi_Moderate', 0))
            low_counts = int(row.get('CyFi_Low', 0))
            
            # 2. Extract Date from sat_item
            # Example: S2A_MSIL2A_20160807T153002... -> '20160807' -> '2016-08-07'
            date_formatted = "N/A"
            if sat_item != "N/A" and "_" in sat_item:
                try:
                    date_part = sat_item.split('_')[2].split('T')[0]
                    if len(date_part) == 8:
                        date_formatted = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
                except Exception:
                    pass

            # 3. Process SCL pixels
            scl_path = os.path.join(folder_path, "SCL_raw.tif")
            if not os.path.exists(scl_path):
                print(f"  [Warning] Missing SCL_raw.tif in {folder_name}. Cannot calculate pixels.")
                missing_scl_count += 1
                continue
                
            try:
                with rasterio.open(scl_path) as src:
                    scl_data = src.read(1)
                    
                # Calculate Pixels
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
                    
            except Exception as e:
                print(f"  [Error] Failed to read SCL_raw.tif in {folder_name}: {e}")
                missing_scl_count += 1
                continue

            # 4. Construct JSON payload
            new_meta = {
                "sat_item": sat_item,
                "date": date_formatted,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "num_pixels": int(num_pixels),
                "water_pixels": int(water_pixels),
                "per_clouds": per_clouds,
                "High counts": high_counts,
                "Moderate counts": mod_counts,
                "Low counts": low_counts
            }
            
            # 5. Write to metadata.json
            with open(json_path, 'w') as jf:
                json.dump(new_meta, jf, indent=4)
                
            processed_count += 1
            print(f"  -> Generated metadata.json for {folder_name}")

    print("=" * 60)
    print("Metadata Generation Complete!")
    print(f"Successfully generated: {processed_count} files.")
    if missing_in_csv_count > 0:
        print(f"Folders skipped (ID not in CSV): {missing_in_csv_count}")
    if missing_scl_count > 0:
        print(f"Folders skipped (Missing/Corrupted SCL_raw.tif): {missing_scl_count}")

if __name__ == "__main__":
    # UPDATE THESE PATHS
    TARGET_DIRECTORY = "/vol2/Amfitrite/OWD/dataset_v1"
    REFERENCE_CSV = "/vol2/Amfitrite/OWD/processed_results/merged_2_and_imputed_v2.csv"
    
    generate_missing_metadata(TARGET_DIRECTORY, REFERENCE_CSV)