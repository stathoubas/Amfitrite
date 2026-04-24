# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:52:14 2026

@author: K. Pikounis
"""

import os
import rasterio
import numpy as np
import json
import pandas as pd
import argparse

def find_pure_tiles(base_folder_path, output_excel_path):
    print(f"Scanning directory: {base_folder_path}")
    print("Looking for pure cloud (>95%) or pure land (>95%) tiles...")
    
    results = []
    processed_count = 0
    found_count = 0
    
    # Recursively walk through all subdirectories
    for root, dirs, files in os.walk(base_folder_path):
        if "SCL_raw.tif" in files:
            scl_path = os.path.join(root, "SCL_raw.tif")
            processed_count += 1
            
            try:
                # 1. Read the SCL image
                with rasterio.open(scl_path) as src:
                    scl_data = src.read(1)
                
                total_pixels = scl_data.size
                
                # 2. Calculate Cloud and Land pixels (Excluding 7 for purity)
                # Clouds: 3 (Shadows), 8 (Med Cloud), 9 (High Cloud), 10 (Cirrus)
                cloud_mask = np.isin(scl_data, [3, 8, 9, 10])
                cloud_pixels = np.sum(cloud_mask)
                cloud_pct = (cloud_pixels / total_pixels) * 100.0
                
                # Land: 4 (Vegetation), 5 (Bare soil), 11 (Snow/Ice)
                land_mask = np.isin(scl_data, [4, 5, 11])
                land_pixels = np.sum(land_mask)
                land_pct = (land_pixels / total_pixels) * 100.0
                
                # 3. Check if it meets the > 95% threshold
                if cloud_pct > 95.0 or land_pct > 95.0:
                    
                    tile_type = "Cloud" if cloud_pct > 95.0 else "Land"
                    
                    # 4. Extract Lat/Lon from metadata.json
                    json_path = os.path.join(root, "metadata.json")
                    center_lat, center_lon = None, None
                    
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as jf:
                            meta = json.load(jf)
                            center_lat = meta.get("center_lat", None)
                            center_lon = meta.get("center_lon", None)
                    else:
                        print(f"  [Warning] metadata.json missing in: {root}")
                    
                    # 5. Append to results
                    results.append({
                        "Folder_Path": root,
                        "Tile_Type": tile_type,
                        "Cloud_Percentage": round(cloud_pct, 2),
                        "Land_Percentage": round(land_pct, 2),
                        "center_lat": center_lat,
                        "center_lon": center_lon
                    })
                    
                    found_count += 1
                    print(f"  -> Found {tile_type} Tile! (Clouds: {cloud_pct:.1f}%, Land: {land_pct:.1f}%) | {root}")
                    
            except Exception as e:
                print(f"Error processing {scl_path}: {e}")
                
            # Print a status update every 500 folders
            if processed_count % 500 == 0:
                print(f"... Scanned {processed_count} folders. Found {found_count} pure tiles so far.")

    # 6. Save to Excel
    print(f"\nScan complete! Scanned {processed_count} total SCL files.")
    if results:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_excel_path)), exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_excel(output_excel_path, index=False)
        print(f"Saved {len(df)} pure tiles to: {output_excel_path}")
    else:
        print("No tiles met the >95% threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan folders for >95% Pure Land or Cloud Sentinel-2 tiles.")
    parser.add_argument("--input", type=str, required=True, help="Base directory path to scan")
    parser.add_argument("--output", type=str, required=True, help="Path for the output Excel file (.xlsx)")
    
    args = parser.parse_args()
    
    find_pure_tiles(args.input, args.output)