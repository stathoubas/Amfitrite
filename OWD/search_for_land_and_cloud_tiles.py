import os
import rasterio
import numpy as np
import json
import pandas as pd

def find_pure_tiles_bulk(base_dir, folder_list, output_csv_path):
    print(f"Starting bulk scan across {len(folder_list)} datasets in: {base_dir}")
    print("Looking for pure cloud (>95%) or pure land (>95%) tiles...")
    
    results = []
    global_processed_count = 0
    global_found_count = 0
    
    # Iterate through your specific list of folders
    for folder_name in folder_list:
        target_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(target_path):
            print(f"\n[!] Warning: Folder not found, skipping -> {target_path}")
            continue
            
        print(f"\n--- Scanning Dataset: {folder_name} ---")
        
        # Recursively walk through all subdirectories in the current dataset
        for root, dirs, files in os.walk(target_path):
            if "SCL_raw.tif" in files:
                scl_path = os.path.join(root, "SCL_raw.tif")
                global_processed_count += 1
                
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
                            "Dataset_Source": folder_name,
                            "Folder_Path": root,
                            "Tile_Type": tile_type,
                            "Cloud_Percentage": round(cloud_pct, 2),
                            "Land_Percentage": round(land_pct, 2),
                            "center_lat": center_lat,
                            "center_lon": center_lon
                        })
                        
                        global_found_count += 1
                        print(f"  -> Found {tile_type}! (Clouds: {cloud_pct:.1f}%, Land: {land_pct:.1f}%) | {root}")
                        
                except Exception as e:
                    print(f"Error processing {scl_path}: {e}")
                    
                # Print a status update every 1000 folders
                if global_processed_count % 1000 == 0:
                    print(f"... Scanned {global_processed_count} total folders. Found {global_found_count} pure tiles so far.")

    # 6. Save to CSV
    print(f"\n" + "="*50)
    print(f"Bulk scan complete! Scanned {global_processed_count} total SCL files across all datasets.")
    
    if results:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print(f"Saved {len(df)} pure tiles to: {output_csv_path}")
    else:
        print("No tiles met the >95% threshold.")


if __name__ == "__main__":
    BASE_DIRECTORY = "/vol2/Amfitrite/OWD"
    
    TARGET_FOLDERS = [
        "Iains_dataset_v3",
        "Florida_v3",
        "obis_data_v3",
        "prd_oas_v3",
        "Arctic_v3",
        "California_v3",
        "Sweden_v3",
        "REPHY_v3"
    ]
    
    OUTPUT_CSV = "/vol2/Amfitrite/OWD/processed_results/land_and_cloud_tiles.csv"
    
    find_pure_tiles_bulk(BASE_DIRECTORY, TARGET_FOLDERS, OUTPUT_CSV)