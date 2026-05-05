# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:00:06 2026

@author: K. Pikounis
"""

import os
import time
import shutil
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

def get_optimal_crop(scl_path, cyfi_csv_path):
    """Finds the optimal 256x256 crop window based on the scoring algorithm."""
    # 1. Load SCL array
    with rasterio.open(scl_path) as src:
        scl_data = src.read(1)
        h, w = scl_data.shape

    # If the image is already 256x256 or smaller, return 0,0
    if h <= 256 and w <= 256:
        return 0, 0

    # 2. Map CyFi points to 2D numpy arrays for O(1) lookup speed
    high_grid = np.zeros((h, w), dtype=int)
    mod_grid = np.zeros((h, w), dtype=int)
    low_grid = np.zeros((h, w), dtype=int)

    if os.path.exists(cyfi_csv_path):
        cyfi_df = pd.read_csv(cyfi_csv_path)
        for _, row in cyfi_df.iterrows():
            r, c = int(row['pixel_row']), int(row['pixel_col'])
            sev = str(row['severity']).strip().lower()
            if r < h and c < w:
                if sev == 'high':
                    high_grid[r, c] = 1
                elif sev == 'moderate':
                    mod_grid[r, c] = 1
                elif sev == 'low':
                    low_grid[r, c] = 1

    max_r_offset = h - 256
    max_c_offset = w - 256

    def score_window(r, c):
        window_water = np.sum(scl_data[r:r+256, c:c+256] == 6)
        window_high = np.sum(high_grid[r:r+256, c:c+256])
        window_mod = np.sum(mod_grid[r:r+256, c:c+256])
        
        water_norm = window_water / 65536.0
        high_norm = window_high / 676.0
        mod_norm = window_mod / 676.0
        
        return (water_norm * 0.2) + (high_norm * 1.0) + (mod_norm * 0.5)

    # 3. Coarse Search (Stride 10)
    best_score = -1.0
    best_coarse_r, best_coarse_c = 0, 0

    for r in range(0, max_r_offset + 1, 10):
        for c in range(0, max_c_offset + 1, 10):
            score = score_window(r, c)
            if score > best_score:
                best_score = score
                best_coarse_r, best_coarse_c = r, c

    # 4. Fine Search (Stride 1 around the coarse winner)
    best_score = -1.0
    best_r, best_c = best_coarse_r, best_coarse_c
    
    r_start = max(0, best_coarse_r - 9)
    r_end = min(max_r_offset, best_coarse_r + 9)
    c_start = max(0, best_coarse_c - 9)
    c_end = min(max_c_offset, best_coarse_c + 9)

    for r in range(r_start, r_end + 1):
        for c in range(c_start, c_end + 1):
            score = score_window(r, c)
            if score > best_score:
                best_score = score
                best_r, best_c = r, c

    return best_r, best_c

def process_dataset(excel_path, input_dir, output_dir, output_excel_path):
    # Load Excel
    df = pd.read_excel(excel_path)
    
    # Create new columns for tracking
    df['crop_row'] = None
    df['crop_col'] = None
    df['new_water_pixels'] = None
    df['new_high_pred'] = None
    df['new_mod_pred'] = None
    df['new_low_pred'] = None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_rows = len(df)
    start_time = time.time()
    
    print(f"Starting processing of {total_rows} tiles...")

    for idx, row in df.iterrows():
        # Find matching folder in INPUT_DIR based on 'uid'
        uid = str(row['uid'])
        folder_name = None
        
        # Searching for the specific folder
        for d in os.listdir(input_dir):
            if d.startswith(uid):  # Matches "104_HAB" or "399_2017-07-14"
                folder_name = d
                break
                
        if not folder_name:
            print(f"[Warning] Folder for UID {uid} not found. Skipping.")
            continue
            
        src_folder = os.path.join(input_dir, folder_name)
        dst_folder = os.path.join(output_dir, folder_name)
        
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        # File paths
        scl_path = os.path.join(src_folder, "SCL_raw.tif")
        cyfi_csv_path = os.path.join(src_folder, "cyfi_lattice_predictions.csv")
        
        if not os.path.exists(scl_path):
            print(f"[Warning] SCL_raw.tif missing in {folder_name}. Skipping.")
            continue

        # 1. FIND OPTIMAL CROP
        best_r, best_c = get_optimal_crop(scl_path, cyfi_csv_path)
        
        # 2. GET NEW METRICS
        with rasterio.open(scl_path) as src:
            scl_data = src.read(1)
        new_water = np.sum(scl_data[best_r:best_r+256, best_c:best_c+256] == 6)
        
        new_high, new_mod, new_low = 0, 0, 0
        if os.path.exists(cyfi_csv_path):
            cyfi_df = pd.read_csv(cyfi_csv_path)
            # Filter points that fall inside our new 256x256 bounding box
            mask = (cyfi_df['pixel_row'] >= best_r) & (cyfi_df['pixel_row'] < best_r + 256) & \
                   (cyfi_df['pixel_col'] >= best_c) & (cyfi_df['pixel_col'] < best_c + 256)
            cropped_cyfi = cyfi_df[mask]
            
            counts = cropped_cyfi['severity'].str.lower().value_counts()
            new_high = counts.get('high', 0)
            new_mod = counts.get('moderate', 0)
            new_low = counts.get('low', 0)

        # Update DataFrame
        df.at[idx, 'crop_row'] = best_r
        df.at[idx, 'crop_col'] = best_c
        df.at[idx, 'new_water_pixels'] = new_water
        df.at[idx, 'new_high_pred'] = new_high
        df.at[idx, 'new_mod_pred'] = new_mod
        df.at[idx, 'new_low_pred'] = new_low

        # 3. CROP AND COPY FILES
        for filename in os.listdir(src_folder):
            src_file = os.path.join(src_folder, filename)
            dst_file = os.path.join(dst_folder, filename)
            
            if filename.endswith(".tif"):
                # Use Rasterio to crop and preserve georeferencing
                with rasterio.open(src_file) as src:
                    window = Window(col_off=best_c, row_off=best_r, width=256, height=256)
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'height': window.height,
                        'width': window.width,
                        'transform': rasterio.windows.transform(window, src.transform)
                    })
                    with rasterio.open(dst_file, 'w', **kwargs) as dst:
                        dst.write(src.read(window=window))
                        
            elif filename.endswith(".png"):
                # Use PIL for standard images
                try:
                    img = Image.open(src_file)
                    # left, upper, right, lower
                    cropped_img = img.crop((best_c, best_r, best_c + 256, best_r + 256))
                    cropped_img.save(dst_file)
                except Exception as e:
                    print(f"Error cropping {filename}: {e}")
                    
            elif filename in ["metadata.json", "cyfi_lattice_predictions.csv"]:
                # Just copy metadata/csv files as they are
                shutil.copy2(src_file, dst_file)

        # 4. PROGRESS TRACKING
        count = idx + 1
        if count % 100 == 0 or count == total_rows:
            elapsed = time.time() - start_time
            avg_time = elapsed / count
            remaining = total_rows - count
            eta_seconds = remaining * avg_time
            eta_minutes = eta_seconds / 60.0
            print(f"Done {count}/{total_rows} in {elapsed:.1f}s | Expecting {eta_minutes:.1f} min for the rest ({remaining} left)")

    # Save enriched Excel file
    df.to_excel(output_excel_path, index=False)
    print(f"\nFinished! Master file saved to: {output_excel_path}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    EXCEL_PATH = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\dataset_summary_for_post_processing.xlsx"
    INPUT_DIR = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\data"
    OUTPUT_DIR = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\data256"
    OUTPUT_EXCEL = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\dataset_summary_for_post_processing_256.xlsx"
    
    process_dataset(EXCEL_PATH, INPUT_DIR, OUTPUT_DIR, OUTPUT_EXCEL)