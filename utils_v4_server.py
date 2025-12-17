# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 10:35:31 2025

@author: K. Pikounis

based on utils_v4
Server-ready utilities with File Locking to safely handle 
concurrent writes to Excel reports from multiple parallel scripts.


Modifications:
1. Fixed JSON serialization error in extract_and_save_tile (handles Timestamps).
2. Server-ready utilities with File Locking. Merges original CyFi/SCL logic with FileLock mechanisms for safe multi-process reporting.
"""

import json
import os
import time
import shutil
import tempfile
import io
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

import rasterio
import rioxarray
from rasterio.transform import rowcol
from pyproj import Transformer, Geod
from PIL import Image as PILImage

# CyFi imports
from cyfi.pipeline import CyFiPipeline
from cyfi.config import FeaturesConfig
from cyfi.data.features import generate_all_features
from cyfi.cli import DEFAULT_MODEL_PATH
import planetary_computer as pc


# =============================================================================
# 1. FILE LOCKING MECHANISM (For Safe Parallel Reporting)
# =============================================================================
class SimpleFileLock:
    """
    A simple cross-platform lock using os.open with O_EXCL.
    Ensures that only one process can write to the specific file at a time.
    """
    def __init__(self, file_path, timeout=120):
        self.lock_file = str(file_path) + ".lock"
        self.timeout = timeout
        self.fd = None

    def acquire(self):
        start_time = time.time()
        while True:
            try:
                # O_CREAT | O_EXCL ensures atomic creation. Fails if file exists.
                self.fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return True
            except OSError:
                # Lock exists, wait and retry
                if time.time() - start_time > self.timeout:
                    print(f"!!! Timeout waiting for lock: {self.lock_file} !!!")
                    return False # Failed to acquire
                time.sleep(0.2 + (time.time() % 0.5)) # Add jitter

    def release(self):
        if self.fd:
            os.close(self.fd)
            self.fd = None
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except OSError:
            pass 

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# =============================================================================
# 2. GEOMETRIC & HELPER FUNCTIONS
# =============================================================================

def get_bounding_box(latitude, longitude, meter_buffer=50000):
    """
    Returns a bounding box [min_lon, min_lat, max_lon, max_lat] around the point.
    """
    g = Geod(ellps='WGS84')
    lon_west, _, _ = g.fwd(longitude, latitude, 270, meter_buffer)
    _, lat_south, _ = g.fwd(longitude, latitude, 180, meter_buffer)
    lon_east, _, _ = g.fwd(longitude, latitude, 90, meter_buffer)
    _, lat_north, _ = g.fwd(longitude, latitude, 0, meter_buffer)
    return [lon_west, lat_south, lon_east, lat_north]

def compress_image_to_bytes(image_path, target_img_size=(200, 200)):
    """Reads an image, resizes it, and returns the byte stream for Excel embedding."""
    try:
        with PILImage.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            img.thumbnail(target_img_size, PILImage.LANCZOS)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', optimize=True)
            img_byte_arr.seek(0)
            return img_byte_arr, img.width, img.height
    except Exception as e:
        print(f"Error compressing image {image_path}: {e}")
        return None, 0, 0

def select_item(items_df):
    """
    Selects the best item from a DataFrame of STAC items.
    Priority: 
    1. Cloud cover < 7.5% (prefer lower)
    2. Date difference (prefer closer to target)
    """
    if items_df.empty: return None, False
    
    # 1. Strict Cloud Filter (as per original utils)
    low_clouds = items_df[items_df.per_clouds < 7.5].copy()
    if len(low_clouds) == 0:
        return None, False
    
    low_clouds["abs_date_difference"] = low_clouds["date_difference"].abs()
    
    # If only one, return it
    if len(low_clouds) == 1:
        return low_clouds.iloc[0], True
        
    # Sort by clouds
    low_clouds.sort_values(by="per_clouds", inplace=True)
    
    # If we have a very clear image (<2%), take the one closest in time
    if low_clouds.iloc[0]["per_clouds"] < 2.0:
        sel_items = low_clouds[low_clouds["per_clouds"] < 2.0].copy()
        sel_items.sort_values(by="abs_date_difference", inplace=True)
        return sel_items.iloc[0], True
        
    # Otherwise just take the least cloudy one
    return low_clouds.iloc[0], True


def extract_and_save_tile(item, points_df, initial_pixel_size=365, save_data=True, output_path=".", print_images=False):
    """
    Downloads the SCL and Band data for the given STAC item, clips it to the bounding box
    of the points, and saves it to the output path.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        
        # 1. Calculate BBox from points + Buffer
        # (Using logic from original script to ensure coverage)
        lat_min, lat_max = points_df.lat.min(), points_df.lat.max()
        lon_min, lon_max = points_df.lon.min(), points_df.lon.max()
        
        # Calculate center and large enough buffer (e.g. 3000m)
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        bbox = get_bounding_box(center_lat, center_lon, 3000)
        
        # 2. Define Assets to Fetch
        # Using config bands implicitly, but ensuring we get SCL + Visual for reports
        assets = ["SCL", "B02", "B03", "B04", "B08", "visual"] 
        
        for asset_key in assets:
            if asset_key not in item.assets: continue
            
            # Sign URL
            href = pc.sign(item.assets[asset_key].href)
            
            # Open and Clip
            ds = rioxarray.open_rasterio(href)
            ds_clip = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326")
            
            # Save
            out_name = "visual_raw.tif" if asset_key == "visual" else f"{asset_key}_raw.tif"
            ds_clip.rio.to_raster(os.path.join(output_path, out_name))
            
        # 3. Save Metadata/Points for CyFi
        # [FIX]: default=str handles Timestamp objects correctly
        metadata = {
            "points_data": points_df.to_dict('records'),
            "additional_points": []
        }
        
        with open(os.path.join(output_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, default=str)
            
        return True

    except Exception as e:
        print(f"Extraction Failed: {e}")
        # import traceback
        # traceback.print_exc()
        return False


# =============================================================================
# 3. CORE CYFI LOGIC (Predict & Visualize)
# =============================================================================

def predict_using_cyfi_pipeline(input_folder_path, 
                                date_str, 
                                metadata_filename="metadata.json",
                                print_images=False):
    """
    Full implementation of the CyFi pipeline logic:
    1. Loads raw tifs.
    2. Generates grid points over water (SCL=6).
    3. Runs cloud filtering (cache generation).
    4. Runs CyFi feature generation & prediction.
    5. Updates metadata & creates visualization.
    """
    input_dir = Path(input_folder_path).resolve()
    print(f"--- Starting CyFi Pipeline: {input_dir.name} ---")

    # 1. Configuration
    features_config = FeaturesConfig()
    CLOUD_THRESHOLD = 0.075 
    features_config.max_cloud_percent = CLOUD_THRESHOLD

    WINDOW_METERS = features_config.image_feature_meter_window 
    PIXEL_SIZE = 10
    WINDOW_PIXELS = WINDOW_METERS // PIXEL_SIZE 
    RADIUS_PIXELS = WINDOW_PIXELS // 2 

    # 2. Load Raw Raster Data & Metadata
    required_bands = features_config.use_sentinel_bands 
    data_store = {}
    
    try:
        scl_da = rioxarray.open_rasterio(input_dir / "SCL_raw.tif").squeeze()
        data_store["SCL"] = scl_da.values
        height, width = scl_da.shape
        transform = scl_da.rio.transform()
        crs = scl_da.rio.crs
    except FileNotFoundError:
        print("Error: SCL_raw.tif not found. Extraction likely failed.")
        return (False, str(input_dir))

    # Load Metadata
    metadata_path = input_dir / metadata_filename
    metadata = {}
    all_reference_points = []
    
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            all_reference_points.extend(metadata.get("points_data", []))
            all_reference_points.extend(metadata.get("additional_points", []))
    except FileNotFoundError:
        print(f"Warning: {metadata_filename} not found.")

    # Load other bands (fill nan if missing)
    for band in required_bands:
        if band == "SCL": continue
        path = input_dir / f"{band}_raw.tif"
        if path.exists():
            data_store[band] = rioxarray.open_rasterio(path).squeeze().values
        else:
            data_store[band] = np.full((height, width), np.nan, dtype=np.float32)

    # 3. Generate Lattice Grid
    # print("Generating 100m Lattice Grid...")
    GRID_STEP = 10 
    
    rows = np.arange(0, height, GRID_STEP)
    cols = np.arange(0, width, GRID_STEP)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
    grid_rows = grid_rows.flatten()
    grid_cols = grid_cols.flatten()
    
    valid_mask = (grid_rows < height) & (grid_cols < width)
    grid_rows = grid_rows[valid_mask]
    grid_cols = grid_cols[valid_mask]
    
    # Check water mask (SCL == 6 is Water)
    water_mask = (data_store["SCL"][grid_rows, grid_cols] == 6)
    target_rows = grid_rows[water_mask]
    target_cols = grid_cols[water_mask]
    
    initial_samples = len(target_rows)
    # print(f"Identified {initial_samples} potential water points.")
    
    if initial_samples == 0:
        return (False, str(input_dir))

    # 4. Create Mock Cache Structure
    temp_cache = Path(tempfile.mkdtemp(prefix="cyfi_lattice_"))
    cache_subdir = temp_cache / f"sentinel_{WINDOW_METERS}"
    fake_item_id = "LATTICE_ITEM" 

    valid_sample_ids = []
    valid_lats = []
    valid_lons = []
    valid_rows = []
    valid_cols = []

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # 5. Filter & Slice Data
    # print("Checking Cloud Cover & Slicing Data...")
    
    skipped_clouds = 0
    
    # Iterate points (removed tqdm for server logs cleanliness)
    for idx, (r, c) in enumerate(zip(target_rows, target_cols)):
        s_id = f"sample_{idx}"
        
        r_min = max(0, r - RADIUS_PIXELS)
        r_max = min(height, r + RADIUS_PIXELS)
        c_min = max(0, c - RADIUS_PIXELS)
        c_max = min(width, c + RADIUS_PIXELS)
        
        if (r_max - r_min) == 0 or (c_max - c_min) == 0:
            continue

        scl_window = data_store["SCL"][r_min:r_max, c_min:c_max]
        # Cloud(8,9), Cirrus(10), Unclassified(7)
        cloud_pixel_count = ((scl_window >= 7) & (scl_window <= 10)).sum()
        total_pixels = scl_window.size
        cloud_ratio = cloud_pixel_count / total_pixels
        
        if cloud_ratio > CLOUD_THRESHOLD:
            skipped_clouds += 1
            continue

        item_dir = cache_subdir / s_id / fake_item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        
        x, y = rasterio.transform.xy(transform, r, c)
        lon, lat = transformer.transform(x, y)
        
        valid_sample_ids.append(s_id)
        valid_lats.append(lat)
        valid_lons.append(lon)
        valid_rows.append(r)
        valid_cols.append(c)

        for band in required_bands:
            arr_window = data_store[band][r_min:r_max, c_min:c_max]
            arr_reshaped = arr_window[np.newaxis, :, :] 
            np.save(item_dir / f"{band}.npy", arr_reshaped)

    if len(valid_sample_ids) == 0:
        # print("No valid points remained after cloud filtering.")
        shutil.rmtree(temp_cache)
        return (False, str(input_dir))

    # 6. Prepare DataFrames for CyFi
    samples_df = pd.DataFrame({
        "sample_id": valid_sample_ids,
        "date": [date_str] * len(valid_sample_ids),
        "latitude": valid_lats,
        "longitude": valid_lons
    }).set_index("sample_id")

    satellite_meta_df = pd.DataFrame({
        "sample_id": valid_sample_ids,
        "item_id": [fake_item_id] * len(valid_sample_ids),
        "days_before_sample": [0] * len(valid_sample_ids), 
        "datetime": [date_str] * len(valid_sample_ids), 
        "visual_href": [None] * len(valid_sample_ids) 
    })

    # 7. Run CyFi Feature Generation
    # print("Running CyFi Feature Generation...")
    try:
        _, features_df = generate_all_features(
            samples=samples_df,
            satellite_meta=satellite_meta_df,
            config=features_config,
            cache_dir=temp_cache
        )
    except Exception as e:
        print(f"Feature generation failed: {e}")
        shutil.rmtree(temp_cache)
        return (False, str(input_dir))

    # 8. Run CyFi Prediction
    # print("Running Prediction...")
    pipeline = CyFiPipeline.from_disk(DEFAULT_MODEL_PATH)
    pipeline.predict_features = features_df
    pipeline.predict_samples = samples_df
    pipeline._predict_model()
    
    results_df = pipeline.output_df.reset_index()
    results_df["pixel_row"] = valid_rows
    results_df["pixel_col"] = valid_cols
    
    csv_path = input_dir / "cyfi_lattice_predictions.csv"
    results_df.to_csv(csv_path, index=False)

    # --- UPDATE METADATA JSON ---
    severity_list = results_df.severity.astype(str).str.lower().to_list()
    
    count_high = severity_list.count("high")
    count_moderate = severity_list.count("moderate")
    count_low = severity_list.count("low")

    metadata["High counts"] = count_high
    metadata["Moderate counts "] = count_moderate
    metadata["Low counts "] = count_low
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    # 9. Visualization
    # ---------------------------------------------------------
    try:
        raw_vis_path = input_dir / "visual_raw.tif"
        if raw_vis_path.exists():
            raw_vis = rioxarray.open_rasterio(raw_vis_path).squeeze()
            bg_img = np.moveaxis(raw_vis.values, 0, -1)
            
            # Normalize
            vmin, vmax = np.nanpercentile(bg_img, [2, 98])
            bg_img = np.clip((bg_img - vmin) / (vmax - vmin), 0, 1)

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(bg_img)
            
            # Predictions (Dots)
            severity_colors_pred = {
                'low': 'green', 'moderate': 'orange', 'high': 'red',
                '1': 'green', '2': 'orange', '3': 'red',
                1: 'green', 2: 'orange', 3: 'red'
            }
            results_df['severity'] = results_df['severity'].astype(str)
            colors_pred = results_df['severity'].map(lambda x: severity_colors_pred.get(x.lower(), 'gray'))
            
            ax.scatter(results_df['pixel_col'], results_df['pixel_row'], 
                       c=colors_pred, s=20, alpha=0.9, edgecolors='black', linewidth=0.5, label='Prediction')
            
            # Reference Points (X)
            if all_reference_points:
                ref_lats = [float(p['lat']) for p in all_reference_points if 'lat' in p]
                ref_lons = [float(p['lon']) for p in all_reference_points if 'lon' in p]
                
                # Simple color logic for reference points (can be expanded)
                ref_colors = ['red'] * len(ref_lats)

                transformer_inv = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                ox, oy = transformer_inv.transform(ref_lons, ref_lats)
                orows, ocols = rowcol(transform, ox, oy)
                
                ax.scatter(ocols, orows, c=ref_colors, s=150, marker='x', linewidths=3, label='Reference')

            # Legend & Layout
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Low', markersize=8),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Moderate', markersize=8),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='High', markersize=8),
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            ax.set_title(f"CyFi Predictions: {date_str} (Cloud < {CLOUD_THRESHOLD*100}%)")
            ax.set_axis_off()
            
            final_img_path = input_dir / "cyfi_prediction_map.png"
            plt.savefig(final_img_path, bbox_inches='tight', dpi=150)
            
            if print_images:
                plt.show()
            plt.close(fig)
            
    except Exception as e:
        print(f"Visualization failed: {e}")

    # Cleanup
    shutil.rmtree(temp_cache)
    
    # 10. Rename Folder with Counts
    try:
        new_folder_name = f"{input_dir.name}_H{count_high}_M{count_moderate}_L{count_low}"
        # Avoid double-renaming if script is re-run
        if not input_dir.name.endswith(f"_L{count_low}"):
            new_folder_path = input_dir.parent / new_folder_name
            input_dir.rename(new_folder_path)
            return (True, new_folder_path)
        return (True, input_dir)
        
    except Exception as e:
        print(f"Could not rename folder: {e}")
        return (True, str(input_dir))


# =============================================================================
# 4. EXCEL REPORTING (THREAD-SAFE)
# =============================================================================

def update_excel_report(new_folder_path, excel_path="Master_Report.xlsx", uids_tracker_path="uids_processed.xlsx"):
    """
    Updates the Master Excel report and the Processed UIDs tracker.
    Uses File Locks to safely allow multiple scripts to run at once.
    """
    new_folder_path = Path(new_folder_path)
    
    # --- A. Update Master Report ---
    report_lock = SimpleFileLock(excel_path, timeout=120)
    
    with report_lock:
        try:
            if os.path.exists(excel_path):
                master_df = pd.read_excel(excel_path)
            else:
                master_df = pd.DataFrame()

            folder_name = new_folder_path.name
            
            metadata_path = new_folder_path / "metadata.json"
            counts = {"High": 0, "Moderate": 0, "Low": 0}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                    counts["High"] = meta.get("High counts", 0)
                    counts["Moderate"] = meta.get("Moderate counts ", 0)
                    counts["Low"] = meta.get("Low counts ", 0)
            
            new_row = {
                "source_path": str(new_folder_path),
                "folder_name": folder_name,
                "high_count": counts["High"],
                "mod_count": counts["Moderate"],
                "low_count": counts["Low"],
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pred_visual": "Image"
            }
            
            new_df = pd.DataFrame([new_row])
            master_df = pd.concat([master_df, new_df], ignore_index=True)
            
            # Write with Images
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                master_df.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                
                image_col_name = "pred_visual"
                filename = "cyfi_prediction_map.png"
                
                if image_col_name in master_df.columns:
                    col_idx = master_df.columns.get_loc(image_col_name)
                    
                    for index, row in master_df.iterrows():
                        excel_row_idx = index + 1
                        row_path_str = row.get('source_path')
                        if not row_path_str or pd.isna(row_path_str): continue
                        
                        img_path = Path(row_path_str) / filename
                        if img_path.exists():
                            worksheet.set_row_pixels(excel_row_idx, 150)
                            img_data, w, h = compress_image_to_bytes(img_path)
                            if img_data:
                                worksheet.embed_image(excel_row_idx, col_idx, filename, {
                                    'image_data': img_data, 
                                    'object_position': 1
                                })
                                worksheet.set_column_pixels(col_idx, col_idx, w + 10)
        
        except Exception as e:
            print(f"Error updating Master Report: {e}")

    # --- B. Update UID Tracker ---
    tracker_lock = SimpleFileLock(uids_tracker_path, timeout=60)
    
    with tracker_lock:
        try:
            if os.path.exists(uids_tracker_path):
                tracker_df = pd.read_excel(uids_tracker_path)
            else:
                tracker_df = pd.DataFrame(columns=['uid'])
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                    points = meta.get("points_data", [])
                    current_uids = [str(p.get('uid')) for p in points if 'uid' in p]
                    
                    if current_uids:
                        new_uids_df = pd.DataFrame({'uid': current_uids})
                        tracker_df = pd.concat([tracker_df, new_uids_df], ignore_index=True)
                        tracker_df.drop_duplicates(subset=['uid'], inplace=True)
                        tracker_df.to_excel(uids_tracker_path, index=False)
        except Exception as e:
            print(f"Error updating UID tracker: {e}")
            
            