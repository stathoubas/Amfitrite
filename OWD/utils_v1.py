# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:36:39 2026

@author: K. Pikounis

based on utils_v5 of IWD folder
"""

import json
from pathlib import Path
from datetime import timedelta
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import tempfile
import shutil
from tqdm import tqdm

import rasterio
import rioxarray
from rasterio.transform import rowcol
from rioxarray.exceptions import NoDataInBounds
import geopandas as gpd

from pyproj import CRS, Transformer, Geod

# CyFi imports
from cyfi.pipeline import CyFiPipeline
from cyfi.config import FeaturesConfig
from cyfi.data.features import generate_all_features
from cyfi.cli import DEFAULT_MODEL_PATH

# images in excel
import io

import planetary_computer as pc
from pystac_client import Client

import pandas as pd
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as L
import timm

import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from PIL import Image, ImageDraw, ImageFont



def get_bounding_box(latitude, longitude, meter_buffer=50000):
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box [min_lon, min_lat, max_lon, max_lat] around the point.
    """
    g = Geod(ellps='WGS84')
    # Forward calculation: (lon, lat, back_az)
    lon_west, _, _ = g.fwd(longitude, latitude, 270, meter_buffer)
    _, lat_south, _ = g.fwd(longitude, latitude, 180, meter_buffer)
    lon_east, _, _ = g.fwd(longitude, latitude, 90, meter_buffer)
    _, lat_north, _ = g.fwd(longitude, latitude, 0, meter_buffer)
    return [lon_west, lat_south, lon_east, lat_north]

def get_date_range(date, time_buffer_days=15):
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will include the sample date
    and time_buffer_days days prior

    Returns a string"""
    datetime_format = "%Y-%m-%d"
    range_start = pd.to_datetime(date) - timedelta(days=time_buffer_days)
    range_end = pd.to_datetime(date) + timedelta(days=time_buffer_days)
    date_range = f"{range_start.strftime(datetime_format)}/{range_end.strftime(datetime_format)}"

    return date_range

def search_with_retry(search_obj, max_retries=5):
    """
    Executes search.item_collection() with retries to handle API timeouts.
    """
    for attempt in range(max_retries):
        try:
            # item_collection() is the modern replacement for get_all_items()
            return search_obj.item_collection()
        except Exception as e:
            error_msg = str(e)
            # Check for timeout or server availability errors
            if "maximum allowed time" in error_msg or "504" in error_msg or "503" in error_msg:
                wait_time = (2 ** attempt) + (random.random() * 2)
                print(f"   >>> API Timeout/Error. Retrying in {wait_time:.2f}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                # If it's a logic error (not network/timeout), raise immediately
                raise e
                
    print(f"   >>> Failed after {max_retries} retries. Skipping this interval.")
    return []

def find_satelite_images(p_lat, p_lon, sel_date, meter_buffer = 3840):

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    
    bbox = get_bounding_box(p_lat, p_lon, meter_buffer)
    print(f"Search BBox: {bbox}")
    date_range = get_date_range(sel_date)

    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=date_range
    )

    items = search_with_retry(search)
    
    item_details = pd.DataFrame(
        [
            {
                "datetime": item.datetime.strftime("%Y-%m-%d"),
                "platform": item.properties["platform"],
                "min_long": item.bbox[0],
                "max_long": item.bbox[2],
                "min_lat": item.bbox[1],
                "max_lat": item.bbox[3],
                "bbox": item.bbox,
                "item_obj": item,
            }
            for item in items
        ]
    )

    # check which rows actually contain the sample location
    item_details["contains_sample_point"] = (
        (item_details.min_lat < p_lat)
        & (item_details.max_lat > p_lat)
        & (item_details.min_long < p_lon)
        & (item_details.max_long > p_lon)
    )

    print(
        f"Filtering from {len(item_details)} returned to {item_details.contains_sample_point.sum()} items that contain the sample location"
    )
    
    item_details = item_details[item_details["contains_sample_point"]]
    item_details[["datetime", "platform", "contains_sample_point", "bbox"]].sort_values(
        by="datetime"
    )

    item_details["per_clouds"] = -1.0
    # the commented code below finds the cloud coverage of the whole image
    '''
    for it, row in item_details.iterrows():
        ar = rioxarray.open_rasterio(pc.sign(row.item_obj.assets["SCL"].href)).to_numpy()
        mask = (ar == 3) | ((ar >= 7) & (ar <= 10))
        cloud_per = round(100 * np.sum(mask) / ar.size, 2)
        item_details.at[it, "per_clouds"] = cloud_per
    '''
    
    # the code below finds the cloud coverage of the box of interest
    for it, row in item_details.iterrows():
        try:
            # 1. Lazy load the SCL asset URL
            scl_href = pc.sign(row.item_obj.assets["SCL"].href)
            
            # 2. Open and Clip to the bounding box (Lazy loading)
            # This ensures we don't download the whole tile, just the area of interest
            ds = rioxarray.open_rasterio(scl_href)
            ds_clip = ds.rio.clip_box(
                minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326"
            )
            
            # 3. Load values into memory (now it's a small 2D array)
            ar = ds_clip.values.squeeze()
            
            # 4. Calculate Cloud Percentage
            # We filter for: 
            #   3: Cloud Shadows
            #   7: Unclassified
            #   8: Cloud Medium Probability
            #   9: Cloud High Probability
            #   10: Cirrus
            # Create a boolean mask
            cloud_mask = (ar == 3) | ((ar >= 7) & (ar <= 10))
            
            # Calculate percentage: (Count of True / Total Pixels) * 100
            cloud_per = round(100 * np.sum(cloud_mask) / ar.size, 2)
            
            item_details.at[it, "per_clouds"] = cloud_per            
        except Exception as e:
            print(f"Error calculating clouds for item {it}: {e}")
    
    item_details['date_difference'] = (pd.to_datetime(item_details['datetime']) - pd.to_datetime(sel_date)).dt.days
    
    return item_details

#def select_item(item_details):
#    
#    if item_details.empty:
#        return False, None, None
#    
#    low_clouds = item_details[item_details.per_clouds < 7.5]
#    if len(low_clouds) == 0:
#        return False, None
#    
#    # Sort by lowest clouds
#    sel_items = low_clouds.sort_values(by="per_clouds", ascending=True)
#    best_row = sel_items.iloc[0]
#    
#    return True, best_row.name, best_row["item_obj"]

def select_item(item_details):
    
    if item_details.empty:
        return False, None, None
    
    # Sort by lowest clouds
    sel_items = item_details.sort_values(by="per_clouds", ascending=True)
    best_row = sel_items.iloc[0]
    
    return True, best_row.name, best_row["item_obj"]

def extract_and_save_tile(
    item,
    lat,
    lon,
    case_id,
    initial_pixel_size=100,
    save_data=False,
    output_path=None, 
    print_images=True):
    
    print("--- Starting Tile Extraction and Processing (9 Squares) ---")
    
    # 1. Setup CRS and Center
    target_crs = CRS.from_string(item.properties["proj:code"])
    center_lat = lat
    center_lon = lon
    print(f"Center Point (Lat/Lon): ({center_lat:.4f}, {center_lon:.4f})")
    
    # 2. Calculate Center in UTM 
    transformer_latlon_to_utm = Transformer.from_crs(CRS.from_string("EPSG:4326"), target_crs, always_xy=True)
    center_x, center_y = transformer_latlon_to_utm.transform(center_lon, center_lat)

    final_pixel_side = int(initial_pixel_size)
    L = final_pixel_side * 10  # Tile size in meters
    half_side = L / 2
    
    # Define the 9 centers (0=Center, 1=Upper, 2=Upper-Left, 3=Left, etc.)
    offsets = {
        0: (0, 0),    # center
        1: (0, 1),    # upper
        2: (-1, 1),   # upper-left
        3: (-1, 0),   # left
        4: (-1, -1),  # bottom-left
        5: (0, -1),   # bottom
        6: (1, -1),   # bottom-right
        7: (1, 0),    # right
        8: (1, 1)     # upper-right
    }

    try:
        b04_href = pc.sign(item.assets["B04"].href)
        b04_ds_full = rioxarray.open_rasterio(b04_href)
    except Exception as e:
         print(f"Error opening B04: {e}")
         return False

    band_gsd_map = {
        "B02": 10, "B03": 10, "B04": 10, "B08": 10, "visual": 10, "AOT": 10, "WVP": 60,
        "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "SCL": 20, "B11": 20, "B12": 20,
        "B01": 60, "B09": 60
    }
    
    # Pre-open all assets to save HTTP request overhead
    opened_assets = {}
    for asset_key in band_gsd_map.keys():
        if asset_key == "B04": continue
        href = pc.sign(item.assets[asset_key].href)
        opened_assets[asset_key] = rioxarray.open_rasterio(href)

    base_output_dir = Path(output_path) if output_path else None
    success_any = False

    for tile_idx, (dx, dy) in offsets.items():
        print(f"\n--- Processing Tile {tile_idx} ---")
        
        tile_center_x = center_x + dx * L
        tile_center_y = center_y + dy * L
        
        min_x_final = tile_center_x - half_side
        max_x_final = tile_center_x + half_side
        min_y_final = tile_center_y - half_side
        max_y_final = tile_center_y + half_side
        
        clip_bbox_utm = (min_x_final, min_y_final, max_x_final, max_y_final)
        clipped_data = {}
        
        try:
            b04_clip = b04_ds_full.rio.clip_box(
                minx=clip_bbox_utm[0], miny=clip_bbox_utm[1], 
                maxx=clip_bbox_utm[2], maxy=clip_bbox_utm[3],
                crs=target_crs
            )
            b04_clip = b04_clip.isel(y=slice(0, final_pixel_side), x=slice(0, final_pixel_side))
            clipped_data["B04"] = b04_clip.squeeze() 
            
            for asset_key, ds in opened_assets.items():
                ds_clip = ds.rio.clip_box(
                    minx=clip_bbox_utm[0], miny=clip_bbox_utm[1], 
                    maxx=clip_bbox_utm[2], maxy=clip_bbox_utm[3],
                    crs=target_crs
                )
                resampling_method = rasterio.enums.Resampling.nearest if asset_key == "SCL" else rasterio.enums.Resampling.bilinear
                ds_aligned = ds_clip.rio.reproject_match(b04_clip, resampling=resampling_method)
                
                if asset_key != "visual":
                    clipped_data[asset_key] = ds_aligned.squeeze()
                else:
                     clipped_data[asset_key] = ds_aligned
                     
        except NoDataInBounds:
            print(f"Error: Bounding box outside bounds for tile {tile_idx}. Skipping.")
            continue
        except Exception as e:
            print(f"An error occurred during clipping tile {tile_idx}: {e}")
            continue
            
        # --- CALCULATE CLOUD & WATER STATS FROM SCL ---
        scl_vals = clipped_data["SCL"].values
        cloud_mask = (scl_vals == 3) | ((scl_vals >= 7) & (scl_vals <= 10))
        water_mask = (scl_vals == 6)
        
        calculated_per_clouds = round((np.sum(cloud_mask) / scl_vals.size) * 100, 2)
        calculated_water_pixels = int(np.sum(water_mask))
        print(f"Tile {tile_idx} Stats - Clouds: {calculated_per_clouds}%, Water Pixels: {calculated_water_pixels}")

        if save_data and base_output_dir:
            output_dir = base_output_dir / f"tile_{tile_idx}"
            output_dir.mkdir(parents=True, exist_ok=True)
            for key, da in clipped_data.items():
                da.rio.to_raster(output_dir / f"{key}_raw.tif")

            # Calculate Indices
            SCALE_FACTOR = 10000.0 
            b04_ref = (clipped_data["B04"] / SCALE_FACTOR).astype(np.float32)
            b05_ref = (clipped_data["B05"] / SCALE_FACTOR).astype(np.float32)
            b07_ref = (clipped_data["B07"] / SCALE_FACTOR).astype(np.float32)
            b08_ref = (clipped_data["B08"] / SCALE_FACTOR).astype(np.float32)
            
            sum_b8_b4 = b08_ref + b04_ref
            ndvi = (b08_ref - b04_ref) / sum_b8_b4.where(sum_b8_b4 != 0, np.nan) 
            sum_b7_b5 = b07_ref + b05_ref
            ndci = (b07_ref - b05_ref) / sum_b7_b5.where(sum_b7_b5 != 0, np.nan) 
            
            vis_data = clipped_data.copy()
            vis_data["NDVI"] = ndvi
            vis_data["NDCI"] = ndci
            
            # Save Previews
            colors = ['white', 'white', 'black', 'black', 'green', 'saddlebrown', 'lightblue', 'grey', 'grey', 'grey', 'grey']
            cmap_scl = ListedColormap(colors)
            bounds = np.arange(12)
            norm_scl = BoundaryNorm(bounds, cmap_scl.N)

            def save_plot_with_points(da, key, cmap=None, norm=None, vmin=None, vmax=None, rgb=False):
                fig, ax = plt.subplots(figsize=(10, 10))
                extent_utm = [da.x.min(), da.x.max(), da.y.min(), da.y.max()]
                if rgb:
                    arr = da.transpose('y', 'x', 'band').values
                    vmin, vmax = np.nanpercentile(arr, [2, 98])
                    arr_scaled = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
                    ax.imshow(arr_scaled, extent=extent_utm, origin='upper')
                elif key == "SCL":
                    ax.imshow(da.values, cmap=cmap, norm=norm, extent=extent_utm, origin='upper')
                else:
                    robust = True if vmin is None else False
                    arr_2d = da.values.squeeze()
                    if robust:
                        vmin, vmax = np.nanpercentile(arr_2d, [2, 98])
                    ax.imshow(arr_2d, cmap=cmap, extent=extent_utm, origin='upper', vmin=vmin, vmax=vmax)
                
                ax.set_axis_off() 
                plt.savefig(output_dir / f"{key}_preview.png", bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            for key in vis_data.keys():
                da = vis_data[key]
                if key == "visual": save_plot_with_points(da, key, rgb=True)
                elif key == "SCL": save_plot_with_points(da, key, cmap=cmap_scl, norm=norm_scl)
                elif key == "NDVI": save_plot_with_points(da, key, cmap="RdYlGn", vmin=-1, vmax=1)
                elif key == "NDCI": save_plot_with_points(da, key, cmap="jet", vmin=-1, vmax=1)
                else: save_plot_with_points(da, key, cmap="gray")

            # Convert tile center to lat/lon for metadata
            transformer_utm_to_latlon = Transformer.from_crs(target_crs, CRS.from_string("EPSG:4326"), always_xy=True)
            t_lon, t_lat = transformer_utm_to_latlon.transform(tile_center_x, tile_center_y)

            metadata = {
                "item_id": item.id,
                "per_clouds": calculated_per_clouds,
                "water_pixels": calculated_water_pixels,
                "num_pixels":scl_vals.size,
                "uid": f"{case_id}_tile_{tile_idx}", 
                "abun": "N/A", 
                "tile_size_10m_pixels": final_pixel_side,
                "date": item.properties["datetime"].split('T')[0],
                "center_lat": t_lat,
                "center_lon": t_lon,
                "points_data": [{
                    'case': str(case_id.split("_")[0]),
                    'lat': t_lat,
                    'lon': t_lon,
                    'date': str(case_id.split("_")[1])
                }]
            }
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
                
        success_any = True

    return success_any

def predict_using_cyfi_pipeline(base_folder_path, 
                                date_str, 
                                metadata_filename="metadata.json",
                                print_images=False):
    
    base_dir = Path(base_folder_path).resolve()
    print(f"--- Starting CyFi Pipeline Integration for 9 Tiles in: {base_dir} ---")
    
    features_config = FeaturesConfig()
    CLOUD_THRESHOLD = 0.075 
    features_config.max_cloud_percent = CLOUD_THRESHOLD

    WINDOW_METERS = features_config.image_feature_meter_window 
    PIXEL_SIZE = 10
    WINDOW_PIXELS = WINDOW_METERS // PIXEL_SIZE 
    RADIUS_PIXELS = WINDOW_PIXELS // 2 
    required_bands = features_config.use_sentinel_bands 
    
    processed_folders = []

    for i in range(9):
        input_dir = base_dir / f"tile_{i}"
        if not input_dir.exists():
            continue
            
        print(f"\n--- Running CyFi for {input_dir.name} ---")
        
        data_store = {}
        try:
            scl_da = rioxarray.open_rasterio(input_dir / "SCL_raw.tif").squeeze()
            data_store["SCL"] = scl_da.values
            height, width = scl_da.shape
            transform = scl_da.rio.transform()
            crs = scl_da.rio.crs
        except FileNotFoundError:
            print(f"Error: SCL_raw.tif not found in {input_dir}. Skipping.")
            continue

        metadata_path = input_dir / metadata_filename
        metadata = {}
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {metadata_filename} not found.")

        for band in required_bands:
            if band == "SCL": continue
            path = input_dir / f"{band}_raw.tif"
            if path.exists():
                data_store[band] = rioxarray.open_rasterio(path).squeeze().values
            else:
                data_store[band] = np.full((height, width), np.nan, dtype=np.float32)

        print("Generating 100m Lattice Grid...")
        GRID_STEP = 10 
        
        rows = np.arange(0, height, GRID_STEP)
        cols = np.arange(0, width, GRID_STEP)
        grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
        grid_rows = grid_rows.flatten()
        grid_cols = grid_cols.flatten()
        
        valid_mask = (grid_rows < height) & (grid_cols < width)
        grid_rows = grid_rows[valid_mask]
        grid_cols = grid_cols[valid_mask]
        
        water_mask = (data_store["SCL"][grid_rows, grid_cols] == 6)
        target_rows = grid_rows[water_mask]
        target_cols = grid_cols[water_mask]
        
        initial_samples = len(target_rows)
        print(f"Identified {initial_samples} potential water points.")
        if initial_samples == 0: 
            processed_folders.append((False, str(input_dir)))
            continue

        temp_cache = Path(tempfile.mkdtemp(prefix=f"cyfi_lattice_tile{i}_"))
        cache_subdir = temp_cache / f"sentinel_{WINDOW_METERS}"
        fake_item_id = "LATTICE_ITEM" 

        valid_sample_ids = []
        valid_lats = []
        valid_lons = []
        valid_rows = []
        valid_cols = []

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        print("Checking Cloud Cover & Slicing Data...")
        skipped_clouds = 0
        
        for idx, (r, c) in enumerate(tqdm(zip(target_rows, target_cols), total=initial_samples)):
            s_id = f"sample_{idx}"
            r_min = max(0, r - RADIUS_PIXELS)
            r_max = min(height, r + RADIUS_PIXELS)
            c_min = max(0, c - RADIUS_PIXELS)
            c_max = min(width, c + RADIUS_PIXELS)
            
            if (r_max - r_min) == 0 or (c_max - c_min) == 0: continue

            scl_window = data_store["SCL"][r_min:r_max, c_min:c_max]
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

        print(f" - Valid Points to Predict: {len(valid_sample_ids)}")

        if len(valid_sample_ids) == 0:
            print("No valid points remained after cloud filtering.")
            shutil.rmtree(temp_cache)
            processed_folders.append((False, str(input_dir)))
            continue

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

        print("Running CyFi Feature Generation...")
        try:
            _, features_df = generate_all_features(
                samples=samples_df,
                satellite_meta=satellite_meta_df,
                config=features_config,
                cache_dir=temp_cache
            )
        except SystemExit as e:
            print(f"   > CyFi triggered SystemExit (likely no valid data): {e}")
            shutil.rmtree(temp_cache)
            processed_folders.append((False, str(input_dir)))
            continue
        except Exception as e:
            print(f"Feature generation failed: {e}")
            shutil.rmtree(temp_cache)
            processed_folders.append((False, str(input_dir)))
            continue

        print("Running Prediction...")
        pipeline = CyFiPipeline.from_disk(DEFAULT_MODEL_PATH)
        pipeline.predict_features = features_df
        pipeline.predict_samples = samples_df
        pipeline._predict_model()
        
        results_df = pipeline.output_df.reset_index()
        results_df["pixel_row"] = valid_rows
        results_df["pixel_col"] = valid_cols
        
        csv_path = input_dir / "cyfi_lattice_predictions.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")

        severity_list = results_df.severity.astype(str).str.lower().to_list()
        count_high = severity_list.count("high")
        count_moderate = severity_list.count("moderate")
        count_low = severity_list.count("low")

        metadata["High counts"] = count_high
        metadata["Moderate counts "] = count_moderate
        metadata["Low counts "] = count_low
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print("Generating Visualization...")
        try:
            raw_vis_path = input_dir / "visual_raw.tif"
            if raw_vis_path.exists():
                raw_vis = rioxarray.open_rasterio(raw_vis_path).squeeze()
                bg_img = np.moveaxis(raw_vis.values, 0, -1)
                vmin, vmax = np.nanpercentile(bg_img, [2, 98])
                bg_img = np.clip((bg_img - vmin) / (vmax - vmin), 0, 1)

                fig, ax = plt.subplots(figsize=(12, 12))
                ax.imshow(bg_img)
                
                severity_colors_pred = {
                    'low': 'green', 'moderate': 'orange', 'high': 'red',
                    '1': 'green', '2': 'orange', '3': 'red',
                    1: 'green', 2: 'orange', 3: 'red'
                }
                results_df['severity'] = results_df['severity'].astype(str)
                colors_pred = results_df['severity'].map(lambda x: severity_colors_pred.get(x.lower(), 'gray'))
                
                ax.scatter(results_df['pixel_col'], results_df['pixel_row'], 
                           c=colors_pred, s=20, alpha=0.9, edgecolors='black', linewidth=0.5, label='Prediction')
                
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Low', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Moderate', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='High', markersize=8)
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                ax.set_title(f"CyFi Predictions: {date_str} (Cloud < {CLOUD_THRESHOLD*100}%)")
                ax.set_axis_off()
                
                final_img_path = input_dir / "cyfi_prediction_map.png"
                plt.savefig(final_img_path, bbox_inches='tight', dpi=150)
                if print_images: plt.show()
                plt.close(fig)
        except Exception as e:
            print(f"Visualization failed: {e}")

        shutil.rmtree(temp_cache)
        time.sleep(1)
        
        try:
            new_folder_name = f"{input_dir.name}_H{count_high}_M{count_moderate}_L{count_low}"
            if not input_dir.name.endswith(f"_H{count_high}_M{count_moderate}_L{count_low}"):
                new_folder_path = input_dir.parent / new_folder_name
                input_dir.rename(new_folder_path)
                print(f"Successfully renamed folder to: {new_folder_path}")
                processed_folders.append((True, new_folder_path))
            else:
                processed_folders.append((True, input_dir))
        except Exception as e:
            print(f"Could not rename folder: {e}")
            processed_folders.append((True, str(input_dir)))

    return processed_folders
    

def update_uids_and_summary(folder_path, uids_file, summary_file):
    
    folder = Path(folder_path)
    
    # 1. Load Metadata
    try:
        with open(folder / "metadata.json") as f:
            meta = json.load(f)
    except FileNotFoundError:
        print(f"Error: metadata.json not found in {folder}")
        return

    # --- 2. UID TRACKER (Modified for CSV & Top-level UID) ---
    current_uid = meta.get("uid", "N/A")

    # UID tracker
    if Path(uids_file).exists():
        df_u = pd.read_excel(uids_file)
    else:
        df_u = pd.DataFrame(columns=["uid"])

    # Add new UID and save
    new_row = pd.DataFrame({"uid": [current_uid]})
    df_u = pd.concat([df_u, new_row], ignore_index=True).drop_duplicates(subset=['uid'])
    df_u.to_excel(uids_file, index=False)

    # --- 3. SUMMARY (Excel) ---
    # Use meta.get for 'case' instead of parsing filename (safer)
    # Note: 'points_data' inside meta is now a list with one dict, 
    # but we can just grab top-level data or the first item.
    
    points_data_list = meta.get("points_data", [])
    first_point = points_data_list[0] if points_data_list else {}

    row = {
        "uid": current_uid,
        "case": meta.get("case", first_point.get("case", "N/A")), 
        "item_id": meta.get("item_id", "N/A"),
        "date": meta.get("date", "N/A"),
        "lat": meta.get("center_lat", "N/A"),
        "lon": meta.get("center_lon", "N/A"),
        "abun": meta.get("abun", "N/A"),
        "abun_list": "N/A", # No list in this workflow
        "per_clouds": meta.get("per_clouds", "N/A"), 
        "water_pixels": meta.get("water_pixels", "N/A"),
        "all_picesl": meta.get("num_pixels", "N/A"),
        "high_pred": meta.get("High counts", 0),
        "mod_pred": meta.get("Moderate counts ", 0),
        "low_pred": meta.get("Low counts ", 0),
        "pred_visual": "", 
        "source_path": str(folder)
    }

    if Path(summary_file).exists():
        df_s = pd.read_excel(summary_file)
    else:
        df_s = pd.DataFrame()

    df_s = pd.concat([df_s, pd.DataFrame([row])], ignore_index=True)
    df_s.to_excel(summary_file, index=False)
    
    print(f"Summary updated for {current_uid}")
    
def load_tensor_from_folder(folder_path, use_mask=False, img_size=256):
    """Loads bands from a single folder, applies mask if needed, and returns a model-ready tensor."""
    band_names = [
        "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", "B05_raw.tif", "B06_raw.tif",
        "B07_raw.tif", "B08_raw.tif", "B8A_raw.tif", "B11_raw.tif", "B12_raw.tif"
    ]
    
    band_data = []
    for b_name in band_names:
        p = os.path.join(folder_path, b_name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required band file: {p}")
            
        with rasterio.open(p) as src:
            band_data.append(src.read(1).astype(np.float32))
            
    bands_stack = np.stack(band_data, axis=0)
    
    # Apply Mask (If requested)
    if use_mask:
        scl_path = os.path.join(folder_path, "SCL_raw.tif")
        if os.path.exists(scl_path):
            with rasterio.open(scl_path) as src:
                scl = src.read(1)
            # SCL 6 is water
            mask = (scl == 6).astype(np.float32)
            bands_stack = bands_stack * mask
        else:
            print("  [Warning] SCL_raw.tif not found in folder. Proceeding without mask.")
            
    # Convert to Tensor and add Batch Dimension -> Shape: (1, 10, H, W)
    tensor = torch.from_numpy(bands_stack).unsqueeze(0)
    
    # Resize to expected dimensions
    tensor = torch.nn.functional.interpolate(
        tensor, size=(img_size, img_size), 
        mode='bilinear', align_corners=False
    )
    
    # Normalize
    tensor = tensor / 10000.0
    
    return tensor


class HABLightningModel(L.LightningModule):
    def __init__(self, arch_name='resnet18', num_classes=2, in_chans=10):
        super().__init__()
        self.model = self._build_model(arch_name, num_classes, in_chans)

    def _build_model(self, arch, num_classes, in_chans):
        if arch == 'resnet18':
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        elif arch == 'convnextv2_base':
            return timm.create_model('convnextv2_base', pretrained=False, num_classes=num_classes, in_chans=in_chans)
        elif arch == 'rdnet_base':
            return timm.create_model('rdnet_base', pretrained=False, num_classes=num_classes, in_chans=in_chans)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, x):
        return self.model(x)


def run_single_folder(base_folder_path, model_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # Save a master results file in the root
    output_txt_path = os.path.join(base_folder_path, "cnn_results_all_tiles.txt")
    
    with open(output_txt_path, "w") as f:
        header = f"Analyzing Base Folder: {base_folder_path}\n" + "-" * 50
        print(header)
        f.write(header + "\n")
        
        scenarios = [
            ("res18_scl",      "resnet18",        True,  256),
            ("res18_no_scl",   "resnet18",        False, 256),
            ("convnext_scl",   "convnextv2_base", True,  256),
            ("rdnet_no_scl",   "rdnet_base",      False, 256)
        ]
        
        # Dynamically find all subdirectories starting with "tile_" to account for CyFi's folder renaming
        subdirs = [d for d in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, d)) and d.startswith("tile_")]
        subdirs.sort(key=lambda x: int(x.split('_')[1]))
        
        for tile_dir_name in subdirs:
            tile_folder = os.path.join(base_folder_path, tile_dir_name)
            
            tile_header = f"\n--- Results for {tile_dir_name} ---"
            print(tile_header)
            f.write(tile_header + "\n")
            
            for scenario_name, arch, use_mask, img_size in scenarios:
                try:
                    input_tensor = load_tensor_from_folder(tile_folder, use_mask=use_mask, img_size=img_size)
                    input_tensor = input_tensor.to(device)
                    
                    model_wrapper = HABLightningModel(arch_name=arch)
                    ckpt_path = model_paths.get(scenario_name)
                    
                    if not ckpt_path or not os.path.exists(ckpt_path):
                        error_msg = f"{scenario_name:<15} | ERROR: Weights not found at {ckpt_path}"
                        print(error_msg)
                        f.write(error_msg + "\n")
                        continue
                        
                    if ckpt_path.endswith('.ckpt'):
                        checkpoint = torch.load(ckpt_path, map_location='cpu')
                        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
                        model_wrapper.model.load_state_dict(state_dict, strict=False)
                    else:
                        state_dict = torch.load(ckpt_path, map_location='cpu')
                        model_wrapper.model.load_state_dict(state_dict, strict=False)
                        
                    model_wrapper.to(device)
                    model_wrapper.eval()
                    
                    with torch.no_grad():
                        outputs = model_wrapper(input_tensor)
                        _, preds = torch.max(outputs, 1)
                        pred_class = preds.item()
                        
                    str_pred = "YES" if pred_class == 1 else "NO"
                    result_line = f"Model: {scenario_name:<15} | HAB Detected: {str_pred}"
                    
                    print(result_line)
                    f.write(result_line + "\n")
                    
                except Exception as e:
                    err_line = f"Model: {scenario_name:<15} | ERROR: {str(e)}"
                    print(err_line)
                    f.write(err_line + "\n")
                    
    print(f"\nResults have been successfully saved to: {output_txt_path}")
    

def create_collage(base_dir):
    """
    Creates a 3x3 collage of the 9 tiles with black borders.
    Draws the tile number in white in the top-left corner of each tile.
    Saves the output as 'colage.png' in the base directory.
    """
    print("--- Generating 3x3 Collage ---")
    
    # Configuration for the collage
    tile_size = 500  # Resize all images to 500x500 pixels for perfect alignment
    border = 10      # 10 pixel black line between tiles
    total_size = (tile_size * 3) + (border * 4)
    
    # Map the tile index to a 3x3 grid (row, col)
    idx_to_pos = {
        2: (0, 0), 1: (0, 1), 8: (0, 2),
        3: (1, 0), 0: (1, 1), 7: (1, 2),
        4: (2, 0), 5: (2, 1), 6: (2, 2)
    }
    
    # Try to load a large font if supported by the Pillow version, otherwise use default
    try:
        font = ImageFont.load_default(size=50)
    except TypeError:
        font = ImageFont.load_default()
    
    # Create a completely black canvas
    collage = Image.new("RGB", (total_size, total_size), "black")
    draw = ImageDraw.Draw(collage)
    
    # Loop through the 9 expected tiles
    for i in range(9):
        row, col = idx_to_pos[i]
        
        folders = list(base_dir.glob(f"tile_{i}*"))
        if not folders or not folders[0].is_dir():
            continue  
            
        tile_dir = folders[0]
        
        # 1. Try to find the CyFi Prediction Map
        img_path = tile_dir / "cyfi_prediction_map.png"
        
        # 2. If it doesn't exist, fallback to the Visual Preview
        if not img_path.exists():
            img_path = tile_dir / "visual_preview.png"
            
        # If an image was found, process, paste, and label it
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((tile_size, tile_size))
                
                # Calculate coordinates
                x = border + col * (tile_size + border)
                y = border + row * (tile_size + border)
                
                # Paste into the collage
                collage.paste(img, (x, y))
                
                # Draw the number with a black outline for visibility
                text = str(i)
                text_x, text_y = x + 15, y + 15
                
                # Draw black outline
                outline_color = "black"
                draw.text((text_x - 2, text_y - 2), text, font=font, fill=outline_color)
                draw.text((text_x + 2, text_y - 2), text, font=font, fill=outline_color)
                draw.text((text_x - 2, text_y + 2), text, font=font, fill=outline_color)
                draw.text((text_x + 2, text_y + 2), text, font=font, fill=outline_color)
                
                # Draw white text over it
                draw.text((text_x, text_y), text, font=font, fill="white")
                
            except Exception as e:
                print(f"  [Warning] Could not process image for tile_{i}: {e}")
                
    # Save the final collage
    out_path = base_dir / "colage.png"
    collage.save(out_path)
    print(f"Collage successfully saved to: {out_path}")


def folder_summary(base_folder_path, add_images_to_excel=True):
    """
    Summarizes CyFi and CNN results, calculates totals, creates a collage,
    and optionally embeds prediction map thumbnails into the Excel file.
    """
    base_dir = Path(base_folder_path)
    print(f"--- Generating Summary for {base_dir.name} ---")
    
    # 1. Generate Collage
    create_collage(base_dir)
    
    # 2. Parse CNN results
    cnn_results_file = base_dir / "cnn_results_all_tiles.txt"
    cnn_data = {}
    
    if cnn_results_file.exists():
        with open(cnn_results_file, 'r') as f:
            lines = f.readlines()
        
        current_tile = None
        for line in lines:
            line = line.strip()
            if line.startswith("--- Results for "):
                current_tile = line.replace("--- Results for ", "").replace(" ---", "")
                cnn_data[current_tile] = {}
            elif line.startswith("Model:") and current_tile is not None:
                parts = line.split("|")
                model_name = parts[0].replace("Model:", "").strip()
                prediction = parts[1].replace("HAB Detected:", "").strip()
                cnn_data[current_tile][model_name] = prediction
    else:
        print(f"  [Warning] CNN results file not found at {cnn_results_file}")

    # 3. Gather CyFi Stats
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d) and d.startswith("tile_")]
    
    try:
        subdirs.sort(key=lambda x: int(x.split('_')[1]))
    except Exception:
        subdirs.sort()
        
    summary_list = []
    
    for tile_dir_name in subdirs:
        tile_path = base_dir / tile_dir_name
        meta_path = tile_path / "metadata.json"
        
        h_count, m_count, l_count, water_pixels = 0, 0, 0, 0
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                h_count = meta.get("High counts", 0)
                m_count = meta.get("Moderate counts ", meta.get("Moderate counts", 0)) 
                l_count = meta.get("Low counts ", meta.get("Low counts", 0))
                water_pixels = meta.get("water_pixels", 0)
                all_pixels = meta.get("num_pixels", 0)
                
        total_cyfi_points = h_count + m_count + l_count
        
        row_data = {
            "Tile_Folder": tile_dir_name,
            "CyFi_Map": "", 
            "Total_pixels": all_pixels,
            "Total_CyFi_Points": total_cyfi_points,
            "SCL_Water_Pixels": water_pixels,
            "CyFi_High": h_count,
            "CyFi_Moderate": m_count,
            "CyFi_Low": l_count
        }
        
        tile_cnn = cnn_data.get(tile_dir_name, {})
        row_data["res18_scl"] = tile_cnn.get("res18_scl", "N/A")
        row_data["res18_no_scl"] = tile_cnn.get("res18_no_scl", "N/A")
        row_data["convnext_scl"] = tile_cnn.get("convnext_scl", "N/A")
        row_data["rdnet_no_scl"] = tile_cnn.get("rdnet_no_scl", "N/A")
        
        summary_list.append(row_data)
        
    # 4. Create DataFrame and inject images into Excel
    if summary_list:
        df = pd.DataFrame(summary_list)
        output_excel = base_dir / "folder_summary.xlsx"
        
        df.to_excel(output_excel, index=False)
        
        if add_images_to_excel:
            print("--- Adding Image Thumbnails to Excel ---")
            wb = openpyxl.load_workbook(output_excel)
            ws = wb.active
            
            ws.column_dimensions['B'].width = 18
            
            for idx, tile_dir_name in enumerate(df['Tile_Folder']):
                row_num = idx + 2  
                ws.row_dimensions[row_num].height = 80
                
                img_path = base_dir / tile_dir_name / "cyfi_prediction_map.png"
                
                if img_path.exists():
                    try:
                        img = OpenpyxlImage(str(img_path))
                        img.width = 100
                        img.height = 100
                        ws.add_image(img, f"B{row_num}")
                    except Exception as e:
                        print(f"  [Warning] Could not insert image for {tile_dir_name}: {e}")

            wb.save(output_excel)
            
        print(f"Summary successfully saved to: {output_excel}")
        #display(df.drop(columns=["CyFi_Map"]))
        return df
    else:
        print("No tile subfolders found to summarize.")
        return None
    
