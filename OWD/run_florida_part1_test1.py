# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:15:50 2026

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
import rioxarray
from pyproj import Transformer, CRS
import numpy as np
from datetime import timedelta

# Setup STAC Client
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def setup_logger(output_root, base_name):
    """Sets up a text logger for detailed debugging."""
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    log_file = os.path.join(output_root, f"prescreen_log_{base_name}.log")
    
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return log_file

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance in meters between two points (or a point and a pandas series)."""
    R = 6371000  # Radius of Earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def check_scl_water(item, lat, lon):
    """Downloads ONLY the SCL band for a 7680m x 7680m bounding box to check for water pixels."""
    try:
        target_crs = CRS.from_string(item.properties["proj:code"])
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        center_x, center_y = transformer.transform(lon, lat)

        # 7680m box (768 pixels * 10m)
        half_side = 3840
        minx, maxx = center_x - half_side, center_x + half_side
        miny, maxy = center_y - half_side, center_y + half_side

        scl_href = pc.sign(item.assets["SCL"].href)
        # Open SCL directly from cloud without downloading whole image
        da_scl = rioxarray.open_rasterio(scl_href)
        da_clip = da_scl.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=target_crs)
        
        # SCL Value 6 is Water
        water_pixels = np.sum(da_clip.values == 6)
        return int(water_pixels) > 0
    except Exception as e:
        logging.warning(f"SCL Check failed for {item.id}: {e}")
        return False

def search_stac(lat, lon, target_date, y_days):
    """Searches PC for Sentinel-2 items within +- Y days, < 50% clouds."""
    # Approximate 10km buffer for STAC geometry intersection
    buffer_deg = 0.1 
    bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
    
    start_date = pd.to_datetime(target_date) - timedelta(days=y_days)
    end_date = pd.to_datetime(target_date) + timedelta(days=y_days)
    date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    
    search = catalog.search(
        collections=["sentinel-2-l2a"], 
        bbox=bbox, 
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 50}}
    )
    items = list(search.items())
    # Sort from clearest to cloudiest
    items.sort(key=lambda x: x.properties["eo:cloud_cover"])
    return items

def run_prescreener(csv_path, output_dir, x_days, y_days):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    setup_logger(output_dir, base_name)
    
    logging.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path) if csv_path.endswith('.csv') else pd.read_excel(csv_path)
    
    # Initialize new columns if they don't exist
    if 'sat_item' not in df.columns: df['sat_item'] = ""
    if 'removed_by_idx' not in df.columns: df['removed_by_idx'] = ""
    
    df['Event_Date'] = pd.to_datetime(df['eventDate'], errors='coerce') # Ensure correct date col name
    
    # 1. Get HAB targets, sorted by highest organismQuantity
    hab_mask = (df['is_HAB'].str.strip().str.upper() == 'YES')
    df_habs = df[hab_mask].sort_values(by='organismQuantity', ascending=False)
    
    habs_processed = 0
    
    logging.info("Starting Phase 1: Greedy Algorithm...")
    for idx, row in df_habs.iterrows():
        # Check if it was suppressed in a previous loop
        if pd.notna(df.at[idx, 'removed_by_idx']) and str(df.at[idx, 'removed_by_idx']).strip() != "":
            continue
            
        current_id = str(row['id_x'])
        lat, lon = row['decimalLatitude'], row['decimalLongitude']
        date = row['Event_Date']
        
        logging.info(f"\nEvaluating HAB {current_id} | Qty: {row['organismQuantity']} | Date: {date.date()}")
        
        # 2. Search for satellite image (+- Y days)
        items = search_stac(lat, lon, date, y_days)
        best_hab_item = None
        
        for item in items:
            logging.info(f"  Checking SCL water for {item.id} ({item.properties['eo:cloud_cover']:.1f}% clouds)...")
            if check_scl_water(item, lat, lon):
                best_hab_item = item.id
                break
                
        if not best_hab_item:
            logging.info("  No valid water image found. Skipping.")
            continue
            
        logging.info(f"  [FOUND] HAB Item: {best_hab_item}")
        df.at[idx, 'sat_item'] = best_hab_item
        
        # 3. Suppress all other HABs within 7680m and +- X days
        dist_mask = haversine_distance(lat, lon, df['decimalLatitude'], df['decimalLongitude']) <= 7680
        time_mask = abs(df['Event_Date'] - date).dt.days <= x_days
        
        # Apply suppression to OTHER HABs
        suppress_hab_mask = dist_mask & time_mask & (df['is_HAB'].str.strip().str.upper() == 'YES') & (df.index != idx)
        df.loc[suppress_hab_mask, 'removed_by_idx'] = current_id
        logging.info(f"  Suppressed {suppress_hab_mask.sum()} neighboring HAB points.")
        
        # 4. Find the Paired Non-HAB candidate
        # Must be in same 7680m, is_HAB == No, removed_by_idx is empty. Sort by lowest quantity.
        non_hab_mask = dist_mask & (df['is_HAB'].str.strip().str.upper() == 'NO') & \
                       (df['removed_by_idx'].isna() | (df['removed_by_idx'] == ""))
                       
        df_non_habs = df[non_hab_mask].sort_values(by='organismQuantity', ascending=True)
        
        best_non_hab_item = None
        for nh_idx, nh_row in df_non_habs.iterrows():
            logging.info(f"  Trying Non-HAB pair {nh_row['id_x']} | Qty: {nh_row['organismQuantity']}")
            nh_items = search_stac(lat, lon, nh_row['Event_Date'], y_days)
            
            for nh_item in nh_items:
                if check_scl_water(nh_item, lat, lon):
                    best_non_hab_item = nh_item.id
                    break
                    
            if best_non_hab_item:
                logging.info(f"  [FOUND] Paired Clean Item: {best_non_hab_item}")
                df.at[nh_idx, 'sat_item'] = best_non_hab_item
                df.at[nh_idx, 'removed_by_idx'] = current_id # Link it to the HAB that claimed it
                
                # Suppress other Non-HABs around this clean event
                nh_time_mask = abs(df['Event_Date'] - nh_row['Event_Date']).dt.days <= x_days
                suppress_nh_mask = dist_mask & nh_time_mask & (df['is_HAB'].str.strip().str.upper() == 'NO') & (df.index != nh_idx)
                df.loc[suppress_nh_mask, 'removed_by_idx'] = nh_row['id_x']
                break
                
        if not best_non_hab_item:
            logging.info("  Could not find a valid clean satellite pair for this HAB.")

        # 5. Periodic Saving (Every 100 HABs evaluated)
        habs_processed += 1
        if habs_processed % 100 == 0:
            out_file = os.path.join(output_dir, f"{base_name}_prescreened_part_{habs_processed}.csv")
            df.to_csv(out_file, index=False)
            logging.info(f"--- Checkpoint Saved: {out_file} ---")

    # Final Save
    final_file = os.path.join(output_dir, f"{base_name}_prescreened_FINAL.csv")
    df.to_csv(final_file, index=False)
    logging.info(f"Pipeline Complete! Final output saved to: {final_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greedy HAB API Pre-Screener")
    parser.add_argument("--input", type=str, required=True, help="Input CSV/Excel file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    # Get user inputs for spatial/temporal rules
    #try:
    #    x_days = int(input("Enter 'X' (Days for temporal suppression window, e.g., 14): "))
    #    y_days = int(input("Enter 'Y' (Days for Satellite STAC search window, e.g., 5): "))
    #except ValueError:
    #    print("Invalid input. Defaulting to X=14, Y=5.")
    #    x_days = 14
    #    y_days = 5
     
    x_days = 14
    y_days = 5
    run_prescreener(args.input, args.output, x_days, y_days)