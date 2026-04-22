# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:44:50 2026

@author: K. Pikounis
"""

import pandas as pd
import numpy as np
import planetary_computer as pc
from pystac_client import Client
import rioxarray
from pyproj import Transformer, CRS
import datetime
import random
import ast
import time
import os
import logging
import warnings

# Suppress rioxarray/pyproj warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize STAC client
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def get_random_safe_dates(existing_dates_str, num_dates=8):
    """
    Parses the original dates list and generates 'num_dates' random dates 
    between 2016 and today that do NOT fall within +/- 60 days of any existing date.
    """
    # Parse the string representation of the list into actual datetime objects
    try:
        raw_dates = ast.literal_eval(existing_dates_str)
        existing_dates = [pd.to_datetime(d) for d in raw_dates]
    except Exception:
        existing_dates = []

    safe_dates = []
    start_date = datetime.datetime(2016, 1, 1)
    end_date = datetime.datetime.now()
    
    start_ts = start_date.timestamp()
    end_ts = end_date.timestamp()
    
    attempts = 0
    while len(safe_dates) < num_dates and attempts < 1000:
        attempts += 1
        rand_ts = random.uniform(start_ts, end_ts)
        candidate = datetime.datetime.fromtimestamp(rand_ts)
        
        # Check against forbidden zones (+/- 60 days from existing dates)
        is_safe = True
        for ed in existing_dates:
            if abs((candidate - ed).days) <= 60:
                is_safe = False
                break
                
        # Make sure our new candidate isn't too close to another candidate we just picked
        if is_safe:
            for sd in safe_dates:
                if abs((candidate - sd).days) <= 15:
                    is_safe = False
                    break
                    
        if is_safe:
            safe_dates.append(candidate)
            
    return safe_dates

def check_tile_quality(item, lat, lon):
    """
    Clips the SCL band to exactly 2560x2560m (256px) around the coordinate.
    Returns (water_pixels, local_cloud_cover_percentage) or (-1, -1) if invalid.
    """
    try:
        target_crs = CRS.from_string(item.properties["proj:code"])
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        center_x, center_y = transformer.transform(lon, lat)

        half_side = 1280 # 1280m = 128 pixels * 10m/pixel (256 total width)
        minx, maxx = center_x - half_side, center_x + half_side
        miny, maxy = center_y - half_side, center_y + half_side

        scl_href = pc.sign(item.assets["SCL"].href)
        # Open and clip efficiently using rioxarray
        da_scl = rioxarray.open_rasterio(scl_href)
        da_clip = da_scl.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=target_crs)
        
        # SCL 6 = Water
        water_pixels = np.sum(da_clip.values == 6)
        
        # SCL 3=Shadow, 8=Med Cloud, 9=High Cloud, 10=Cirrus
        cloud_pixels = np.sum(np.isin(da_clip.values, [3, 8, 9, 10]))
        local_cloud_cover = (cloud_pixels / 65536) * 100.0
        
        return water_pixels, local_cloud_cover
        
    except Exception:
        # Fails if box goes off the edge of the image, etc.
        return -1, -1

def run_imputation_pipeline(input_csv, output_csv):
    print(f"Loading input dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Identify the ID column (assuming it's literally the first column as requested)
    id_col = df.columns[0]
    print(f"Using '{id_col}' as the ID column.")
    
    # Make sure we have the necessary coordinate columns
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    date_list_col = next((c for c in df.columns if 'date' in c.lower() and 'list' in c.lower()), 'dates_list')
    
    if not lat_col or not lon_col:
        print("Error: Could not find latitude/longitude columns.")
        return

    exploded_results = []
    
    total_rows = len(df)
    start_time = time.time()
    processed_count = 0
    
    print(f"\nStarting imputation for {total_rows} rows...")
    print("=" * 60)
    
    for index, row in df.iterrows():
        row_id = row[id_col]
        lat = row[lat_col]
        lon = row[lon_col]
        dates_str = str(row.get(date_list_col, "[]"))
        
        # 1. Generate 8 Safe Candidate Dates
        candidate_dates = get_random_safe_dates(dates_str, num_dates=8)
        
        valid_items_found = []
        
        # 2. Search STAC for each Candidate Date (+/- 15 Days)
        for c_date in candidate_dates:
            start_search = c_date - datetime.timedelta(days=15)
            end_search = c_date + datetime.timedelta(days=15)
            date_range = f"{start_search.strftime('%Y-%m-%d')}/{end_search.strftime('%Y-%m-%d')}"
            
            bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]
            
            try:
                search = catalog.search(
                    collections=["sentinel-2-l2a"], 
                    bbox=bbox, 
                    datetime=date_range, 
                    query={"eo:cloud_cover": {"lt": 50}} # Initial loose filter to save time
                )
                items = list(search.items())
            except Exception as e:
                items = []
                
            # 3. Dynamic SCL Checking
            for item in items:
                water_px, local_clouds = check_tile_quality(item, lat, lon)
                
                # Rule: Must have > 10,000 water pixels
                if water_px >= 10000:
                    item_date = item.datetime.strftime('%Y-%m-%d')
                    valid_items_found.append({
                        "sat_item": item.id,
                        "date": item_date, # Final explicit date column added for you
                        "local_cloud_cover": round(local_clouds, 2),
                        "water_pixels": int(water_px)
                    })
                    
        # 4. Sort strictly by Lowest Local Cloud Cover and keep Top 5
        if valid_items_found:
            # Sort by local cloud cover ascending
            valid_items_found.sort(key=lambda x: x["local_cloud_cover"])
            top_5_items = valid_items_found[:5]
            
            # 5. Explode into new rows
            for best_item in top_5_items:
                new_row = {
                    id_col: row_id,
                    "center_lat": lat,
                    "center_lon": lon,
                    "sat_item": best_item["sat_item"],
                    "date": best_item["date"],
                    "local_cloud_cover": best_item["local_cloud_cover"],
                    "water_pixels": best_item["water_pixels"],
                    "original_dates_list": dates_str
                }
                exploded_results.append(new_row)
                
        # --- Tracking Print Statement ---
        processed_count += 1
        if processed_count % 5 == 0 or processed_count == total_rows:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count
            est_remaining = avg_time * (total_rows - processed_count)
            print(f"Finished {processed_count}/{total_rows} entries in {elapsed:.1f}s | Est. Remaining: {est_remaining:.1f}s")

    # 6. Save Exploded Output
    final_df = pd.DataFrame(exploded_results)
    
    if not final_df.empty:
        final_df.to_csv(output_csv, index=False)
        print("\n" + "=" * 60)
        print(f"PIPELINE COMPLETE! Generated {len(final_df)} imputed satellite items.")
        print(f"Saved to: {output_csv}")
    else:
        print("\n[!] No valid satellite items were found for any coordinates.")


if __name__ == "__main__":
    # UPDATE THESE PATHS TO YOUR CSV FILES
    INPUT_CSV = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\tiles\merged_2_HABs_to_be_processed.csv"
    OUTPUT_CSV = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\tiles\merged_2_HABs_satelite_items.csv"
    
    run_imputation_pipeline(INPUT_CSV, OUTPUT_CSV)