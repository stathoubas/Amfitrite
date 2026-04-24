# -*- coding: utf-8 -*-
"""
Created for AMFITRITE 
Pipeline: Dynamic 90-Day Forbidden Interval Satellite Imputation
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
import warnings

# Suppress rioxarray/pyproj warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize STAC client
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def check_tile_quality(item, lat, lon):
    """
    Clips the SCL band to exactly 2560x2560m around the coordinate.
    Returns (water_pixels, local_cloud_cover_percentage) or (-1, -1) if invalid.
    """
    try:
        # Reverted back to your original, correct logic: proj:code
        proj_code = item.properties.get("proj:code")
        if not proj_code:
            return -1, -1
            
        target_crs = CRS.from_string(proj_code)
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        center_x, center_y = transformer.transform(lon, lat)

        half_side = 1280 # 1280m radius (2560m total width)
        minx, maxx = center_x - half_side, center_x + half_side
        miny, maxy = center_y - half_side, center_y + half_side

        scl_href = pc.sign(item.assets["SCL"].href)
        
        # Open and clip efficiently using rioxarray
        da_scl = rioxarray.open_rasterio(scl_href)
        da_clip = da_scl.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=target_crs)
        
        # Calculate native 20m pixels
        water_pixels_20m = np.sum(da_clip.values == 6)
        cloud_pixels_20m = np.sum(np.isin(da_clip.values, [3, 8, 9, 10]))
        
        # Upscale 20m pixels to 10m CyFi pixels (1 native pixel -> 4 CyFi pixels)
        # Native array is 16,384 pixels. CyFi array is 65,536 pixels.
        water_pixels_10m = water_pixels_20m * 4
        cloud_pixels_10m = cloud_pixels_20m * 4
        
        local_cloud_cover = (cloud_pixels_10m / 65536.0) * 100.0
        
        return water_pixels_10m, local_cloud_cover
        
    except Exception as e:
        print(f"  [Debug] Tile check failed for {item.id}: {e}")
        return -1, -1


def is_date_safe(candidate_date, forbidden_intervals):
    """Checks if a date falls inside ANY of the forbidden [start, end] intervals."""
    for start, end in forbidden_intervals:
        if start <= candidate_date <= end:
            return False
    return True

def run_imputation_pipeline(input_csv, output_csv):
    print(f"Loading input dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    
    id_col = df.columns[0]
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
    
    print(f"\nStarting dynamic interval imputation for {total_rows} rows...")
    print("=" * 60)
    
    start_date_limit = datetime.datetime(2016, 1, 1)
    end_date_limit = datetime.datetime.now()
    start_ts = start_date_limit.timestamp()
    end_ts = end_date_limit.timestamp()
    
    for index, row in df.iterrows():
        row_id = row[id_col]
        lat = row[lat_col]
        lon = row[lon_col]
        dates_str = str(row.get(date_list_col, "[]"))
        
        # 1. Initialize Forbidden Intervals with provided dates
        forbidden_intervals = []
        try:
            raw_dates = ast.literal_eval(dates_str)
            for d in raw_dates:
                d_obj = pd.to_datetime(d)
                forbidden_intervals.append(
                    (d_obj - datetime.timedelta(days=45), d_obj + datetime.timedelta(days=45))
                )
        except Exception:
            pass
            
        kept_items = []
        api_searches = 0
        
        # 2. Main Discovery Loop (Max 10 API searches or 8 kept images)
        while api_searches < 10 and len(kept_items) < 8:
            
            # 3. Memory Guessing: Find a safe random date
            candidate_date = None
            for _ in range(1000): # Allow up to 1000 memory guesses per search
                rand_ts = random.uniform(start_ts, end_ts)
                temp_date = datetime.datetime.fromtimestamp(rand_ts)
                if is_date_safe(temp_date, forbidden_intervals):
                    candidate_date = temp_date
                    break
            
            # SAFETY STOP CONDITION
            if not candidate_date:
                print(f"  [!] Timeline saturated for ID {row_id} after 1000 memory guesses. Halting search.")
                break 
                
            # 4. Perform STAC Search (+/- 15 Days around candidate)
            api_searches += 1
            start_search = candidate_date - datetime.timedelta(days=15)
            end_search = candidate_date + datetime.timedelta(days=15)
            date_range = f"{start_search.strftime('%Y-%m-%dT00:00:00Z')}/{end_search.strftime('%Y-%m-%dT23:59:59Z')}"
            bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]
            
            try:
                search = catalog.search(
                    collections=["sentinel-2-l2a"], 
                    bbox=bbox, 
                    datetime=date_range, 
                    query={"eo:cloud_cover": {"lt": 50}} # Loose global filter
                )
                items = list(search.items())
            except Exception:
                items = []
                
            # Sort items by lowest global cloud cover to evaluate best images first
            items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100.0))
            
            found_valid_image = False
            
            for item in items:
                water_px, local_clouds = check_tile_quality(item, lat, lon)
                
                # Rule: Must have > 32,000 water pixels AND <= 15% local cloud cover
                if water_px >= 32000 and local_clouds <= 15.0:
                    item_date_str = item.datetime.strftime('%Y-%m-%d')
                    item_date_obj = pd.to_datetime(item_date_str)
                    
                    kept_items.append({
                        "sat_item": item.id,
                        "date": item_date_str,
                        "local_cloud_cover": round(local_clouds, 2),
                        "water_pixels": int(water_px)
                    })
                    
                    # Add new +/- 45 days (90 day window) to forbidden zones
                    forbidden_intervals.append(
                        (item_date_obj - datetime.timedelta(days=45), item_date_obj + datetime.timedelta(days=45))
                    )
                    found_valid_image = True
                    break # We found the best image in this 30-day window, move to next random search
                    
            # If the entire 30-day window was a bust (too cloudy or no water)
            if not found_valid_image:
                forbidden_intervals.append(
                    (start_search, end_search)
                )
                
        # 5. Cap and Sort Results
        if kept_items:
            # Sort the discovered safe images by absolute best local cloud cover
            kept_items.sort(key=lambda x: x["local_cloud_cover"])
            top_5_items = kept_items[:5]
            
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
        print(f"PIPELINE COMPLETE! Generated {len(final_df)} perfectly spaced satellite items.")
        print(f"Saved to: {output_csv}")
    else:
        print("\n[!] No valid satellite items were found for any coordinates.")


if __name__ == "__main__":
    INPUT_CSV = r"/vol2/Amfitrite/OWD/processed_results/merged_2_HABs_to_be_processed.csv"
    OUTPUT_CSV = r"/vol2/Amfitrite/OWD/processed_results/merged_2_HABs_satelite_items.csv"
    
    run_imputation_pipeline(INPUT_CSV, OUTPUT_CSV)