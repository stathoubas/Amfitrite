# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:49:32 2026

@author: K. Pikounis
"""

import pandas as pd
import planetary_computer as pc
from pystac_client import Client
import datetime
import random
import ast
import time
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# Initialize STAC client
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def get_random_safe_dates(existing_dates_str, num_dates=8):
    """
    Parses the original dates list and generates 'num_dates' random dates 
    between 2016 and today that do NOT fall within +/- 60 days of any existing date.
    """
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

def run_fast_imputation_pipeline(input_csv, output_csv):
    print(f"Loading input dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Identify the ID column (assuming it's the first column)
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
    
    print(f"\nStarting FAST imputation for {total_rows} rows...")
    print("=" * 60)
    
    for index, row in df.iterrows():
        row_id = row[id_col]
        lat = row[lat_col]
        lon = row[lon_col]
        dates_str = str(row.get(date_list_col, "[]"))
        
        # 1. Generate 8 Safe Candidate Dates
        candidate_dates = get_random_safe_dates(dates_str, num_dates=8)
        
        # Using a dictionary to prevent duplicate satellite items if search windows overlap
        found_items_dict = {}
        
        # 2. Search STAC for each Candidate Date (+/- 15 Days)
        for c_date in candidate_dates:
            start_search = c_date - datetime.timedelta(days=15)
            end_search = c_date + datetime.timedelta(days=15)
            date_range = f"{start_search.strftime('%Y-%m-%d')}/{end_search.strftime('%Y-%m-%d')}"
            
            bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]
            
            try:
                # Fast metadata search
                search = catalog.search(
                    collections=["sentinel-2-l2a"], 
                    bbox=bbox, 
                    datetime=date_range
                )
                items = list(search.items())
            except Exception:
                items = []
                
            # 3. Collect pure STAC metadata (EXTREMELY FAST)
            for item in items:
                cc = item.properties.get("eo:cloud_cover", 100.0)
                item_date = item.datetime.strftime('%Y-%m-%d')
                
                found_items_dict[item.id] = {
                    "sat_item": item.id,
                    "date": item_date,
                    "global_cloud_cover": cc
                }
                    
        # 4. Implement the 3-Tier Priority System
        valid_items_found = list(found_items_dict.values())
        
        if valid_items_found:
            # Group 1: 0.1 < CC < 25 (Ideal tiles)
            group_1 = [i for i in valid_items_found if 0.1 < i["global_cloud_cover"] < 25]
            group_1.sort(key=lambda x: x["global_cloud_cover"]) # Ascending (Lowest clouds first)
            
            # Group 2: CC <= 0.1 (High risk of nodata edge, sort descending toward 0)
            group_2 = [i for i in valid_items_found if i["global_cloud_cover"] <= 0.1]
            group_2.sort(key=lambda x: x["global_cloud_cover"], reverse=True) # Descending (e.g., 0.1, 0.05, 0.0)
            
            # Group 3: CC >= 25 (Cloudy fallback)
            group_3 = [i for i in valid_items_found if i["global_cloud_cover"] >= 25]
            group_3.sort(key=lambda x: x["global_cloud_cover"]) # Ascending (Lowest clouds first)
            
            # Combine in priority order
            prioritized_list = group_1 + group_2 + group_3
            
            # Take the absolute best 5 items from the prioritized list
            top_5_items = prioritized_list[:5]
            
            # 5. Explode into new rows
            for best_item in top_5_items:
                new_row = {
                    id_col: row_id,
                    "center_lat": lat,
                    "center_lon": lon,
                    "sat_item": best_item["sat_item"],
                    "date": best_item["date"],
                    "global_cloud_cover": best_item["global_cloud_cover"],
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
    
    run_fast_imputation_pipeline(INPUT_CSV, OUTPUT_CSV)