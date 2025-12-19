# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 10:35:32 2025

@author: K. Pikounis

Parallel-Ready Script.
Usage: Run multiple instances of this script simultaneously.
Run for seveirty 4 and 5 excluding any case with severity 1
"""

import pandas as pd
import numpy as np
import os
import time
import random
from datetime import timedelta
from pystac_client import Client
import planetary_computer as pc
import rioxarray

# Import from the server-utils
from utils_v5_server import (
    update_uids_and_summary_locked, 
    predict_using_cyfi_pipeline, 
    extract_and_save_tile, 
    select_item, 
    get_bounding_box, 
    SimpleFileLock # Key for safe parallel logging
)

# --- Helper Functions ---

def log_problem_uids(df, reason, file_path):
    if df.empty: return
    
    # LOCK: Prevent log file corruption
    lock = SimpleFileLock(file_path, timeout=60)
    with lock:
        log_data = pd.DataFrame({
            'uid': df['uid'],
            'case': df.iloc[0]['case'] if 'case' in df.columns else 'N/A',
            'reason': reason
        })
        # Check if file exists/is empty to write headers
        header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        
        if header:
            log_data.to_csv(file_path, mode='w', header=True, index=False)
        else:
            log_data.to_csv(file_path, mode='a', header=False, index=False)
            
    print(f"Logged {len(df)} UIDs as '{reason}'")

def merge_time_windows(date_list, buffer_days=15):
    if not date_list: return []
    intervals = []
    for d in date_list:
        dt = pd.to_datetime(d)
        start = dt - timedelta(days=buffer_days)
        end = dt + timedelta(days=buffer_days)
        intervals.append((start, end))
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    if not intervals: return merged
    curr_start, curr_end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= curr_end: 
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    return merged

def search_with_retry(search_obj, max_retries=5):
    """Executes search.item_collection() with random jitter backoff."""
    for attempt in range(max_retries):
        try:
            return search_obj.item_collection()
        except Exception as e:
            error_msg = str(e)
            if "maximum allowed time" in error_msg or "504" in error_msg or "503" in error_msg:
                # Add random jitter (1-5s) to desynchronize parallel workers
                wait_time = (2 ** attempt) + (random.random() * 5)
                print(f"   >>> API Busy. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                raise e
    print(f"   >>> Failed after {max_retries} retries.")
    return []

def search_optimized_windows(lat, lon, date_list):
    time_ranges = merge_time_windows(date_list)
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    bbox = get_bounding_box(lat, lon, 3000)
    
    all_items = []
    seen_ids = set() 
    
    for start, end in time_ranges:
        datetime_format = "%Y-%m-%d"
        date_range_str = f"{start.strftime(datetime_format)}/{end.strftime(datetime_format)}"
        
        print(f"   > Searching: {date_range_str}")
        search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=date_range_str)
        
        items = search_with_retry(search)
        
        for item in items:
            if item.id not in seen_ids:
                all_items.append(item)
                seen_ids.add(item.id)
    
    if not all_items: return pd.DataFrame()

    item_details = pd.DataFrame([
        {
            "datetime": item.datetime.strftime("%Y-%m-%d"),
            "datetime_obj": item.datetime,
            "platform": item.properties["platform"],
            "item_obj": item,
            "bbox": item.bbox
        }
        for item in all_items
    ])
    
    # Calculate Cloud Cover Locally
    item_details["per_clouds"] = -1.0
    for it, row in item_details.iterrows():
        try:
            scl_href = pc.sign(row.item_obj.assets["SCL"].href)
            ds = rioxarray.open_rasterio(scl_href)
            ds_clip = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326")
            ar = ds_clip.values.squeeze()
            cloud_mask = (ar == 3) | ((ar >= 7) & (ar <= 10))
            cloud_per = round(100 * np.sum(cloud_mask) / ar.size, 2)
            item_details.at[it, "per_clouds"] = cloud_per            
        except:
            pass
            
    return item_details

# --- Configuration ---
root_folder = r"/vol/Amfitrite"
sat_test_folder = os.path.join(root_folder, "sat_data", "v4_3")
uids_processed_path = os.path.join(sat_test_folder, "uids_processed.xlsx")
problematic_uids_path = os.path.join(sat_test_folder, "problematic_uids_45.csv")

os.makedirs(sat_test_folder, exist_ok=True)

# 1. Load Data
df = pd.read_excel(os.path.join(root_folder, "clustered_only_square_2560m.xlsx"))
after2017 = df[(df.date >= "2017-01-01")]
after2017_near_water = after2017[after2017["distance_to_water_m"] <= 10]
sev_1 = after2017_near_water[(after2017_near_water.severity <= 1)].copy()
cases1 = list(dict.fromkeys(sev_1.case.to_list()))

sev_45 = after2017_near_water[(after2017_near_water.severity >= 4)].copy()
sev_45.sort_values(by = "abun")
cases45 = list(dict.fromkeys(sev_45.case.to_list()))
cases45_not_prev = [i for i in cases45 if i not in cases1]


print(f"----------> Starting Parallel Worker for cases with severity 4 and 5: {len(cases45_not_prev)}")

# 3. Processing Loop
for iii, sel_case in enumerate(cases45_not_prev):
    
    # Short random sleep to prevent all workers hitting the API at the exact same time
    time.sleep(random.random() * 3)

    case_df = after2017_near_water[after2017_near_water.case == sel_case].copy()
    
    # --- CHECK 1: Is it already processed? (Thread-Safe) ---
    lock = SimpleFileLock(uids_processed_path, timeout=60)
    with lock:
        if os.path.exists(uids_processed_path):
            try:
                processed_df = pd.read_excel(uids_processed_path)
                if not processed_df.empty:
                    processed_uids = processed_df['uid'].astype(str).unique()
                    case_df = case_df[~case_df['uid'].astype(str).isin(processed_uids)]
            except: pass

    # --- CHECK 2: Is it problematic/failed? (Thread-Safe) ---
    p_lock = SimpleFileLock(problematic_uids_path, timeout=60)
    with p_lock:
        if os.path.exists(problematic_uids_path) and os.path.getsize(problematic_uids_path) > 0:
            try:
                prob_df = pd.read_csv(problematic_uids_path)
                if not prob_df.empty and 'uid' in prob_df.columns:
                    prob_uids = prob_df['uid'].astype(str).unique()
                    case_df = case_df[~case_df['uid'].astype(str).isin(prob_uids)]
            except: pass

    if case_df.empty:
        print("----------> Skipping Case {sel_case} (all UIDs processed or problematic).")
        continue

    print(f"----------> Processing Case {sel_case} ({len(case_df)} UIDs remaining) {iii} from {len(cases45_not_prev)}")

    # 4. Search
    case_df['date_str'] = pd.to_datetime(case_df['date']).dt.strftime('%Y-%m-%d')
    unique_dates = sorted(case_df['date_str'].unique())
    
    all_items_df = search_optimized_windows(case_df['lat'].mean(), case_df['lon'].mean(), unique_dates)
    
    if all_items_df.empty:
        log_problem_uids(case_df, "no suitable item", problematic_uids_path)
        continue

    # 5. Grouping Logic (Standard)
    date_candidates = {}
    for target_date in unique_dates:
        target_dt = pd.to_datetime(target_date)
        all_items_df['diff'] = (pd.to_datetime(all_items_df['datetime']) - target_dt).abs().dt.days
        candidates = all_items_df[all_items_df['diff'] <= 15].copy()
        date_candidates[target_date] = candidates

    selected_item_map = {} 
    for date, candidates in date_candidates.items():
        if candidates.empty:
            selected_item_map[date] = None
        else:
            candidates['date_difference'] = (pd.to_datetime(candidates['datetime']) - pd.to_datetime(date)).dt.days
            best_item, found = select_item(candidates)
            selected_item_map[date] = best_item if found else None

    processing_groups = {}
    for date, item in selected_item_map.items():
        if item is None:
            sub_df = case_df[case_df['date_str'] == date]
            log_problem_uids(sub_df, "no suitable item", problematic_uids_path)
            continue
            
        item_id = item.item_obj.id 
        if item_id not in processing_groups:
            processing_groups[item_id] = {'item': item, 'dates': []}
        processing_groups[item_id]['dates'].append(date)

    # 6. Execution
    for item_id, group in processing_groups.items():
        dates_in_group = group['dates']
        item_row = group['item']
        
        points_df = case_df[case_df['date_str'].isin(dates_in_group)].copy()
        abun = str(int(max(points_df["abun"].to_list())))
        final_date = item_row.datetime
        pixel_size = 365
        
        out_folder = os.path.join(sat_test_folder, f"case{sel_case}_abun{abun}_{pixel_size}_date{final_date}")
        
        # Download Tile
        extract_success = extract_and_save_tile(
            item_row.item_obj, 
            points_df, 
            initial_pixel_size=pixel_size,
            save_data=True, 
            output_path=out_folder, 
            print_images=False
        )
        
        if not extract_success:
            print(f"----------> Extraction failed for item {item_id}.")
            log_problem_uids(points_df, "raster outside bounds", problematic_uids_path)
            continue
        
        # Run CyFi
        try:
            folder_ok, new_path = predict_using_cyfi_pipeline(out_folder, final_date)
            
            if folder_ok:
                summary_file_path_name = os.path.join(sat_test_folder, "Master_summary.xlsx")
                processed_tracker = uids_processed_path
                
                # This handles locking internally now
                update_uids_and_summary_locked(folder_path=new_path, uids_file=uids_processed_path, summary_file=summary_file_path_name, retries=5)
                print(f"----------> [SUCCESS] Case {sel_case} Updated.")
            else:
                print(f"----------> [FAIL] CyFi failed for {sel_case}.")
                log_problem_uids(points_df, "cyfi failed", problematic_uids_path)
        except:
            print(f"----------> [FAIL] CyFi failed for {sel_case}.")
            log_problem_uids(points_df, "cyfi failed", problematic_uids_path)

print("----------> Worker batch complete.")