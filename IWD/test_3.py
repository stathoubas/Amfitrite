# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 11:53:15 2025

@author: K. Pikounis
"""
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from pystac_client import Client
import planetary_computer as pc
import rioxarray
from utils_v4 import update_excel_report, predict_using_cyfi_pipeline, extract_and_save_tile, select_item, get_bounding_box


def log_problem_uids(df, reason, file_path):
    if df.empty: return
    log_data = pd.DataFrame({
        'uid': df['uid'],
        'case': df.iloc[0]['case'] if 'case' in df.columns else 'N/A',
        'reason': reason
    })
    if os.path.exists(file_path):
        log_data.to_csv(file_path, mode='a', header=False, index=False)
    else:
        log_data.to_csv(file_path, mode='w', header=True, index=False)
    print(f"Logged {len(df)} UIDs as '{reason}'")

def merge_time_windows(date_list, buffer_days=15):
    """
    Takes a list of date strings, creates windows of +/- buffer_days,
    and merges overlapping windows.
    Returns a list of tuples: [(start_date, end_date), ...]
    """
    if not date_list:
        return []

    # 1. Create individual intervals
    intervals = []
    for d in date_list:
        dt = pd.to_datetime(d)
        start = dt - timedelta(days=buffer_days)
        end = dt + timedelta(days=buffer_days)
        intervals.append((start, end))
    
    # 2. Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    # 3. Merge overlaps
    merged = []
    if not intervals: return merged
    
    curr_start, curr_end = intervals[0]
    
    for next_start, next_end in intervals[1:]:
        if next_start <= curr_end: 
            # Overlap exists, extend the current end
            curr_end = max(curr_end, next_end)
        else:
            # No overlap, push current and start new
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
            
    merged.append((curr_start, curr_end))
    return merged

def search_optimized_windows(lat, lon, date_list):
    """
    Searches Planetary Computer using merged time windows to avoid 
    fetching unnecessary data gaps.
    """
    # 1. Get efficient time ranges
    time_ranges = merge_time_windows(date_list)
    
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    bbox = get_bounding_box(lat, lon, 3000)
    
    all_items = []
    seen_ids = set() 
    
    # 2. Search for each specific range
    for start, end in time_ranges:
        datetime_format = "%Y-%m-%d"
        date_range_str = f"{start.strftime(datetime_format)}/{end.strftime(datetime_format)}"
        
        print(f"   > Searching interval: {date_range_str}")
        search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=date_range_str)
        
        for item in search.get_all_items():
            if item.id not in seen_ids:
                all_items.append(item)
                seen_ids.add(item.id)
    
    if not all_items: return pd.DataFrame()

    # 3. Convert to DataFrame
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
    
    # 4. Calculate Cloud Cover (Batch)
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


# --- CONFIGURATION ---
root_folder = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD"
sat_test_folder = os.path.join(root_folder, "sat data test", "v1")
uids_processed_path = os.path.join(sat_test_folder, "uids_processed.xlsx")
problematic_uids_path = os.path.join(sat_test_folder, "problematic_uids.csv")

os.makedirs(sat_test_folder, exist_ok=True)

# --- MAIN EXECUTION ---
df = pd.read_excel(os.path.join(root_folder, "tick tick bloom data", "clustered_only_square_2560m.xlsx"))
after2017 = df[(df.date >= "2017-01-01")]
after2017_near_water = after2017[after2017["distance_to_water_m"] <= 10]
low_sev = after2017_near_water[(after2017_near_water.severity <= 1)]
low_sev.sort_values(by = "abun", inplace = True)

low_sev_cases = list(dict.fromkeys(low_sev.case.to_list()))

for iii, sel_case in enumerate(low_sev_cases):
    print(f"----------> PROCESSING CASE {iii}: {sel_case}")
    
    case_df = after2017_near_water[after2017_near_water.case == sel_case].copy()
    
    # Filter processed UIDs
    if os.path.exists(uids_processed_path):
        try:
            processed_df = pd.read_excel(uids_processed_path)
            if not processed_df.empty:
                processed_uids = processed_df['uid'].astype(str).unique()
                case_df = case_df[~case_df['uid'].astype(str).isin(processed_uids)]
        except: pass

    if case_df.empty:
        print("----------> Skipping case (all UIDs processed).")
        continue

    # 1. Create sub-clusters based on date
    case_df['date_str'] = pd.to_datetime(case_df['date']).dt.strftime('%Y-%m-%d')
    unique_dates = sorted(case_df['date_str'].unique())
    print(f"----------> Unique dates in case: {unique_dates}")
    
    # 2. Search PC with OPTIMIZED windows
    all_items_df = search_optimized_windows(case_df['lat'].mean(), case_df['lon'].mean(), unique_dates)
    
    if all_items_df.empty:
        log_problem_uids(case_df, "no suitable item", problematic_uids_path)
        continue

    # 3. Assign items to nearest dates
    date_candidates = {}
    
    for target_date in unique_dates:
        target_dt = pd.to_datetime(target_date)
        
        # Filter items within +- 15 days of THIS target date
        # Note: We search based on efficient windows, but now we map the results back to specific dates
        all_items_df['diff'] = (pd.to_datetime(all_items_df['datetime']) - target_dt).abs().dt.days
        
        # Keep items within 15 days
        candidates = all_items_df[all_items_df['diff'] <= 15].copy()
        date_candidates[target_date] = candidates

    #Select Optimal Items & Handle Merging
    selected_item_map = {} 
    
    for date, candidates in date_candidates.items():
        if candidates.empty:
            selected_item_map[date] = None
        else:
            # Need signed difference for select_item preference logic
            candidates['date_difference'] = (pd.to_datetime(candidates['datetime']) - pd.to_datetime(date)).dt.days
            
            best_item, found = select_item(candidates)
            if found:
                selected_item_map[date] = best_item
            else:
                selected_item_map[date] = None

    # Group dates by their selected Item ID to perform merging
    processing_groups = {}
    
    for date, item in selected_item_map.items():
        if item is None:
            sub_df = case_df[case_df['date_str'] == date]
            log_problem_uids(sub_df, "no suitable item", problematic_uids_path)
            continue
            
        item_id = item.item_obj.id 
        
        if item_id not in processing_groups:
            processing_groups[item_id] = {
                'item': item,
                'dates': []
            }
        processing_groups[item_id]['dates'].append(date)

    # 6. Execute Processing for each Group
    for item_id, group in processing_groups.items():
        dates_in_group = group['dates']
        item_row = group['item']
        
        print(f"----------> Processing Item {item_id} for dates: {dates_in_group}")
        
        # a) Define points_df (merged rows for all dates in this group)
        points_df = case_df[case_df['date_str'].isin(dates_in_group)].copy()
        
        # b) Run steps
        abun = str(int(max(points_df["abun"].to_list())))
        final_date = item_row.datetime
        pixel_size = 365
        
        out_folder = os.path.join(sat_test_folder, f"case{sel_case}_abun{abun}_{pixel_size}_date{final_date}")
        
        # Extract
        extract_and_save_tile(
            item_row.item_obj, 
            points_df, 
            initial_pixel_size=pixel_size,
            save_data=True, 
            output_path=out_folder, 
            print_images=False
        )
        
        # CyFi
        folder_ok, new_path = predict_using_cyfi_pipeline(out_folder, final_date)
        
        if folder_ok:
            summary_file_path_name = os.path.join(sat_test_folder, "Master_Report.xlsx")
            processed_uids_path_name = uids_processed_path
            
            update_excel_report(new_path, excel_path=summary_file_path_name, uids_tracker_path=processed_uids_path_name)
            print("----------> Report Updated.")
        else:
            print("----------> CyFi Failed.")
            log_problem_uids(points_df, "cyfi failed", problematic_uids_path)

print("----------> Batch processing complete.")