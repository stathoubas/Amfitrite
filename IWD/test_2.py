# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 13:04:23 2025

@author: K. Pikounis
"""


import pandas as pd
import os
from utils_v2 import find_satelite_images, update_excel_report, predict_using_cyfi_pipeline, extract_and_save_tile, select_item

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


root_folder = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD"
sat_test_folder = os.path.join(root_folder, "sat data test")
uids_processed_path = os.path.join(sat_test_folder, "uids_processed.xlsx")
problematic_uids_path = os.path.join(sat_test_folder, "problematic_uids.csv")


df = pd.read_excel(os.path.join(root_folder, "tick tick bloom data", "clustered_data_15days_2560m.xlsx"))
after2017 = df[(df.date >= "2017-01-01")]
after2017_near_water = after2017[after2017["distance_to_water_m"] <= 10]
low_sev = after2017_near_water[(after2017_near_water.severity <= 1)]
low_sev.sort_values(by = "abun", inplace = True)
low_sev_cases = list(dict.fromkeys(low_sev.case.to_list()))

for iii, sel_case in enumerate(low_sev_cases[:100]):
    
    print("----------> HELLO ", iii, sel_case)
    
    #points_df = low_sev[low_sev.case == sel_case].copy()
    points_df = after2017_near_water[after2017_near_water.case == sel_case].copy()
    if 2 in points_df.severity.to_list() or 3 in points_df.severity.to_list() or 4 in points_df.severity.to_list() or 5 in points_df.severity.to_list():
        print("----------> Not pure in severities ", iii, sel_case)
        continue
    
    if os.path.exists(uids_processed_path):
        try:
            processed_df = pd.read_excel(uids_processed_path)
            if not processed_df.empty:
                # Convert to string to ensure matching types
                processed_uids = processed_df['uid'].astype(str).unique()
                
                # Keep only UIDs that are NOT in the processed list
                points_df = points_df[~points_df['uid'].astype(str).isin(processed_uids)]
        except Exception as e:
            print(f"Warning: Could not read processed UIDs file: {e}")

    # If no points left to process for this case, skip to next case
    if points_df.empty:
        print(f"Case {sel_case}: All UIDs already processed. Skipping.")
        continue

    abun = str(int(max(points_df["abun"].to_list())))
    dates = [i.strftime('%Y-%m-%d') for i in points_df["date"].value_counts().index.to_list()]
    found = False
    item = None
    
    for date in dates:
        print("----------> date is ", date)
        item_details = find_satelite_images(points_df["lat"].mean(), points_df["lon"].mean(), date)
        if item_details.empty:
            continue
            
        item, found = select_item(item_details)
        if found:
            break
            
    if found:
        final_date = item.datetime
        print("----------> Found  item!!!  with date = ", final_date)
        pixel_size = 365
        
        out_folder = os.path.join(sat_test_folder, f"case{sel_case}_abun{abun}_{pixel_size}_date{final_date}")
        
        # extract tile
        extract_and_save_tile(item.item_obj, points_df, after2017_near_water, initial_pixel_size=pixel_size,
            save_data=True, output_path=out_folder, print_images=False)
        
        print("----------> Finished extract and save")
        
        # run cyfi pipeline
        folder_ok, new_path = predict_using_cyfi_pipeline(out_folder, final_date)
        
        print("----------> Finished cyfi")
        
        if folder_ok:
            summary_file_path_name = os.path.join(sat_test_folder, "Master_Report.xlsx")
            processed_uids_path_name = uids_processed_path
            
            # updates the report and adds the UIDs to uids_processed.xlsx
            update_excel_report(new_path, excel_path=summary_file_path_name, uids_tracker_path=processed_uids_path_name)
            print("----------> Excel report updated")
        else:
            #Log "cyfi failed"
            print("----------> cyfi failed")
            log_problem_uids(points_df, "cyfi failed", problematic_uids_path)

    else:
        # Log "no suitable item"
        print("----------> no suitable item")
        log_problem_uids(points_df, "no suitable item", problematic_uids_path)
        
        
high_sev = after2017_near_water[(after2017_near_water.severity >= 3)]
high_sev.sort_values(by = "abun", ascending = False, inplace = True)
high_sev_cases = list(dict.fromkeys(high_sev.case.to_list()))

for iii, sel_case in enumerate(high_sev_cases[:100]):
    
    print("----------> HELLO High Sev", iii, sel_case)
    
    #points_df = low_sev[low_sev.case == sel_case].copy()
    points_df = after2017_near_water[after2017_near_water.case == sel_case].copy()
    if 1 in points_df.severity.to_list() or 2 in points_df.severity.to_list():
        print("----------> Not pure in severities ", iii, sel_case)
        continue
    
    if os.path.exists(uids_processed_path):
        try:
            processed_df = pd.read_excel(uids_processed_path)
            if not processed_df.empty:
                # Convert to string to ensure matching types
                processed_uids = processed_df['uid'].astype(str).unique()
                
                # Keep only UIDs that are NOT in the processed list
                points_df = points_df[~points_df['uid'].astype(str).isin(processed_uids)]
        except Exception as e:
            print(f"Warning: Could not read processed UIDs file: {e}")

    # If no points left to process for this case, skip to next case
    if points_df.empty:
        print(f"Case {sel_case}: All UIDs already processed. Skipping.")
        continue

    abun = str(int(max(points_df["abun"].to_list())))
    dates = [i.strftime('%Y-%m-%d') for i in points_df["date"].value_counts().index.to_list()]
    found = False
    item = None
    
    for date in dates:
        print("----------> date is ", date)
        item_details = find_satelite_images(points_df["lat"].mean(), points_df["lon"].mean(), date)
        if item_details.empty:
            continue
            
        item, found = select_item(item_details)
        if found:
            break
            
    if found:
        final_date = item.datetime
        print("----------> Found  item!!!  with date = ", final_date)
        pixel_size = 365
        
        out_folder = os.path.join(sat_test_folder, f"case{sel_case}_abun{abun}_{pixel_size}_date{final_date}")
        
        # extract tile
        extract_and_save_tile(item.item_obj, points_df, after2017_near_water, initial_pixel_size=pixel_size,
            save_data=True, output_path=out_folder, print_images=False)
        
        print("----------> Finished extract and save")
        
        # run cyfi pipeline
        folder_ok, new_path = predict_using_cyfi_pipeline(out_folder, final_date)
        
        print("----------> Finished cyfi")
        
        if folder_ok:
            summary_file_path_name = os.path.join(sat_test_folder, "Master_Report.xlsx")
            processed_uids_path_name = uids_processed_path
            
            # updates the report and adds the UIDs to uids_processed.xlsx
            update_excel_report(new_path, excel_path=summary_file_path_name, uids_tracker_path=processed_uids_path_name)
            print("----------> Excel report updated")
        else:
            #Log "cyfi failed"
            print("----------> cyfi failed")
            log_problem_uids(points_df, "cyfi failed", problematic_uids_path)

    else:
        # Log "no suitable item"
        print("----------> no suitable item")
        log_problem_uids(points_df, "no suitable item", problematic_uids_path)
        
