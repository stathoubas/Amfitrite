# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 00:43:09 2026

@author: K. Pikounis
"""

import pandas as pd
import numpy as np
import argparse
import os

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

def cross_reference_datasets(florida_csv, new_dataset_csv, output_csv, x_days=14, distance_m=7680):
    print(f"Loading reference dataset: {florida_csv}")
    try:
        # Added low_memory=False to suppress the DtypeWarning
        df_fl = pd.read_csv(florida_csv, low_memory=False)
    except Exception as e:
        print(f"Error loading Florida dataset: {e}")
        return

    print(f"Loading target dataset to be filtered: {new_dataset_csv}")
    try:
        df_new = pd.read_csv(new_dataset_csv, low_memory=False) if new_dataset_csv.endswith('.csv') else pd.read_excel(new_dataset_csv)
    except Exception as e:
        print(f"Error loading new dataset: {e}")
        return

    # 1. Prepare Florida Reference Data
    df_fl['Event_Date_Parsed'] = pd.to_datetime(df_fl['eventDate'], errors='coerce').dt.tz_localize(None)
    df_fl['qty_numeric'] = pd.to_numeric(df_fl['organismQuantity'], errors='coerce').fillna(0)
    
    # Filter for is_HAB == Yes AND sat_item is NOT empty
    hab_mask = (df_fl['is_HAB'].str.strip().str.upper() == 'YES')
    sat_mask = df_fl['sat_item'].notna() & (df_fl['sat_item'].str.strip() != "")
    df_fl_valid = df_fl[hab_mask & sat_mask].copy()
    
    # Sort by concentration descending to match the main pipeline's processing order
    df_fl_valid = df_fl_valid.sort_values(by='qty_numeric', ascending=False).reset_index(drop=True)
    
    # Create the ascending "processing order" count (1, 2, 3...)
    df_fl_valid['processing_order'] = df_fl_valid.index + 1
    
    print(f"Found {len(df_fl_valid)} valid, pre-screened HAB events in the Florida dataset to use as filters.")

    # 2. Prepare the New Target Dataset
    # FIX: Strip timezones here too so they perfectly match!
    df_new['Event_Date_Parsed'] = pd.to_datetime(df_new['eventDate'], errors='coerce').dt.tz_localize(None)
    
    # Initialize the tracking columns as empty strings
    df_new['removed_by_ID'] = ""
    df_new['removed_by_num'] = ""
    
    # Handle potentially truncated column names from Excel (decimalLa vs decimalLatitude)
    lat_col = 'decimalLatitude' if 'decimalLatitude' in df_new.columns else 'decimalLa'
    lon_col = 'decimalLongitude' if 'decimalLongitude' in df_new.columns else 'decimalLo'

    print(f"Scanning new dataset ({len(df_new)} rows) for overlaps within {distance_m}m and +-{x_days} days...")
    
    # 3. Apply the Sweeping Logic
    overlap_count = 0
    
    for _, fl_row in df_fl_valid.iterrows():
        fl_lat = float(fl_row['decimalLatitude'])
        fl_lon = float(fl_row['decimalLongitude'])
        fl_date = fl_row['Event_Date_Parsed']
        fl_hab_id = str(fl_row.get('HAB_ID', fl_row.get('id_x', 'Unknown_ID')))
        fl_num = fl_row['processing_order']
        
        if pd.isna(fl_date):
            continue
            
        # Spatial mask (Haversine distance)
        dist_mask = haversine_distance(fl_lat, fl_lon, df_new[lat_col], df_new[lon_col]) <= distance_m
        
        # Temporal mask (+- X days)
        time_mask = abs(df_new['Event_Date_Parsed'] - fl_date).dt.days <= x_days
        
        # Ensure we only overwrite rows that haven't already been removed by a HIGHER priority Florida HAB
        unassigned_mask = df_new['removed_by_ID'] == ""
        
        # Combine masks
        match_mask = dist_mask & time_mask & unassigned_mask
        
        matches = match_mask.sum()
        if matches > 0:
            df_new.loc[match_mask, 'removed_by_ID'] = fl_hab_id
            df_new.loc[match_mask, 'removed_by_num'] = fl_num
            overlap_count += matches
            
    # 4. Cleanup and Save
    df_new.drop(columns=['Event_Date_Parsed'], inplace=True, errors='ignore')
    
    df_new.to_csv(output_csv, index=False)
    print(f"\nFinished! Found and flagged {overlap_count} overlapping entries.")
    print(f"Modified dataset saved to: {output_csv}")
    

def cross_reference_datasets_v2(florida_csv, new_dataset_csv, output_csv, x_days=14, distance_m=7680):
    '''
    uses column output_folder to locate those cases already processed by florida dataset
    '''
    print(f"Loading reference dataset: {florida_csv}")
    try:
        df_fl = pd.read_csv(florida_csv, low_memory=False)
    except Exception as e:
        print(f"Error loading Florida dataset: {e}")
        return

    # Safety check for the new column
    if 'output_folder' not in df_fl.columns:
        print("Error: 'output_folder' column not found in the Florida dataset. Please check the file.")
        return

    print(f"Loading target dataset to be filtered: {new_dataset_csv}")
    try:
        df_new = pd.read_csv(new_dataset_csv, low_memory=False) if new_dataset_csv.endswith('.csv') else pd.read_excel(new_dataset_csv)
    except Exception as e:
        print(f"Error loading new dataset: {e}")
        return

    # 1. Prepare Florida Reference Data
    # Strip timezones from the dates
    df_fl['Event_Date_Parsed'] = pd.to_datetime(df_fl['eventDate'], errors='coerce').dt.tz_localize(None)
    df_fl['qty_numeric'] = pd.to_numeric(df_fl['organismQuantity'], errors='coerce').fillna(0)
    
    # Filter for is_HAB == Yes AND output_folder is NOT empty
    hab_mask = (df_fl['is_HAB'].str.strip().str.upper() == 'YES')
    
    # Robust check for successfully processed folders
    processed_mask = df_fl['output_folder'].notna() & \
                     (df_fl['output_folder'].astype(str).str.strip() != "") & \
                     (df_fl['output_folder'].astype(str).str.strip().str.lower() != "nan")
                     
    df_fl_valid = df_fl[hab_mask & processed_mask].copy()
    
    # Sort by concentration descending to match the main pipeline's processing order
    df_fl_valid = df_fl_valid.sort_values(by='qty_numeric', ascending=False).reset_index(drop=True)
    
    # Create the ascending "processing order" count (1, 2, 3...)
    df_fl_valid['processing_order'] = df_fl_valid.index + 1
    
    print(f"Found {len(df_fl_valid)} fully processed HAB events in the Florida dataset to use as filters.")

    # 2. Prepare the New Target Dataset
    # Strip timezones here too so they perfectly match!
    df_new['Event_Date_Parsed'] = pd.to_datetime(df_new['eventDate'], errors='coerce').dt.tz_localize(None)
    
    # Initialize the tracking columns as empty strings
    df_new['removed_by_ID'] = ""
    df_new['removed_by_num'] = ""
    
    # Handle potentially truncated column names from Excel (decimalLa vs decimalLatitude)
    lat_col = 'decimalLatitude' if 'decimalLatitude' in df_new.columns else 'decimalLa'
    lon_col = 'decimalLongitude' if 'decimalLongitude' in df_new.columns else 'decimalLo'

    print(f"Scanning new dataset ({len(df_new)} rows) for overlaps within {distance_m}m and +-{x_days} days...")
    
    # 3. Apply the Sweeping Logic
    overlap_count = 0
    
    for _, fl_row in df_fl_valid.iterrows():
        fl_lat = float(fl_row['decimalLatitude'])
        fl_lon = float(fl_row['decimalLongitude'])
        fl_date = fl_row['Event_Date_Parsed']
        fl_hab_id = str(fl_row.get('HAB_ID', fl_row.get('id_x', 'Unknown_ID')))
        fl_num = fl_row['processing_order']
        
        if pd.isna(fl_date):
            continue
            
        # Spatial mask (Haversine distance)
        dist_mask = haversine_distance(fl_lat, fl_lon, df_new[lat_col], df_new[lon_col]) <= distance_m
        
        # Temporal mask (+- X days)
        time_mask = abs(df_new['Event_Date_Parsed'] - fl_date).dt.days <= x_days
        
        # Ensure we only overwrite rows that haven't already been removed by a HIGHER priority Florida HAB
        unassigned_mask = df_new['removed_by_ID'] == ""
        
        # Combine masks
        match_mask = dist_mask & time_mask & unassigned_mask
        
        matches = match_mask.sum()
        if matches > 0:
            df_new.loc[match_mask, 'removed_by_ID'] = fl_hab_id
            df_new.loc[match_mask, 'removed_by_num'] = fl_num
            overlap_count += matches
            
    # 4. Cleanup and Save
    df_new.drop(columns=['Event_Date_Parsed'], inplace=True, errors='ignore')
    
    df_new.to_csv(output_csv, index=False)
    print(f"\nFinished! Found and flagged {overlap_count} overlapping entries.")
    print(f"Modified dataset saved to: {output_csv}")



'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Reference and Suppress Overlapping Datasets")
    parser.add_argument("--florida", type=str, required=True, help="Path to the PRE-SCREENED Florida FINAL csv")
    parser.add_argument("--new_data", type=str, required=True, help="Path to the new dataset to be filtered")
    parser.add_argument("--output", type=str, required=True, help="Path to save the modified new dataset")
    parser.add_argument("--days", type=int, default=14, help="Temporal window in days (default: 14)")
    args = parser.parse_args()
    
    #cross_reference_datasets(args.florida, args.new_data, args.output, x_days=args.days, distance_m=7680)
    cross_reference_datasets_v2(args.florida, args.new_data, args.output, x_days=args.days, distance_m=7680)
'''