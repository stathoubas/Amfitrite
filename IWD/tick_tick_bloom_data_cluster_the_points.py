# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 16:58:16 2025

@author: K. Pikounis
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from time import time


#([lon1, lat1], [lon2, lat2])
def Haversine(pt_1: np.ndarray, pt_2: np.ndarray) -> float:
    R = 6371.0  # Earth radius in kilometers
    # Convert latitude and longitude from degrees to radians
    lon_1, lat_1 = np.radians(pt_1)
    lon_2, lat_2 = np.radians(pt_2)
    # Compute differences in coordinates 
    dlat = lat_2 - lat_1
    dlon = lon_2 - lon_1
    # Haversine formula 
    a = np.sin(dlat / 2)**2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(dlon / 2)**2
    c = 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a))
    distance = (R * c) * 1000 # in meters
    return distance


def perform_clustering(df: pd.DataFrame, time_window_days: int, max_distance_meters: float) -> pd.DataFrame:
    """
    Performs spatio-temporal clustering based on the described logic.

    Args:
        df: The input DataFrame.
        time_window_days: The number of days (+/-) for the temporal window.
        max_distance_meters: The radius (C) for Haversine distance.

    Returns:
        A new DataFrame with 'case' and 'case_str' columns populated.
    """
    
    # --- 1. Data Preparation ---
    df_proc = df.copy()
    
    # Ensure date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_proc['date']):
        try:
            df_proc['date'] = pd.to_datetime(df_proc['date'], format='%Y%m%d')
        except ValueError:
            print("Could not parse 'date' column with format YYYYMMDD. Please ensure it's in this format.")
            return df

    # Initialize cluster columns
    df_proc['case'] = pd.NA
    df_proc['case'] = df_proc['case'].astype(pd.Int64Dtype())
    df_proc['case_str'] = pd.NA
    df_proc['case_str'] = df_proc['case_str'].astype(object)

    # Pre-calculate lon/lat array and index mapping for speed
    lon_lat_array = df_proc[['lon', 'lat']].values
    get_loc = df_proc.index.get_loc # Helper to map index label to integer position

    cluster_id_counter = 0
    processed_indices = set() # Tracks all rows that are part of *any* cluster
    
    time_delta = pd.Timedelta(days=time_window_days)
    lat_delta = 0.01
    lon_delta = 0.01


    # --- 2. Main Clustering Loop ---
    t0 = time()
    for nnn, i in enumerate(df_proc.index):
        # If this row is already in a cluster, skip it
        if i in processed_indices:
            continue
        if nnn/250 in [i for i in range(240)]:
            t1 = time()
            print(nnn, "in ", round((t1-t0), 1), "s")
            t0 = t1

        # Start a new cluster
        current_cluster_id = cluster_id_counter
        process_stack = [i] # Use a stack for Depth-First Search (DFS)
        
        # Tracks rows visited *during this specific cluster search*
        # to prevent infinite loops (e.g., A -> B, B -> A)
        visited_in_this_cluster = set()

        while process_stack:
            current_idx = process_stack.pop()

            if current_idx in visited_in_this_cluster:
                continue
            
            visited_in_this_cluster.add(current_idx)
            processed_indices.add(current_idx) # Mark as globally processed

            current_row = df_proc.loc[current_idx]
            current_case = current_row['case']

            # --- 3. Handle Assignment & Merging ---
            if pd.isna(current_case):
                # This row is unassigned. Assign it to the new cluster.
                df_proc.at[current_idx, 'case'] = current_cluster_id
                df_proc.at[current_idx, 'case_str'] = str(current_cluster_id)
            
            elif current_case != current_cluster_id:
                # This row belongs to an OLD cluster. MERGE required.
                old_cluster_id = current_case
                
                # Find all rows belonging to the old cluster
                merge_indices = df_proc.index[df_proc['case'] == old_cluster_id]
                
                # Update all rows from the old cluster to the new cluster ID
                df_proc.loc[merge_indices, 'case'] = current_cluster_id
                
                # Append the merge string
                merge_str = f" -> {current_cluster_id}"
                df_proc.loc[merge_indices, 'case_str'] = df_proc.loc[merge_indices, 'case_str'] + merge_str
                
                # Add all merged rows to the stack (if not visited yet)
                # to process *their* neighbors under the new cluster ID
                for merge_idx in merge_indices:
                    if merge_idx not in visited_in_this_cluster:
                        process_stack.append(merge_idx)
            
            # --- 4. Find Neighbors ---
            
            # Define search windows
            min_date, max_date = current_row['date'] - time_delta, current_row['date'] + time_delta
            min_lat, max_lat = current_row['lat'] - lat_delta, current_row['lat'] + lat_delta
            min_lon, max_lon = current_row['lon'] - lon_delta, current_row['lon'] + lon_delta

            pt_1 = lon_lat_array[get_loc(current_idx)]

            # 4a. Fast Bounding-Box Filter
            candidate_indices = df_proc.index[
                (df_proc['date'] >= min_date) & (df_proc['date'] <= max_date) &
                (df_proc['lat'] >= min_lat) & (df_proc['lat'] <= max_lat) &
                (df_proc['lon'] >= min_lon) & (df_proc['lon'] <= max_lon)
            ]

            # 4b. Precise Haversine Filter
            for j in candidate_indices:
                # Don't check against self or already-visited rows
                if j == current_idx or j in visited_in_this_cluster:
                    continue

                pt_2 = lon_lat_array[get_loc(j)]
                distance = Haversine(pt_1, pt_2)

                if distance < max_distance_meters:
                    # Found a true neighbor. Add it to the stack to be processed.
                    process_stack.append(j)
        
        # Finished with all connected components for this cluster
        cluster_id_counter += 1

    print("Clustering complete.")
    return df_proc

df = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\tick tick bloom data\CAML_cyanobacteria_abundance_20211229_R1_date_extended.csv")

n = 0
for it, row in df.iterrows():
    
    
    

### load df

### filter lon lat +- 0.01
### filter date +- 10 days

### for the filtered entries apply the Haversine to find neighbours <= 1.000 m

