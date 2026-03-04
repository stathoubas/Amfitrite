# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:16:09 2026

@author: K. Pikounis

basd on 
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from time import time

def Haversine(pt_1: np.ndarray, pt_2: np.ndarray) -> float:
    """
    Calculate the great-circle distance between two points
    on the Earth (specified in decimal degrees)
    
    pt_1 and pt_2 are [lon, lat] numpy arrays.
    Returns distance in meters.
    """
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



def perform_spatiotemporal_clustering(df: pd.DataFrame, max_distance_meters: float, time_window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs spatial clustering using an exact Haversine radius (optimized with a bounding box),
    followed by temporal sub-clustering. 
    Returns the updated original dataframe and an aggregated summary dataframe.
    time_window_days = half time, so a +-10 days time window time_window_days = 15 NOT 30
    """
    print("Preparing data for spatio-temporal clustering...")
    df_proc = df.copy()
    
    # Column names based on the target dataframe structure
    lat_col = 'decimalLatitude'
    lon_col = 'decimalLongitude'
    date_col = 'eventDate'
    hab_id_col = 'HAB_ID'
    org_qty_col = 'organismQuantity'
    
    # Ensure datetime format for the date column
    if not pd.api.types.is_datetime64_any_dtype(df_proc[date_col]):
        try:
            df_proc[date_col] = pd.to_datetime(df_proc[date_col])
        except ValueError:
            print(f"Could not parse '{date_col}'. Please ensure it is in a valid datetime format.")
            return df, pd.DataFrame()
            
    # Initialize ID columns
    df_proc['PosID'] = pd.NA
    df_proc['PosID'] = df_proc['PosID'].astype('Int64')
    df_proc['DatId'] = pd.NA
    df_proc['DatId'] = df_proc['DatId'].astype('Int64')
    df_proc['PosID-DatId'] = pd.NA
    df_proc['PosID-DatId'] = df_proc['PosID-DatId'].astype('string')
    
    METERS_PER_DEG_LAT = 111132.0
    
    # Nested Helper Function
    def get_points_in_radius(c_lat, c_lon, available_indices):
        """
        Finds points within an exact radius by first using a fast bounding box pre-filter,
        then applying the exact Haversine calculation to the remaining candidates.
        """
        # 1. FAST PRE-FILTER (Bounding Box)
        d_lat = max_distance_meters / METERS_PER_DEG_LAT
        d_lon = max_distance_meters / (METERS_PER_DEG_LAT * np.cos(np.radians(c_lat)))
        
        min_lat, max_lat = c_lat - d_lat, c_lat + d_lat
        min_lon, max_lon = c_lon - d_lon, c_lon + d_lon
        
        subset = df_proc.loc[available_indices]
        mask = (
            (subset[lat_col] >= min_lat) & (subset[lat_col] <= max_lat) &
            (subset[lon_col] >= min_lon) & (subset[lon_col] <= max_lon)
        )
        candidates = subset[mask]
        
        # 2. EXACT FILTER (Haversine)
        exact_indices = []
        pt1 = np.array([c_lon, c_lat]) # Haversine function expects [lon, lat]
        
        for idx, row in candidates.iterrows():
            pt2 = np.array([row[lon_col], row[lat_col]])
            dist = Haversine(pt1, pt2)
            if dist <= max_distance_meters:
                exact_indices.append(idx)
                
        return exact_indices

    # =========================================================
    # Step 1: Spatial Clustering (Assign PosID)
    # =========================================================
    print("Starting Spatial Clustering...")
    case_counter = 1
    all_indices = df_proc.index.tolist()
    
    for i in all_indices:      
        if not pd.isna(df_proc.at[i, 'PosID']):
            continue
            
        current_lat = df_proc.at[i, lat_col]
        current_lon = df_proc.at[i, lon_col]
        
        # Search among points with NO PosID assigned yet
        unassigned_indices = df_proc.index[df_proc['PosID'].isna()].tolist()
        candidates_1 = get_points_in_radius(current_lat, current_lon, unassigned_indices)
        
        if not candidates_1:
            continue

        # Find barycenter and repeat
        subset_1 = df_proc.loc[candidates_1]
        bary_lat_1 = subset_1[lat_col].mean()
        bary_lon_1 = subset_1[lon_col].mean()
        
        candidates_2 = get_points_in_radius(bary_lat_1, bary_lon_1, unassigned_indices)
        final_candidates = candidates_2
        
        # Check for changes and do a 3rd time if needed to stabilize the barycenter
        if set(candidates_1) != set(candidates_2) and candidates_2:
            subset_2 = df_proc.loc[candidates_2]
            bary_lat_2 = subset_2[lat_col].mean()
            bary_lon_2 = subset_2[lon_col].mean()
            candidates_3 = get_points_in_radius(bary_lat_2, bary_lon_2, unassigned_indices)
            final_candidates = candidates_3
            
        if final_candidates:
            df_proc.loc[final_candidates, 'PosID'] = case_counter
            case_counter += 1

    # =========================================================
    # Step 2: Temporal Sub-clustering (Assign DatId)
    # =========================================================
    print("Starting Temporal Sub-clustering...")
    time_delta = pd.Timedelta(days=time_window_days)
    
    for pos_id in df_proc['PosID'].dropna().unique():
        # Get indices for this specific spatial cluster and sort them chronologically
        cluster_indices = df_proc[df_proc['PosID'] == pos_id].index.tolist()
        cluster_indices = df_proc.loc[cluster_indices].sort_values(date_col).index.tolist()
        
        dat_id_counter = 1
        
        while cluster_indices:
            current_idx = cluster_indices[0]
            current_date = df_proc.at[current_idx, date_col]
            
            # Identify window boundaries
            min_date = current_date - time_delta
            max_date = current_date + time_delta
            
            subset_df = df_proc.loc[cluster_indices]
            in_window_mask = (subset_df[date_col] >= min_date) & (subset_df[date_col] <= max_date)
            assigned_indices = subset_df[in_window_mask].index.tolist()
            
            # Apply assignments
            df_proc.loc[assigned_indices, 'DatId'] = dat_id_counter
            df_proc.loc[assigned_indices, 'PosID-DatId'] = f"{pos_id}-{dat_id_counter}"
            
            # Clean up the pool of available indices in this cluster
            for idx in assigned_indices:
                cluster_indices.remove(idx)
                
            dat_id_counter += 1

    # =========================================================
    # Step 3: Compute Barycenters and Aggregate Summary
    # =========================================================
    print("Computing sub-cluster properties and building summary...")
    
    # Initialize target columns in the main dataframe
    df_proc['bary_lat'] = pd.NA
    df_proc['bary_lon'] = pd.NA
    df_proc['bary_date'] = pd.NaT
    df_proc['cluster_size'] = pd.NA
    
    df_proc['bary_lat'] = df_proc['bary_lat'].astype('float64')
    df_proc['bary_lon'] = df_proc['bary_lon'].astype('float64')
    df_proc['cluster_size'] = df_proc['cluster_size'].astype('Int64')

    summary_records = []

    # Group by the combined spatio-temporal identifier
    grouped = df_proc.dropna(subset=['PosID-DatId']).groupby('PosID-DatId')
    
    for gen_id, group in grouped:
        bary_lat = group[lat_col].mean()
        bary_lon = group[lon_col].mean()
        bary_date = group[date_col].mean()
        c_size = len(group)
        
        # 3.1 Append barycenter and size to the original dataframe
        df_proc.loc[group.index, 'bary_lat'] = bary_lat
        df_proc.loc[group.index, 'bary_lon'] = bary_lon
        df_proc.loc[group.index, 'bary_date'] = bary_date
        df_proc.loc[group.index, 'cluster_size'] = c_size
        
        # 3.2 Prepare lists and metrics for the summary dataframe
        pos_id = group['PosID'].iloc[0]
        dat_id = group['DatId'].iloc[0]
        hab_ids = group[hab_id_col].tolist()
        idxs = group.index.tolist()
        org_qtys = group[org_qty_col].tolist()
        max_org_qty = group[org_qty_col].max()
        
        summary_records.append({
            'PosID': pos_id,
            'DatId': dat_id,
            'PosID-DatId': gen_id,
            'cluster_size': c_size,
            'bary_lat': bary_lat,
            'bary_lon': bary_lon,
            'bary_date': bary_date,
            'HAB_IDs': hab_ids,
            'idxs': idxs,
            'organismQuantities': org_qtys,
            'max_organismQuantity': max_org_qty
        })

    df_summary = pd.DataFrame(summary_records)
    
    print("Clustering complete!")
    return df_proc, df_summary


df = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Florida\Historic_Harmful_Algal_Bloom_Events_2015_-_2023_homogenised.csv")

df["eventDate"] = pd.to_datetime(df["eventDate"])

filtered_df = df[(df.eventDate >= "2016-01-01")&( df.minimumDepthInMeters <= 10 )].copy()

df_proc, df_summary = perform_spatiotemporal_clustering(filtered_df)
df_proc.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Florida\clustered.xlsx", index = False)
df_summary.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Florida\summary.xslx", index = False)