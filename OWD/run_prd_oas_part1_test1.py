# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:05:52 2026

@author: K. Pikounis
"""
import pandas as pd
import numpy as np
import argparse
import os
import logging
import traceback
import planetary_computer as pc
from pystac_client import Client
import rioxarray
from pyproj import Transformer, CRS
from datetime import timedelta

# Setup STAC Client
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def setup_logger(output_root, base_name):
    """Sets up a text logger for detailed debugging."""
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    log_file = os.path.join(output_root, f"prescreen_clustered_log_{base_name}.log")
    
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
    """Calculates distance in meters between two points."""
    R = 6371000  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def check_scl_water(item, lat, lon):
    """Downloads ONLY the SCL band to check for water pixels."""
    try:
        target_crs = CRS.from_string(item.properties["proj:code"])
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        center_x, center_y = transformer.transform(lon, lat)

        half_side = 3840
        minx, maxx = center_x - half_side, center_x + half_side
        miny, maxy = center_y - half_side, center_y + half_side

        scl_href = pc.sign(item.assets["SCL"].href)
        da_scl = rioxarray.open_rasterio(scl_href)
        da_clip = da_scl.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=target_crs)
        
        water_pixels = np.sum(da_clip.values == 6)
        return int(water_pixels) > 0
    except Exception as e:
        logging.warning(f"SCL Check failed for {item.id}: {e}")
        return False

def search_stac(lat, lon, target_date, y_days):
    """Searches PC for Sentinel-2 items within +- Y days, safely handling API timeouts."""
    buffer_deg = 0.1 
    bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
    
    start_date = pd.to_datetime(target_date) - timedelta(days=y_days)
    end_date = pd.to_datetime(target_date) + timedelta(days=y_days)
    date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    
    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"], 
            bbox=bbox, datetime=date_range, query={"eo:cloud_cover": {"lt": 50}}
        )
        items = list(search.items())
        items.sort(key=lambda x: x.properties["eo:cloud_cover"])
        return items
    except Exception as e:
        logging.error(f"    -> [API ERROR] STAC search timed out or failed for {target_date.date()}: {str(e)}")
        return []

def is_unsuppressed(val):
    return pd.isna(val) or str(val).strip() == ""

def run_clustered_prescreener(csv_path, output_dir, x_days, y_days, purity_days, n_cases):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    setup_logger(output_dir, base_name)
    
    logging.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False) if csv_path.endswith('.csv') else pd.read_excel(csv_path)
    
    # 1. Standardize Columns & Parse Data Safely
    lat_col = 'decimalLatitude' if 'decimalLatitude' in df.columns else 'decimalLa'
    lon_col = 'decimalLongitude' if 'decimalLongitude' in df.columns else 'decimalLo'
    
    df['Event_Date_Parsed'] = pd.to_datetime(df['eventDate'], errors='coerce').dt.tz_localize(None)
    df['qty_numeric'] = pd.to_numeric(df['organismQuantity'] if 'organismQuantity' in df.columns else df['organismC'], errors='coerce').fillna(0)
    
    # Initialize Tracking Columns
    if 'sat_item' not in df.columns: df['sat_item'] = ""
    if 'cluster_ID' not in df.columns: df['cluster_ID'] = np.nan
    if 'suppressed_by' not in df.columns: df['suppressed_by'] = ""
    if 'removed_by_ID' not in df.columns: df['removed_by_ID'] = "" 
    
    # Pre-sort the entire DataFrame by concentration descending to identify our Seed points
    df_sorted = df.sort_values(by='qty_numeric', ascending=False)
    
    cluster_id_counter = 1
    
    logging.info(f"Starting Phase 1: Target {n_cases} HABs and {n_cases} Cleans | Purity Window: +-{purity_days} days...")
    
    # Main Loop over Potential Seed Points
    for seed_idx, seed_row in df_sorted.iterrows():
        if pd.notna(df.at[seed_idx, 'cluster_ID']): continue
        if not is_unsuppressed(df.at[seed_idx, 'suppressed_by']): continue
        if not is_unsuppressed(df.at[seed_idx, 'removed_by_ID']): continue
        if str(seed_row.get('is_HAB', '')).strip().upper() != 'YES': continue
        
        seed_lat, seed_lon = seed_row[lat_col], seed_row[lon_col]
        seed_id = seed_row.get('id_x', seed_idx)
        
        logging.info(f"\n{'='*60}\nCREATING CLUSTER {cluster_id_counter} | Center: ({seed_lat:.4f}, {seed_lon:.4f}) | Seed ID: {seed_id}\n{'='*60}")
        
        # --- A. DEFINE THE SPATIAL CLUSTER ---
        dist_mask = haversine_distance(seed_lat, seed_lon, df[lat_col], df[lon_col]) <= 7680
        unclustered_mask = df['cluster_ID'].isna()
        
        cluster_mask = dist_mask & unclustered_mask
        df.loc[cluster_mask, 'cluster_ID'] = cluster_id_counter
        
        logging.info(f"  -> Captured {cluster_mask.sum()} raw entries in Cluster {cluster_id_counter}.")
        cluster_indices = df[cluster_mask].index
        
        # --- B. FIND `N` HAB CASES IN THIS CLUSTER ---
        hab_mask = (df.loc[cluster_indices, 'is_HAB'].str.strip().str.upper() == 'YES')
        cluster_habs = df.loc[cluster_indices][hab_mask].sort_values(by='qty_numeric', ascending=False)
        
        habs_found = 0
        for cand_idx, cand_row in cluster_habs.iterrows():
            if habs_found >= n_cases: break
            if not is_unsuppressed(df.at[cand_idx, 'suppressed_by']): continue
            if not is_unsuppressed(df.at[cand_idx, 'removed_by_ID']): continue
            
            cand_date = cand_row['Event_Date_Parsed']
            logging.info(f"  [HAB Search] Trying ID {cand_row.get('id_x')} | Qty: {cand_row['qty_numeric']} | Date: {cand_date.date()}")
            
            items = search_stac(seed_lat, seed_lon, cand_date, y_days)
            for item in items:
                if check_scl_water(item, seed_lat, seed_lon):
                    logging.info(f"    -> [SUCCESS] Found HAB item: {item.id}")
                    df.at[cand_idx, 'sat_item'] = item.id
                    habs_found += 1
                    
                    time_mask = abs(df.loc[cluster_indices, 'Event_Date_Parsed'] - cand_date).dt.days <= x_days
                    suppress_mask = time_mask & (df.loc[cluster_indices, 'sat_item'] == "") 
                    df.loc[cluster_indices[suppress_mask], 'suppressed_by'] = f"HAB_{cand_row.get('id_x')}"
                    break
        
        # --- C. FIND `N` CLEAN (NON-HAB) CASES IN THIS CLUSTER ---
        non_hab_mask = (df.loc[cluster_indices, 'is_HAB'].str.strip().str.upper() == 'NO')
        cluster_cleans = df.loc[cluster_indices][non_hab_mask].sort_values(by='qty_numeric', ascending=True)
        
        cleans_found = 0
        for cand_idx, cand_row in cluster_cleans.iterrows():
            if cleans_found >= n_cases: break
            if not is_unsuppressed(df.at[cand_idx, 'suppressed_by']): continue
            if not is_unsuppressed(df.at[cand_idx, 'removed_by_ID']): continue
            
            cand_date = cand_row['Event_Date_Parsed']
            logging.info(f"  [Clean Search] Trying ID {cand_row.get('id_x')} | Qty: {cand_row['qty_numeric']} | Date: {cand_date.date()}")
            
            # --- THE NEW STRICT PURITY CHECK ---
            purity_time_mask = abs(df.loc[cluster_indices, 'Event_Date_Parsed'] - cand_date).dt.days <= purity_days
            contaminants = df.loc[cluster_indices][
                purity_time_mask & 
                ((df.loc[cluster_indices, 'is_HAB'].str.strip().str.upper() == 'YES') | 
                 (df.loc[cluster_indices, 'qty_numeric'] > 10000))
            ]
            
            if not contaminants.empty:
                logging.info(f"    -> [!] Purity Check Failed: Contamination found within +-{purity_days} days. Skipping.")
                continue
            # -----------------------------------
                
            items = search_stac(seed_lat, seed_lon, cand_date, y_days)
            for item in items:
                if check_scl_water(item, seed_lat, seed_lon):
                    logging.info(f"    -> [SUCCESS] Found Clean item: {item.id}")
                    df.at[cand_idx, 'sat_item'] = item.id
                    cleans_found += 1
                    
                    time_mask = abs(df.loc[cluster_indices, 'Event_Date_Parsed'] - cand_date).dt.days <= x_days
                    suppress_mask = time_mask & (df.loc[cluster_indices, 'sat_item'] == "")
                    df.loc[cluster_indices[suppress_mask], 'suppressed_by'] = f"Clean_{cand_row.get('id_x')}"
                    break
                    
        # --- D. FINAL CLUSTER CLEANUP ---
        leftover_mask = (df.loc[cluster_indices, 'sat_item'] == "") & (df.loc[cluster_indices, 'suppressed_by'] == "")
        df.loc[cluster_indices[leftover_mask], 'suppressed_by'] = f"Cluster_{cluster_id_counter}_Cleanup"
        
        logging.info(f"  Cluster {cluster_id_counter} Complete | Yield: {habs_found} HABs, {cleans_found} Cleans.")
        cluster_id_counter += 1
        
        if cluster_id_counter % 20 == 0:
            out_file = os.path.join(output_dir, f"{base_name}_clustered_part_{cluster_id_counter}.csv")
            df.drop(columns=['Event_Date_Parsed', 'qty_numeric'], errors='ignore').to_csv(out_file, index=False)
            logging.info(f"--- Checkpoint Saved: {out_file} ---")

    # Final Save
    df.drop(columns=['Event_Date_Parsed', 'qty_numeric'], inplace=True, errors='ignore')
    final_file = os.path.join(output_dir, f"{base_name}_clustered_FINAL.csv")
    df.to_csv(final_file, index=False)
    logging.info(f"\nPipeline Complete! Generated {cluster_id_counter - 1} spatial clusters.")
    logging.info(f"Final output saved to: {final_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid-Based Clustered HAB API Pre-Screener")
    parser.add_argument("--input", type=str, required=True, help="Input CSV/Excel file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    #try:
    #    n_cases = int(input("Enter 'N' (Max number of HAB and Clean pairs per cluster, e.g., 2): "))
    #    x_days = int(input("Enter 'X' (Days for temporal suppression window inside cluster, e.g., 14): "))
    #    y_days = int(input("Enter 'Y' (Days for Satellite STAC search window, e.g., 5): "))
    #    purity_days = int(input("Enter 'Purity Window' (Days strictly clean around Non-HABs, e.g., 10): "))
    #except ValueError:
    #    print("Invalid input. Defaulting to N=2, X=14, Y=5, Purity=10.")
    #    n_cases, x_days, y_days, purity_days = 2, 14, 5, 10
    
    n_cases, x_days, y_days, purity_days = 2, 14, 5, 10
    run_clustered_prescreener(args.input, args.output, x_days, y_days, purity_days, n_cases)