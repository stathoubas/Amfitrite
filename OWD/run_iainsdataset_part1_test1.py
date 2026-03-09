# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:49:14 2026

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

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def setup_logger(output_root, base_name):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    log_file = os.path.join(output_root, f"prescreen_priority_{base_name}.log")
    
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
    R = 6371000  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def check_scl_water(item, lat, lon):
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
        logging.error(f"    -> [API ERROR] STAC search failed for {target_date.date()}: {str(e)}")
        return []

def is_unsuppressed(val):
    return pd.isna(val) or str(val).strip() == ""

def temporal_dispersion_order(indices, df):
    """Orders a list of indices to maximize the temporal distance between consecutive picks."""
    if len(indices) <= 2:
        return list(indices)
        
    dates = df.loc[indices, 'Event_Date_Parsed']
    valid_mask = dates.notna()
    valid_indices = indices[valid_mask].tolist()
    
    if len(valid_indices) <= 2:
        return list(indices)
        
    # Sort indices strictly by actual Date
    sorted_indices = sorted(valid_indices, key=lambda idx: df.at[idx, 'Event_Date_Parsed'])
    
    ordered = []
    # 1. Grab the absolute First date
    ordered.append(sorted_indices.pop(0))
    # 2. Grab the absolute Last date
    ordered.append(sorted_indices.pop(-1))
    
    # 3. Greedy Max-Min selection for the remaining dates
    while sorted_indices:
        best_idx = None
        max_min_dist = -1
        
        for cand_idx in sorted_indices:
            cand_date = df.at[cand_idx, 'Event_Date_Parsed']
            # Find the distance to the CLOSEST date we have already picked
            min_dist = min([abs((cand_date - df.at[p, 'Event_Date_Parsed']).days) for p in ordered])
            
            # We want to MAXIMIZE that minimum distance
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = cand_idx
                
        ordered.append(best_idx)
        sorted_indices.remove(best_idx)
        
    # Append any dates that were NaT at the very end just in case
    ordered.extend(indices[~valid_mask].tolist())
    return ordered

def run_decoupled_prescreener(csv_path, output_dir, x_days, y_days1, y_days2, purity_days, gap_days, n_cases):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    setup_logger(output_dir, base_name)
    
    logging.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False) if csv_path.endswith('.csv') else pd.read_excel(csv_path)
    
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), 'latitude')
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), 'longitude')
    date_col = next((c for c in df.columns if 'date' in c.lower()), 'date')
    
    id_col = 'num' if 'num' in df.columns else 'generated_id'
    if id_col not in df.columns:
        df['generated_id'] = [f"Row_{i}" for i in range(len(df))]

    initial_len = len(df)
    df['Event_Date_Parsed'] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    df = df.dropna(subset=['Event_Date_Parsed']).reset_index(drop=True)
    if len(df) < initial_len:
        logging.info(f"Dropped {initial_len - len(df)} rows due to missing dates (NaT).")

    if 'is_HAB' not in df.columns:
        df['is_HAB'] = 'Yes'

    for col in ['sat_item', 'suppressed_by', 'removed_by_ID', 'priority']:
        if col not in df.columns: df[col] = ""
    if 'cluster_ID' not in df.columns: df['cluster_ID'] = np.nan
    
    cluster_id_counter = 1
    total_habs_found = 0
    total_cleans_found = 0

    # ==========================================
    # PHASE 1: PROCESS ALL HABs (TWO-PASS PRIORITY)
    # ==========================================
    logging.info(f"\n{'='*60}\nPHASE 1: PRIORITY HUNTING FOR HABs (Max {n_cases} per cluster)\n{'='*60}")
    
    # We iterate over unique locations to form clusters, instead of iterating over sorted concentrations
    df_habs = df[df['is_HAB'].str.strip().str.upper() == 'YES']
    
    for seed_idx, seed_row in df_habs.iterrows():
        if pd.notna(df.at[seed_idx, 'cluster_ID']) or not is_unsuppressed(df.at[seed_idx, 'suppressed_by']): continue
        
        seed_lat, seed_lon = seed_row[lat_col], seed_row[lon_col]
        seed_id = seed_row[id_col]
        
        dist_mask = haversine_distance(seed_lat, seed_lon, df[lat_col], df[lon_col]) <= 7680
        unclustered_mask = df['cluster_ID'].isna()
        cluster_mask = dist_mask & unclustered_mask
        df.loc[cluster_mask, 'cluster_ID'] = cluster_id_counter
        cluster_indices = df[cluster_mask].index
        
        hab_indices = cluster_indices[df.loc[cluster_indices, 'is_HAB'].str.strip().str.upper() == 'YES']
        
        # Order the indices by maximum temporal spread
        ordered_hab_indices = temporal_dispersion_order(hab_indices, df)
        
        logging.info(f"\n[HAB CLUSTER {cluster_id_counter}] Seed: {seed_id} | Raw Candidates: {len(ordered_hab_indices)}")
        
        habs_in_this_cluster = 0
        
        # --- PASS 1: y_days1 (Priority 1) ---
        logging.info(f"  -> Pass 1: Searching with +-{y_days1} days window")
        for cand_idx in ordered_hab_indices:
            if habs_in_this_cluster >= n_cases: break
            if not is_unsuppressed(df.at[cand_idx, 'suppressed_by']): continue
            if df.at[cand_idx, 'sat_item'] != "": continue 
            
            cand_date = df.at[cand_idx, 'Event_Date_Parsed']
            items = search_stac(seed_lat, seed_lon, cand_date, y_days1)
            for item in items:
                if check_scl_water(item, seed_lat, seed_lon):
                    logging.info(f"    [P1 SUCCESS] Found HAB item: {item.id} for date {cand_date.date()}")
                    df.at[cand_idx, 'sat_item'] = item.id
                    df.at[cand_idx, 'priority'] = 1
                    habs_in_this_cluster += 1
                    total_habs_found += 1
                    
                    time_mask = abs(df.loc[cluster_indices, 'Event_Date_Parsed'] - cand_date).dt.days <= x_days
                    suppress_mask = time_mask & (df.loc[cluster_indices, 'sat_item'] == "") 
                    df.loc[cluster_indices[suppress_mask], 'suppressed_by'] = f"HAB_{df.at[cand_idx, id_col]}"
                    break

        # --- PASS 2: y_days2 (Priority 2) ---
        if habs_in_this_cluster < n_cases:
            logging.info(f"  -> Pass 2: Fallback search with +-{y_days2} days window")
            for cand_idx in ordered_hab_indices:
                if habs_in_this_cluster >= n_cases: break
                if not is_unsuppressed(df.at[cand_idx, 'suppressed_by']): continue
                if df.at[cand_idx, 'sat_item'] != "": continue 
                
                cand_date = df.at[cand_idx, 'Event_Date_Parsed']
                items = search_stac(seed_lat, seed_lon, cand_date, y_days2)
                for item in items:
                    if check_scl_water(item, seed_lat, seed_lon):
                        logging.info(f"    [P2 SUCCESS] Found HAB item: {item.id} for date {cand_date.date()}")
                        df.at[cand_idx, 'sat_item'] = item.id
                        df.at[cand_idx, 'priority'] = 2
                        habs_in_this_cluster += 1
                        total_habs_found += 1
                        
                        time_mask = abs(df.loc[cluster_indices, 'Event_Date_Parsed'] - cand_date).dt.days <= x_days
                        suppress_mask = time_mask & (df.loc[cluster_indices, 'sat_item'] == "") 
                        df.loc[cluster_indices[suppress_mask], 'suppressed_by'] = f"HAB_{df.at[cand_idx, id_col]}"
                        break
                        
        leftover_habs = (df.loc[cluster_indices, 'is_HAB'].str.strip().str.upper() == 'YES') & (df.loc[cluster_indices, 'sat_item'] == "") & (df.loc[cluster_indices, 'suppressed_by'] == "")
        df.loc[cluster_indices[leftover_habs], 'suppressed_by'] = f"HAB_Cleanup_{cluster_id_counter}"
        
        cluster_id_counter += 1

    logging.info(f"\nPHASE 1 COMPLETE. Total HABs secured: {total_habs_found}")

    # ==========================================
    # PHASE 2: HUNT EXISTING CLEANS (Skipped for Incident Dataset)
    # ==========================================

    # ==========================================
    # PHASE 3: IMPUTING POTENTIAL NO CASES
    # ==========================================
    if total_cleans_found < total_habs_found:
        logging.info(f"\n{'='*60}\nPHASE 3: IMPUTING 'potential No' CASES\n{'='*60}")
        
        valid_hab_clusters = df[(df['is_HAB'].str.strip().str.upper() == 'YES') & (df['sat_item'] != "")]['cluster_ID'].dropna().unique()
        new_synthetic_rows = []
        
        for cid in valid_hab_clusters:
            if total_cleans_found >= total_habs_found: break
            
            cluster_mask = (df['cluster_ID'] == cid)
            all_dates = df[cluster_mask]['Event_Date_Parsed'].dt.normalize().dropna().sort_values().unique()
            
            if len(all_dates) == 0: continue
            
            seed_row = df[cluster_mask].iloc[0]
            c_lat, c_lon = seed_row[lat_col], seed_row[lon_col]
            
            candidate_dates = []
            
            if len(all_dates) > 1:
                for i in range(len(all_dates)-1):
                    gap = (all_dates[i+1] - all_dates[i]).days
                    if gap >= gap_days:
                        midpoint = all_dates[i] + pd.Timedelta(days=gap//2)
                        candidate_dates.append(midpoint)
            
            candidate_dates.append(all_dates[0] - pd.Timedelta(days=gap_days))
            candidate_dates.append(all_dates[-1] + pd.Timedelta(days=gap_days))
            
            for cand_date in candidate_dates:
                if total_cleans_found >= total_habs_found: break
                
                too_close = any(abs((cand_date - d).days) <= ((gap_days // 2) - 1) for d in all_dates)
                if too_close: continue
                
                logging.info(f"  [Impute Search] Cluster {cid} | Candidate Date: {cand_date.date()}")
                
                # --- PASS 1 FOR IMPUTED CLEANS ---
                items = search_stac(c_lat, c_lon, cand_date, y_days1)
                found_item = False
                for item in items:
                    if check_scl_water(item, c_lat, c_lon):
                        logging.info(f"    -> [P1 SUCCESS] Created 'potential No' item: {item.id}")
                        assigned_priority = 1
                        assigned_id = item.id
                        found_item = True
                        break
                        
                # --- PASS 2 FOR IMPUTED CLEANS ---
                if not found_item:
                    items = search_stac(c_lat, c_lon, cand_date, y_days2)
                    for item in items:
                        if check_scl_water(item, c_lat, c_lon):
                            logging.info(f"    -> [P2 SUCCESS] Created 'potential No' item: {item.id}")
                            assigned_priority = 2
                            assigned_id = item.id
                            found_item = True
                            break
                            
                if found_item:
                    synth_row = seed_row.copy()
                    synth_row['Event_Date_Parsed'] = cand_date
                    synth_row[date_col] = cand_date.strftime('%Y-%m-%d')
                    synth_row['is_HAB'] = 'potential No'
                    synth_row['sat_item'] = assigned_id
                    synth_row['priority'] = assigned_priority
                    synth_row['suppressed_by'] = 'Imputed_Phase3'
                    synth_row[id_col] = f"Imputed_{len(df) + len(new_synthetic_rows)}"
                    
                    new_synthetic_rows.append(synth_row.to_dict())
                    total_cleans_found += 1
                    break

        if new_synthetic_rows:
            df = pd.concat([df, pd.DataFrame(new_synthetic_rows)], ignore_index=True)
            logging.info(f"\nPhase 3 Complete: Added {len(new_synthetic_rows)} synthetic 'potential No' cases.")

    # ==========================================
    # FINAL CLEANUP & SAVE
    # ==========================================
    df.drop(columns=['Event_Date_Parsed', 'qty_numeric'], inplace=True, errors='ignore')
    final_file = os.path.join(output_dir, f"{base_name}_priority_FINAL.csv")
    df.to_csv(final_file, index=False)
    
    logging.info(f"\n{'='*60}\nPIPELINE COMPLETE!\nTotal HABs: {total_habs_found}\nTotal Cleans: {total_cleans_found}\nOutput: {final_file}\n{'='*60}")

n_cases, x_days, y_days1, y_days2, purity_days, gap_days = 3, 14, 5, 15, 10, 120
#run_decoupled_prescreener(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Iains dataset\pre_processed_v3.csv", r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Iains dataset", x_days, y_days1, y_days2, purity_days, gap_days, n_cases)
run_decoupled_prescreener(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Sweden\metadata_2021_v3.csv", r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Sweden", x_days, y_days1, y_days2, purity_days, gap_days, n_cases)
