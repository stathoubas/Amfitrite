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
    log_file = os.path.join(output_root, f"prescreen_decoupled_{base_name}.log")
    
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

def run_decoupled_prescreener(csv_path, output_dir, x_days, y_days, purity_days, gap_days, n_cases):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    setup_logger(output_dir, base_name)
    
    logging.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False) if csv_path.endswith('.csv') else pd.read_excel(csv_path)
    
    # 1. Standardize Columns
    dec_cols = [c for c in df.columns if 'decima' in c.lower()]
    if len(dec_cols) >= 2:
        lat_col, lon_col = dec_cols[0], dec_cols[1]
    else:
        lat_col, lon_col = 'decimalLatitude', 'decimalLongitude'
    
    id_col = 'id_x' if 'id_x' in df.columns else 'HAB_ID'
    if id_col not in df.columns:
        df['generated_id'] = [f"Row_{i}" for i in range(len(df))]
        id_col = 'generated_id'

    qty_col = next((c for c in df.columns if 'organismQ' in c or 'eMoF_measure' in c or 'individual' in c.lower()), None)

    # 2. Parse Dates
    initial_len = len(df)
    df['Event_Date_Parsed'] = pd.to_datetime(df['eventDate'], errors='coerce').dt.normalize()
    df = df.dropna(subset=['Event_Date_Parsed']).reset_index(drop=True)
    if len(df) < initial_len:
        logging.info(f"Dropped {initial_len - len(df)} rows due to missing dates (NaT).")

    df['qty_numeric'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
    
    if 'is_HAB' not in df.columns:
        logging.info("Column 'is_HAB' not found. Generating based on qty_numeric thresholds...")
        df['is_HAB'] = 'Check'
        df.loc[df['qty_numeric'] >= 100000, 'is_HAB'] = 'Yes'
        df.loc[df['qty_numeric'] <= 10000, 'is_HAB'] = 'No'

    for col in ['sat_item', 'suppressed_by', 'removed_by_ID']:
        if col not in df.columns: df[col] = ""
    if 'cluster_ID' not in df.columns: df['cluster_ID'] = np.nan
    
    cluster_id_counter = 1
    total_habs_found = 0
    total_cleans_found = 0

    # ==========================================
    # PHASE 1: PROCESS ALL HABs
    # ==========================================
    logging.info(f"\n{'='*60}\nPHASE 1: HUNTING FOR HABs (Max {n_cases} per cluster)\n{'='*60}")
    
    hab_mask = (df['is_HAB'].str.strip().str.upper() == 'YES')
    df_habs = df[hab_mask].sort_values(by='qty_numeric', ascending=False)
    
    for seed_idx, seed_row in df_habs.iterrows():
        if pd.notna(df.at[seed_idx, 'cluster_ID']) or not is_unsuppressed(df.at[seed_idx, 'suppressed_by']): continue
        
        seed_lat, seed_lon = seed_row[lat_col], seed_row[lon_col]
        seed_id = seed_row[id_col]
        
        logging.info(f"\n[HAB CLUSTER {cluster_id_counter}] Seed: {seed_id} | Qty: {seed_row['qty_numeric']} | Date: {seed_row['Event_Date_Parsed'].date()}")
        
        dist_mask = haversine_distance(seed_lat, seed_lon, df[lat_col], df[lon_col]) <= 7680
        unclustered_mask = df['cluster_ID'].isna()
        cluster_mask = dist_mask & unclustered_mask
        df.loc[cluster_mask, 'cluster_ID'] = cluster_id_counter
        cluster_indices = df[cluster_mask].index
        
        cluster_habs = df.loc[cluster_indices][df.loc[cluster_indices, 'is_HAB'].str.strip().str.upper() == 'YES'].sort_values(by='qty_numeric', ascending=False)
        habs_in_this_cluster = 0
        
        for cand_idx, cand_row in cluster_habs.iterrows():
            if habs_in_this_cluster >= n_cases: break
            if not is_unsuppressed(df.at[cand_idx, 'suppressed_by']): continue
            if df.at[cand_idx, 'sat_item'] != "": continue # THE GHOST FIX
            
            cand_date = cand_row['Event_Date_Parsed']
            items = search_stac(seed_lat, seed_lon, cand_date, y_days)
            for item in items:
                if check_scl_water(item, seed_lat, seed_lon):
                    logging.info(f"  -> [SUCCESS] Found HAB item: {item.id}")
                    df.at[cand_idx, 'sat_item'] = item.id
                    habs_in_this_cluster += 1
                    total_habs_found += 1
                    
                    time_mask = abs(df.loc[cluster_indices, 'Event_Date_Parsed'] - cand_date).dt.days <= x_days
                    suppress_mask = time_mask & (df.loc[cluster_indices, 'sat_item'] == "") 
                    df.loc[cluster_indices[suppress_mask], 'suppressed_by'] = f"HAB_{cand_row[id_col]}"
                    break
                    
        leftover_habs = (df.loc[cluster_indices, 'is_HAB'].str.strip().str.upper() == 'YES') & (df.loc[cluster_indices, 'sat_item'] == "") & (df.loc[cluster_indices, 'suppressed_by'] == "")
        df.loc[cluster_indices[leftover_habs], 'suppressed_by'] = f"HAB_Cleanup_{cluster_id_counter}"
        
        cluster_id_counter += 1

    logging.info(f"\nPHASE 1 COMPLETE. Total HABs secured: {total_habs_found}")

    # ==========================================
    # PHASE 2: HUNT EXISTING CLEANS
    # ==========================================
    logging.info(f"\n{'='*60}\nPHASE 2: HUNTING FOR EXISTING CLEANS\n{'='*60}")
    
    clean_mask = (df['is_HAB'].str.strip().str.upper() == 'NO')
    df_cleans = df[clean_mask].sort_values(by='qty_numeric', ascending=True)
    
    for seed_idx, seed_row in df_cleans.iterrows():
        if total_cleans_found >= total_habs_found: break
        if not is_unsuppressed(df.at[seed_idx, 'suppressed_by']): continue
        if df.at[seed_idx, 'sat_item'] != "": continue # THE GHOST FIX - OUTER LOOP
        
        seed_lat, seed_lon = seed_row[lat_col], seed_row[lon_col]
        seed_id = seed_row[id_col]
        
        if pd.isna(df.at[seed_idx, 'cluster_ID']):
            dist_mask = haversine_distance(seed_lat, seed_lon, df[lat_col], df[lon_col]) <= 7680
            unclustered_mask = df['cluster_ID'].isna()
            cluster_mask = dist_mask & unclustered_mask
            df.loc[cluster_mask, 'cluster_ID'] = cluster_id_counter
            current_cluster_id = cluster_id_counter
            cluster_id_counter += 1
        else:
            current_cluster_id = df.at[seed_idx, 'cluster_ID']
            cluster_mask = (df['cluster_ID'] == current_cluster_id)
            
        cluster_indices = df[cluster_mask].index
        cluster_cleans = df.loc[cluster_indices][df.loc[cluster_indices, 'is_HAB'].str.strip().str.upper() == 'NO'].sort_values(by='qty_numeric', ascending=True)
        cleans_in_this_cluster = 0
        
        for cand_idx, cand_row in cluster_cleans.iterrows():
            if cleans_in_this_cluster >= n_cases or total_cleans_found >= total_habs_found: break
            if not is_unsuppressed(df.at[cand_idx, 'suppressed_by']): continue
            if df.at[cand_idx, 'sat_item'] != "": continue # THE GHOST FIX - INNER LOOP
            
            cand_date = cand_row['Event_Date_Parsed']
            
            purity_time_mask = abs(df.loc[cluster_indices, 'Event_Date_Parsed'] - cand_date).dt.days <= purity_days
            contaminants = df.loc[cluster_indices][
                purity_time_mask & 
                ((df.loc[cluster_indices, 'is_HAB'].str.strip().str.upper() == 'YES') | 
                 (df.loc[cluster_indices, 'qty_numeric'] > 10000))
            ]
            
            if not contaminants.empty: continue
                
            items = search_stac(seed_lat, seed_lon, cand_date, y_days)
            for item in items:
                if check_scl_water(item, seed_lat, seed_lon):
                    logging.info(f"  -> [SUCCESS] Found Clean item: {item.id}")
                    df.at[cand_idx, 'sat_item'] = item.id
                    cleans_in_this_cluster += 1
                    total_cleans_found += 1
                    
                    time_mask = abs(df.loc[cluster_indices, 'Event_Date_Parsed'] - cand_date).dt.days <= x_days
                    suppress_mask = time_mask & (df.loc[cluster_indices, 'sat_item'] == "")
                    df.loc[cluster_indices[suppress_mask], 'suppressed_by'] = f"Clean_{cand_row[id_col]}"
                    break
                    
        leftover_cleans = (df.loc[cluster_indices, 'is_HAB'].str.strip().str.upper() == 'NO') & (df.loc[cluster_indices, 'sat_item'] == "") & (df.loc[cluster_indices, 'suppressed_by'] == "")
        df.loc[cluster_indices[leftover_cleans], 'suppressed_by'] = f"Clean_Cleanup_{current_cluster_id}"

    logging.info(f"\nPHASE 2 COMPLETE. Existing Cleans secured: {total_cleans_found}")

    # ==========================================
    # PHASE 3: IMPUTING POTENTIAL NO CASES
    # ==========================================
    if total_cleans_found < total_habs_found:
        logging.info(f"\n{'='*60}\nPHASE 3: IMPUTING 'potential No' CASES (Need {total_habs_found - total_cleans_found} more)\n{'='*60}")
        
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
                items = search_stac(c_lat, c_lon, cand_date, y_days)
                
                for item in items:
                    if check_scl_water(item, c_lat, c_lon):
                        logging.info(f"    -> [SUCCESS] Created 'potential No' item: {item.id}")
                        
                        synth_row = seed_row.copy()
                        synth_row['Event_Date_Parsed'] = cand_date
                        synth_row['eventDate'] = cand_date.strftime('%Y-%m-%d')
                        synth_row['is_HAB'] = 'potential No'
                        synth_row['sat_item'] = item.id
                        synth_row['qty_numeric'] = 0
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
    final_file = os.path.join(output_dir, f"{base_name}_decoupled_FINAL_v2.csv")
    df.to_csv(final_file, index=False)
    
    logging.info(f"\n{'='*60}\nPIPELINE COMPLETE!\nTotal HABs: {total_habs_found}\nTotal Cleans: {total_cleans_found}\nOutput: {final_file}\n{'='*60}")


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Phase 1 API Pre-Screener (With Temporal Imputation)")
    #parser.add_argument("--input", type=str, required=True, help="Input CSV/Excel file")
    #parser.add_argument("--output", type=str, required=True, help="Output directory")
    #args = parser.parse_args()
    #
    #try:
    #    n_cases = int(input("Enter 'N' (Max cases per cluster, e.g., 2): "))
    #    x_days = int(input("Enter 'X' (Temporal suppression window, e.g., 14): "))
    #    y_days = int(input("Enter 'Y' (STAC search window, e.g., 5): "))
    #    purity_days = int(input("Enter 'Purity Window' (Strictly clean days for Phase 2, e.g., 10): "))
    #    gap_days = int(input("Enter 'Gap Threshold' (Min gap for Phase 3 imputation, e.g., 120): "))
    #except ValueError:
    #    print("Invalid input. Defaulting to N=2, X=14, Y=5, Purity=10, Gap=120.")
    #    n_cases, x_days, y_days, purity_days, gap_days = 2, 14, 5, 10, 120
       
    #run_decoupled_prescreener(args.input, args.output, x_days, y_days, purity_days, gap_days, n_cases)
    
    n_cases, x_days, y_days, purity_days, gap_days = 3, 14, 5, 10, 120
    run_decoupled_prescreener(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS dataset v2\datasets_1_3_4_6_v5.csv", r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS dataset v2", x_days, y_days, purity_days, gap_days, n_cases)