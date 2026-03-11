# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:21:38 2026

@author: K. Pikounis
"""

import pandas as pd
import argparse
import os

def analyze_florida_output(csv_path):
    print(f"Loading dataset: {csv_path}")
    try:
        df = pd.read_csv(csv_path, low_memory=False) if csv_path.endswith('.csv') else pd.read_excel(csv_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 1. Filter: Only keep rows where sat_item is actually populated
    if 'sat_item' not in df.columns:
        print("Error: 'sat_item' column not found in the dataset.")
        return
        
    df_valid = df[df['output_folder'].notna()].copy()
    df_valid = df_valid[df_valid["status"].astype(str).str.upper() == "SUCCESS"]
    df_valid = df_valid[df_valid['output_folder'].astype(str).str.strip() != '']
    df_valid = df_valid[df_valid['output_folder'].astype(str).str.lower() != 'nan']
    df_valid = df_valid[df_valid["Overall_Class"].astype(str).str.lower() != "filters removed all tiles"]

    # 2. Safely locate Lat/Lon columns (handling Excel truncation like 'decima')
    dec_cols = [c for c in df_valid.columns if 'decima' in c.lower()]
    if len(dec_cols) >= 2:
        lat_col, lon_col = dec_cols[0], dec_cols[1]
    else:
        lat_col = next((c for c in df_valid.columns if 'lat' in c.lower()), 'decimalLatitude')
        lon_col = next((c for c in df_valid.columns if 'lon' in c.lower()), 'decimalLongitude')

    # 3. Separate into HABs and Non-HABs
    is_hab_series = df_valid['is_HAB'].astype(str).str.strip().str.upper()
    hab_df = df_valid[is_hab_series == 'YES']
    #non_hab_df = df_valid[is_hab_series == 'NO']
    non_hab_df = df_valid[is_hab_series == 'POTENTIAL NO']

    # 4. Define a helper function to print the stats
    def print_stats(subset, name):
        total = len(subset)
        if total == 0:
            print(f"\n{name} 0")
            return
            
        # Group by exact latitude and longitude to find unique locations
        location_counts = subset.groupby([lat_col, lon_col]).size()
        unique_locations = len(location_counts)
        
        # Count how many locations had 1 event, 2 events, etc.
        multiplicity = location_counts.value_counts().sort_index()
        
        print(f"\n{name} {total}")
        print(f"{name} unique locations {unique_locations}")
        print(f"{name} multiplicity:")
        
        # Format as "1: 150, 2: 20, 3: 10..."
        mult_str = ", ".join([f"{k}: {v}" for k, v in multiplicity.items()])
        print(mult_str)

    # 5. Execute and print
    print("\n" + "="*40)
    print("FLORIDA DATASET YIELD ANALYSIS")
    print("="*40)
    
    print_stats(hab_df, "HABs")
    print("-" * 40)
    print_stats(non_hab_df, "Non-HABs")
    print("="*40 + "\n")

#analyze_florida_output(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Florida\Historic_Harmful_Algal_Bloom_Events_2015_-_2023_homogenised_v2_prescreened_FINAL_with_results.csv")
#analyze_florida_output(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\North American Arctic region\homogenized_arctic_plankton_v1_decoupled_FINAL_v2_with_results.csv")
#analyze_florida_output(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS dataset v2\datasets_1_3_4_6_v5_decoupled_FINAL_v2_with_results.csv")
#analyze_florida_output(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Iains dataset\pre_processed_v3_priority_FINAL_with_results.csv")
analyze_florida_output(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Sweden\metadata_2021_v3_priority_FINAL_with_results.csv")