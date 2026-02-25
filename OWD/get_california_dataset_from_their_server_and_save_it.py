# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 16:18:24 2026

@author: K. Pikounis
"""

import pandas as pd
import urllib.error
import os

# The major CalHABMAP dataset IDs on the SCCOOS ERDDAP server
calhabmap_datasets = [
    'HABs-ScrippsPier', 'HABs-SantaMonicaPier', 'HABs-NewportBeachPier', 
    'HABs-StearnsWharf', 'HABs-MontereyWharf', 'HABs-SantaCruzWharf', 
    'HABs-BodegaMarineLab', 'HABs-TrinidadPier'
]

# ERDDAP API endpoint for CSV without the messy units row (.csvp) starting from 2015
base_url = "https://erddap.sccoos.org/erddap/tabledap/{}.csvp?&time>=2015-01-01"

dfs = []

for dataset_id in calhabmap_datasets:
    url = base_url.format(dataset_id)
    print(f"Fetching data from {dataset_id}...")
    try:
        # Read directly from the ERDDAP API
        df = pd.read_csv(url)
        df['station_id'] = dataset_id # Add a column to track which pier it came from
        dfs.append(df)
    except Exception as e:
        print(f"Could not fetch {dataset_id} (it might be temporarily down): {e}")

if dfs:
    # Combine all pier data into one dataframe
    cal_df = pd.concat(dfs, ignore_index=True)
    
    # 1. HOMOGENIZE CORE COLUMNS
    # The API returns columns with names like 'time (UTC)'. We find them dynamically.
    rename_map = {}
    for col in cal_df.columns:
        col_lower = col.lower()
        if 'time' in col_lower:
            rename_map[col] = 'eventDate'
        elif 'latitude' in col_lower:
            rename_map[col] = 'decimalLatitude'
        elif 'longitude' in col_lower:
            rename_map[col] = 'decimalLongitude'
        elif 'depth' in col_lower:
            rename_map[col] = 'minimumDepthInMeters'

    cal_df.rename(columns=rename_map, inplace=True)
    
    # 2. MELT SPECIES COLUMNS TO 'LONG' FORMAT
    # Find all columns containing cell counts (they usually have 'cells' or 'abundance' in the name)
    species_cols = [col for col in cal_df.columns if 'cells' in col.lower() or 'abundance' in col.lower()]
    
    # Identify the "extra" columns to keep (temp, salinity, eventDate, lat, lon, etc.)
    id_vars = [col for col in cal_df.columns if col not in species_cols]
    
    print("Reshaping the dataset from Wide to Long format...")
    # 'Melt' turns the separate species columns into rows, keeping the extra columns intact
    melted_df = pd.melt(cal_df, id_vars=id_vars, value_vars=species_cols, 
                        var_name='Original_Species_Column', value_name='organismQuantity')
    
    # Drop rows where cell count is NaN (meaning that specific species wasn't observed on that date)
    melted_df.dropna(subset=['organismQuantity'], inplace=True)
    
    # 3. CLEAN UP SCIENTIFIC NAME AND UNITS
    # Convert something like 'Pseudo_nitzschia_delicatissima_group_cells_L' to 'Pseudo-nitzschia delicatissima group'
    melted_df['scientificName'] = melted_df['Original_Species_Column'].str.replace(r'_?cells_L.*', '', regex=True, case=False)
    melted_df['scientificName'] = melted_df['scientificName'].str.replace('_', ' ')
    
    # The data is inherently in cells per litre
    melted_df['organismQuantityType'] = 'cells per litre'
    
    # 4. SAVE (Without appending to anything else)
    output_file = 'homogenized_calhabmap.csv'
    melted_df.to_csv(os.path.join(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\California", output_file), index=False)
    print(f"Success! Saved {len(melted_df)} rows to {output_file} with all extra columns preserved.")
else:
    print("No data was fetched. Check your internet connection or ERDDAP server status.")