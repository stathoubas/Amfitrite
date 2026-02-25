# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 17:56:53 2026

@author: K. Pikounis
"""

import pandas as pd
import os

# Define file names
root_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Harmful Algal Event Database (HAEDAT)"
file_event = 'event.txt'
file_occ = 'occurrence.txt'
file_emof = 'extendedmeasurementorfact.txt'
output_summary = 'dataset_summary.csv'

def load_dwca_data():
    # 1. Read the Core File: Event
    # event.txt usually contains the location and date information
    # meta.xml indicates fields are tab-separated
    print("Loading Event data...")
    df_event = pd.read_csv(os.path.join(root_path, file_event), sep='\t', on_bad_lines='skip', low_memory=False)
    
    # 2. Read the Extension: Occurrence
    # This contains the species names and specific IDs
    print("Loading Occurrence data...")
    df_occ = pd.read_csv(os.path.join(root_path, file_occ), sep='\t', on_bad_lines='skip', low_memory=False)
    
    # 3. Read the Extension: ExtendedMeasurementOrFact (eMoF)
    # This contains measurements like toxicity levels
    print("Loading Measurement data...")
    df_emof = pd.read_csv(os.path.join(root_path, file_emof), sep='\t', on_bad_lines='skip', low_memory=False)

    # --- MERGING STRATEGY ---
    
    # Step A: Merge Event and Occurrence
    # In DwC-A, the extension (Occurrence) links to the Core (Event) via the 'id' column.
    # Usually, the Core file has an 'id' column, and the extension has a 'coreid' (often the first column).
    # We rename the index/id columns to facilitate a clean merge.
    
    # Check if 'id' exists in event, otherwise use the first column
    event_id_col = df_event.columns[0] 
    occ_coreid_col = df_occ.columns[0]
    
    print(f"Merging Event ({event_id_col}) and Occurrence ({occ_coreid_col})...")
    
    # We do a LEFT merge to keep all occurrences and attach their event details.
    # We verify column names to avoid duplication suffixes where possible.
    df_merged = pd.merge(
        df_occ, 
        df_event, 
        left_on=occ_coreid_col, 
        right_on=event_id_col, 
        how='left',
        suffixes=('_occ', '_event')
    )

    # Step B: Merge Measurements (eMoF)
    # Measurements in HAEDAT can be linked to the Event or the specific Occurrence.
    # Since eMoF has an 'occurrenceID' field (per meta.xml), we link on that 
    # to ensure the toxin measurement is mapped to the correct species.
    
    if 'occurrenceID' in df_emof.columns and 'occurrenceID' in df_merged.columns:
        print("Merging Measurements on occurrenceID...")
        # We rename columns in eMoF to avoid collisions (e.g., measurementType)
        df_emof_renamed = df_emof.rename(columns={
            'measurementType': 'eMoF_measurementType',
            'measurementValue': 'eMoF_measurementValue',
            'measurementUnit': 'eMoF_measurementUnit'
        })
        
        df_unified = pd.merge(
            df_merged,
            df_emof_renamed,
            on='occurrenceID',
            how='left'
        )
    else:
        # Fallback: If no occurrenceID link exists, we link on the Event ID (coreid)
        # Note: This might duplicate rows if multiple occurrences exist per event.
        print("Warning: occurrenceID not found for linking. Merging eMoF on Core ID.")
        emof_coreid_col = df_emof.columns[0]
        df_unified = pd.merge(
            df_merged,
            df_emof,
            left_on=occ_coreid_col, # Link to the Event ID found in the occurrence file
            right_on=emof_coreid_col,
            how='left',
            suffixes=('', '_emof')
        )

    return df_unified

# Execute
try:
    df_final = load_dwca_data()
    print(f"Successfully created unified dataframe with shape: {df_final.shape}")
    
    # Display first few rows to verify
    print(df_final.head())
    
    df_final.to_csv(os.path.join(root_path, "all_data.csv"), index = False)

    # --- CREATE SUMMARY ---
    print(f"Generating summary file: {output_summary}")
    
    summary_data = []
    for col in df_final.columns:
        summary_data.append({
            'Column Name': col,
            'Data Type': df_final[col].dtype,
            'Non-Null Count': df_final[col].count(),
            'Null Count': df_final[col].isna().sum(),
            'Unique Values': df_final[col].nunique(),
            'Sample Value': df_final[col].dropna().iloc[0] if df_final[col].count() > 0 else 'N/A'
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(root_path, output_summary), index=False)
    print("Done.")

except FileNotFoundError as e:
    print(f"Error: Missing file - {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    
