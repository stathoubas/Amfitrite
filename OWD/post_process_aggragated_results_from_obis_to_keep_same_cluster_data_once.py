# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:16:40 2026

@author: K. Pikounis
"""

import pandas as pd
import numpy as np

def assign_event_ids(input_csv, output_csv):
    """
    Reads the HAB dataset, clusters events based on exact Lat/Lon and dates 
    within +/- 7 days, and assigns a unique ascending 'event_cluster_id'.
    """
    print(f"Loading data from {input_csv}...")
    
    try:
        df = pd.read_csv(input_csv) if input_csv.endswith('.csv') else pd.read_excel(input_csv)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
        
    # 1. Safely parse the dates
    df['parsed_date'] = pd.to_datetime(df['eventDate'], errors='coerce')
    
    # 2. Keep track of original order so we don't mess up your CSV layout
    df['original_index'] = df.index
    
    # 3. Sort by Location and Date to bring similar events next to each other
    df_sorted = df.sort_values(by=['decimalLatitude', 'decimalLongitude', 'parsed_date'])
    
    event_ids = []
    current_id = 0
    
    prev_lat = None
    prev_lon = None
    prev_date = pd.NaT

    print("Clustering events within +/- 7 days...")
    # 4. Loop through and assign IDs
    for idx, row in df_sorted.iterrows():
        lat = row['decimalLatitude']
        lon = row['decimalLongitude']
        date = row['parsed_date']
        
        # Check if coordinates match exactly
        is_same_location = (lat == prev_lat) and (lon == prev_lon)
        
        # Check if the date is within 7 days
        is_same_time = False
        if pd.notna(date) and pd.notna(prev_date):
            time_diff = abs((date - prev_date).days)
            if time_diff <= 7:
                is_same_time = True
                
        # Assign IDs
        if is_same_location and is_same_time:
            # Belongs to the same cluster
            event_ids.append(current_id)
        else:
            # New cluster
            current_id += 1
            event_ids.append(current_id)
            
        # Update trackers
        prev_lat = lat
        prev_lon = lon
        prev_date = date

    # Add the new column
    df_sorted['event_cluster_id'] = event_ids
    
    # 5. Restore the original CSV order
    df_final = df_sorted.sort_values(by='original_index').drop(columns=['original_index', 'parsed_date'])
    
    # Move the new ID column to the very front for easy viewing
    cols = ['event_cluster_id'] + [col for col in df_final.columns if col != 'event_cluster_id']
    df_final = df_final[cols]
    
    # 6. Save
    df_final.to_csv(output_csv, index=False)
    
    # Print a quick summary
    total_rows = len(df_final)
    unique_events = df_final['event_cluster_id'].nunique()
    print(f"Done! Reduced {total_rows} reports into {unique_events} unique HAB events.")
    print(f"Saved grouped dataset to: {output_csv}")
    
    return df_final


def merge_duplicate_hab_events(input_file, output_file):
    """
    Groups the final aggregated dataset by Lat, Lon, and Date.
    Checks if pipeline results are identical. If so, collapses them into one row
    and extracts the max organism quantity and a list of all detected species.
    """
    print(f"--- Loading Final Dataset: {input_file} ---")
    
    try:
        df = pd.read_csv(input_file) if input_file.endswith('.csv') else pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
        
    # The columns that contain the pipeline's output (adjust if your names differ slightly)
    result_cols = [
        'Aggregation_Status', 'Overall_Class', 'Max_HAB_Index', 
        'HAB_Tiles_Sorted', 'Max_Check_Index', 'Check_Tiles_Sorted'
    ]
    
    # Ensure result columns exist in the dataframe to avoid KeyErrors
    result_cols = [col for col in result_cols if col in df.columns]

    # Group by Location and Date
    grouped = df.groupby(['decimalLatitude', 'decimalLongitude', 'eventDate'], dropna=False)
    
    final_rows = []
    
    print("Processing groups...")
    for (lat, lon, date), group in grouped:
        
        # 1. Check if the pipeline results are identical across the group
        is_identical = True
        if len(group) > 1:
            for col in result_cols:
                # nunique(dropna=False) counts NaN as a value. If it's > 1, the rows differ.
                if group[col].nunique(dropna=False) > 1:
                    is_identical = False
                    break
        
        # -> If they are NOT identical, print the id_x and keep original rows unmerged
        if not is_identical:
            conflicting_ids = group['id_x'].tolist()
            print(f"WARNING: Results are NOT identical for group (Lat: {lat}, Lon: {lon}, Date: {date}).")
            print(f"         IDs involved: {conflicting_ids}")
            # Append them as-is without merging
            for _, r in group.iterrows():
                final_rows.append(r.to_dict())
            continue
            
        # -> If they ARE identical (or it's just a single row), merge them!
        
        # Safely extract numeric quantities to find the max
        # This strips out text like "present" or "~" so we can do math, replacing them with -1
        qty_numeric = pd.to_numeric(
            group['organismQuantity'].astype(str).str.replace(r'[^0-9.]', '', regex=True), 
            errors='coerce'
        ).fillna(-1)
        
        # Find the row index of the maximum quantity
        max_idx = qty_numeric.idxmax()
        
        max_qty = group.loc[max_idx, 'organismQuantity']
        max_species = group.loc[max_idx, 'scientificName']
        
        # Build the list of all species and their quantities
        all_pairs = []
        for _, r in group.iterrows():
            species = str(r.get('scientificName', 'Unknown'))
            qty = str(r.get('organismQuantity', 'Unknown'))
            all_pairs.append(f"{species}: {qty}")
            
        all_pairs_str = " | ".join(all_pairs)
        
        # Take the first row as the template for the collapsed group
        merged_row = group.iloc[0].to_dict()
        
        # Add the new requested columns
        merged_row['Max_Organism_Species'] = max_species
        merged_row['Max_Organism_Quantity'] = max_qty
        merged_row['All_Organisms_List'] = all_pairs_str
        
        final_rows.append(merged_row)
        
    # Build the final DataFrame
    df_merged = pd.DataFrame(final_rows)
    
    # Reorder columns to put the new organism info near the front/results for easy reading
    new_cols = ['Max_Organism_Species', 'Max_Organism_Quantity', 'All_Organisms_List']
    existing_cols = [col for col in df_merged.columns if col not in new_cols]
    
    # You can customize exactly where you want these columns. 
    # For now, we put them right after the scientificName/organismQuantity columns if they exist.
    df_merged = df_merged[existing_cols + new_cols]
    
    # Save the output
    df_merged.to_csv(output_file, index=False)
    
    print("\n--- Summary ---")
    print(f"Original rows: {len(df)}")
    print(f"Final merged rows: {len(df_merged)}")
    print(f"Saved to: {output_file}")
    
    return df_merged


#df = assign_event_ids(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS datasets\aggregated_data.csv", r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS datasets\aggregated_data_clustered.csv")
df = merge_duplicate_hab_events(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS datasets\aggregated_data.csv", r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS datasets\aggregated_data_clustered_v2.csv")