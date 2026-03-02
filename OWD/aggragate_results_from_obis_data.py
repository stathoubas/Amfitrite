# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:44:10 2026

@author: K. Pikounis
"""

import os
import pandas as pd
import numpy as np

def strip_images_from_excel(summary_excel_path):
    """
    Reads a heavy Excel file (with embedded images), ignores the images,
    and saves a lightweight '_v2.xlsx' version. Returns the clean DataFrame.
    """
    if not os.path.exists(summary_excel_path):
        return None
        
    try:
        df = pd.read_excel(summary_excel_path)
        v2_path = summary_excel_path.replace(".xlsx", "_v2.xlsx")
        df.to_excel(v2_path, index=False)
        
        return df
    except Exception as e:
        print(f"Error stripping images from {summary_excel_path}: {e}")
        return None

def aggregate_pipeline_results(input_csv_path_name, output_root, final_output_csv_name):
    """
    Reads the initial input CSV, finds the processed folders, applies validity filters,
    calculates the Confidence Index, and classifies the tiles.
    """
    print(f"--- Starting Aggregation for {input_csv_path_name} ---")
    
    # 1. Read the initial CSV
    try:
        df_main = pd.read_csv(input_csv_path_name) if input_csv_path_name.endswith('.csv') else pd.read_excel(input_csv_path_name)
    except Exception as e:
        print(f"Failed to read input file {input_csv_path_name}. Error: {e}")
        return
        
    # Prepare new columns
    df_main["Status"] = ""
    df_main["Overall_Class"] = ""
    df_main["Max_HAB_Index"] = np.nan
    df_main["HAB_Tiles_Sorted"] = ""
    df_main["Max_Check_Index"] = np.nan
    df_main["Check_Tiles_Sorted"] = ""
    df_main["Index_per_Tile"] = np.nan
    
    # Get all folders currently in the output root
    existing_folders = os.listdir(output_root) if os.path.exists(output_root) else []
    
    # 2. Iterate through each case
    for idx, row in df_main.iterrows():
        raw_id = str(row.get('id_x', f"Row_{idx}"))
        case_id = raw_id.replace(":", "_")
        
        # Find matching folder
        matched_folders = [f for f in existing_folders if case_id in f]
        
        if not matched_folders:
            df_main.at[idx, "Status"] = "No folder found"
            continue
            
        if not len(matched_folders) > 1:
            df_main.at[idx, "Status"] = "multiple folders"
            continue
        
        target_folder = os.path.join(output_root, matched_folders[0])
        summary_path = os.path.join(target_folder, "folder_summary.xlsx")
        
        # 3. Strip images and load summary dataframe
        df_tiles = strip_images_from_excel(summary_path)
        
        if df_tiles is None or df_tiles.empty:
            df_main.at[idx, "Status"] = "Summary excel missing or empty"
            continue
            
        # 4. Apply Validity Filters
        # Total_pixels == 65536 AND SCL_Water_Pixels >= 32000
        valid_tiles = df_tiles[
            (df_tiles['Total_pixels'] == 65536) & 
            (df_tiles['SCL_Water_Pixels'] >= 32000)
        ].copy()
        
        if valid_tiles.empty:
            df_main.at[idx, "Status"] = "Filters removed all tiles"
            continue
            
        # 5. Evaluate remaining valid tiles
        hab_list = []
        check_list = []
        indices_list = []
        clean_count = 0
        
        for _, tile in valid_tiles.iterrows():
            tile_name = tile['Tile_Folder']
            
            # Count CNN YES votes
            cnn_cols = ['rdnet_no_scl', 'convnext_scl', 'res18_scl', 'res18_no_scl']
            yes_count = sum(1 for col in cnn_cols if str(tile.get(col, '')).strip().upper() == 'YES')
            
            # Calculate CNN Score
            cnn_score = 0
            if str(tile.get('rdnet_no_scl', '')).strip().upper() == 'YES': cnn_score += 30
            if str(tile.get('convnext_scl', '')).strip().upper() == 'YES': cnn_score += 10
            if str(tile.get('res18_scl', '')).strip().upper() == 'YES': cnn_score += 10
            if str(tile.get('res18_no_scl', '')).strip().upper() == 'YES': cnn_score += 10
            
            # Calculate CyFi Score
            cyfi_high = float(tile.get('CyFi_High', 0))
            cyfi_mod = float(tile.get('CyFi_Moderate', 0))
            
            cyfi_mass = cyfi_high + (0.5 * cyfi_mod)
            cyfi_score = (cyfi_mass / 676.0) * 40.0
            
            # Total Index
            total_index = round(cnn_score + cyfi_score, 2)
            
            # Indices list
            indices_list.append((tile_name, total_index))
            
            # Classification Logic
            rdnet_is_yes = str(tile.get('rdnet_no_scl', '')).strip().upper() == 'YES'
            
            is_hab = rdnet_is_yes and (yes_count >= 2) and (cyfi_high >= 20)
            is_clean = (not rdnet_is_yes) and (yes_count <= 1) and (cyfi_high < 5) and (cyfi_mod < 15)
            
            # Append to appropriate lists
            if is_hab:
                hab_list.append((tile_name, total_index))
            elif is_clean:
                clean_count += 1
            else:
                check_list.append((tile_name, total_index))
                
        # 6. Sort lists and aggregate to Main Dataframe
        # Sort descending by index
        hab_list.sort(key=lambda x: x[1], reverse=True)
        check_list.sort(key=lambda x: x[1], reverse=True)
        
        df_main.at[idx, "Status"] = "Processed successfully"
        
        # Determine overall case class
        if len(hab_list) > 0:
            df_main.at[idx, "Overall_Class"] = "HAB Detected"
        elif len(check_list) > 0:
            df_main.at[idx, "Overall_Class"] = "To Be Checked"
        elif clean_count > 0:
            df_main.at[idx, "Overall_Class"] = "Clean"
        else:
            df_main.at[idx, "Overall_Class"] = "Unknown" # Fallback
            
        # Format string lists and max index
        if hab_list:
            df_main.at[idx, "Max_HAB_Index"] = hab_list[0][1]
            df_main.at[idx, "HAB_Tiles_Sorted"] = " | ".join([f"{name} ({idx})" for name, idx in hab_list])
            
        if check_list:
            df_main.at[idx, "Max_Check_Index"] = check_list[0][1]
            df_main.at[idx, "Check_Tiles_Sorted"] = " | ".join([f"{name} ({idx})" for name, idx in check_list])
            
        if indices_list:
            df_main.at[idx, "Index_per_Tile"] = " | ".join([f"{name} ({idx})" for name, idx in indices_list])


    # 7. Save final output
    out_path = os.path.join(output_root, final_output_csv_name)
    df_main.to_csv(out_path, index=False)
    print(f"--- Aggregation Complete! Saved to: {out_path} ---")
    
    return df_main

if __name__ == "__main__":

    df_final = aggregate_pipeline_results("/vol2/Amfitrite/OWD/input_data/datasets_1_3_4_6_v5_part_1.csv", "/vol2/Amfitrite/OWD/obis_data/part1", "/vol2/Amfitrite/OWD/obis_data/aggregated_data_part1.csv")
    df_final = aggregate_pipeline_results("/vol2/Amfitrite/OWD/input_data/datasets_1_3_4_6_v5_part_2.csv", "/vol2/Amfitrite/OWD/obis_data/part2", "/vol2/Amfitrite/OWD/obis_data/aggregated_data_part2.csv")
    df_final = aggregate_pipeline_results("/vol2/Amfitrite/OWD/input_data/datasets_1_3_4_6_v5_part_3.csv", "/vol2/Amfitrite/OWD/obis_data/part3", "/vol2/Amfitrite/OWD/obis_data/aggregated_data_part3.csv")
    df_final = aggregate_pipeline_results("/vol2/Amfitrite/OWD/input_data/datasets_1_3_4_6_v5_part_4.csv", "/vol2/Amfitrite/OWD/obis_data/part4", "/vol2/Amfitrite/OWD/obis_data/aggregated_data_part4.csv")