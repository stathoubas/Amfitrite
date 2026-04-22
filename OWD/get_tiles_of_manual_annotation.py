# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:05:39 2026

@author: K. Pikounis
"""
import pandas as pd
import json
import os

def normalize_path(original_path):
    """Automatically converts slashes to the correct OS format (Windows \ or Linux /)."""
    if pd.isna(original_path) or not str(original_path).strip():
        return ""
    # Replace any existing slashes with the native OS separator
    return str(original_path).strip().replace('\\', os.sep).replace('/', os.sep)

def compile_annotated_tiles(input_csv, output_csv, id_col, keep_cols):
    print(f"Loading annotated dataset: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)

    # 1) Filter rows where 'tiles_HABs' or 'tiles_nonHABs' are not empty
    df['tiles_HABs'] = df['tiles_HABs'].fillna('').astype(str).str.strip()
    df['tiles_nonHABs'] = df['tiles_nonHABs'].fillna('').astype(str).str.strip()
    
    valid_mask = (df['tiles_HABs'] != '') | (df['tiles_nonHABs'] != '')
    filtered_df = df[valid_mask].copy()
    print(f"Found {len(filtered_df)} manually annotated cases.")

    # 2) Check for duplicate IDs
    if not filtered_df[id_col].is_unique:
        print(f"[!] Warning: The ID column '{id_col}' contains duplicates. Dropping duplicate rows...")
        filtered_df = filtered_df.drop_duplicates(subset=[id_col])
    else:
        print(f"ID column '{id_col}' is unique.")

    compiled_data = []

    # 3) Iterate through the selection
    for idx, row in filtered_df.iterrows():
        # a) Parse tiles and assign tile_is_hab boolean
        tiles_to_process = []
        
        if row['tiles_HABs']:
            for t in row['tiles_HABs'].split(','):
                if t.strip():
                    tiles_to_process.append((t.strip(), True))
                    
        if row['tiles_nonHABs']:
            for t in row['tiles_nonHABs'].split(','):
                if t.strip():
                    tiles_to_process.append((t.strip(), False))

        # c) Get output folder and convert to OS-friendly path
        raw_out_folder = row.get('output_folder', '')
        local_base_folder = normalize_path(raw_out_folder)

        # d) i. Read the folder_summary.xlsx (Strictly)
        summary_path = os.path.join(local_base_folder, "folder_summary.xlsx")
        df_summary = pd.DataFrame()
        
        if os.path.exists(summary_path):
            try:
                df_summary = pd.read_excel(summary_path)
            except Exception as e:
                print(f"  [!] Error reading {summary_path}: {e}")
        else:
            print(f"  [!] Missing summary file at: {summary_path}")
                
        # Process each individual tile
        for tile_name, is_hab in tiles_to_process:
            # b) Extract the tile number (e.g., 'tile_0_H7_M12_L497' -> 0)
            try:
                tile_number = int(tile_name.split('_')[1])
            except (IndexError, ValueError):
                tile_number = -1 # Fallback if naming convention is unexpected
                
            # Initialize metrics
            tile_metrics = {
                "Total_pixels": None, "Total_CyFi_Points": None, "SCL_Water_Pixels": None,
                "CyFi_High": None, "CyFi_Moderate": None, "CyFi_Low": None,
                "res18_scl": None, "res18_no_scl": None, "convnext_scl": None, "rdnet_no_scl": None
            }
            
            # Fetch summary metrics for this specific tile
            if not df_summary.empty and 'Tile_Folder' in df_summary.columns:
                tile_row = df_summary[df_summary['Tile_Folder'] == tile_name]
                if not tile_row.empty:
                    t_r = tile_row.iloc[0]
                    for k in tile_metrics.keys():
                        if k in t_r:
                            tile_metrics[k] = t_r[k]

            # d) ii. Read metadata.json from the specific tile folder
            tile_folder_path = os.path.join(local_base_folder, tile_name)
            meta_json_path = os.path.join(tile_folder_path, "metadata.json")
            
            meta_data = {"center_lat": None, "center_lon": None, "date": None}
            if os.path.exists(meta_json_path):
                try:
                    with open(meta_json_path, 'r', encoding='utf-8') as jf:
                        jdata = json.load(jf)
                        for k in meta_data.keys():
                            meta_data[k] = jdata.get(k)
                except Exception as e:
                    print(f"  [!] Failed to parse JSON for {tile_name}: {e}")
            else:
                print(f"  [!] Missing JSON at: {meta_json_path}")

            # e) Assemble the final row data
            new_entry = {id_col: row[id_col]}
            
            # Add requested keep_cols
            for kc in keep_cols:
                new_entry[kc] = row.get(kc)
                
            # Add tile-specific info
            new_entry['tile_name'] = tile_name
            new_entry['tile_number'] = tile_number
            new_entry['tile_is_hab'] = is_hab
            new_entry["tile_folder_path"] = tile_folder_path
            
            # Add metrics and metadata
            new_entry.update(tile_metrics)
            new_entry.update(meta_data)
            
            compiled_data.append(new_entry)

    # Save to final CSV
    if compiled_data:
        final_df = pd.DataFrame(compiled_data)
        final_df.to_csv(output_csv, index=False)
        print(f"\nSuccess! Compiled {len(final_df)} total tiles into {output_csv}")
    else:
        print("\nNo valid tiles were extracted. Output CSV not created.")


if __name__ == "__main__":
    # -------------------------------------------------------------
    # 1. SET YOUR VARIABLES HERE
    # -------------------------------------------------------------
    INPUT_CSV   = "/vol2/Amfitrite/OWD/processed_results/datasets_1_3_4_6_v5_decoupled_FINAL_v2_v3_with_results_processed.csv"
    OUTPUT_CSV  = "/vol2/Amfitrite/OWD/processed_results/datasets_1_3_4_6_v5_decoupled_FINAL_v2_v3_with_results_processed_tiles.csv"
    
    # The column to use as the unique ID
    ID_COLUMN = "id_x" 
    
    # The list of columns from the original dataset you want to carry over to the final Excel
    COLUMNS_TO_KEEP = ['id_x', 'eventDate', 'decimalLatitude', 'decimalLongitude', 
                       'dataset']
    
    # -------------------------------------------------------------
    # 2. RUN SCRIPT
    # -------------------------------------------------------------
    compile_annotated_tiles(INPUT_CSV, OUTPUT_CSV, ID_COLUMN, COLUMNS_TO_KEEP)