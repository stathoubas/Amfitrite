import pandas as pd
import os
import shutil
import glob
import json
from tqdm import tqdm

def organize_dataset_os_specific(excel_path, target_root_dir, current_os_mode):
    """
    current_os_mode: 'windows' or 'ubuntu'
    """
    # 1. Load the Excel file
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: File not found at {excel_path}")
        return

    print(f"Loaded {len(df)} rows.")
    print(f"Running in '{current_os_mode}' mode. Filtering paths...")

    # Create target root if it doesn't exist
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)

    # Initialize new columns if they don't exist
    for col in ['metadata_2_existed', 'uid_old', 'abun_list']:
        if col not in df.columns:
            df[col] = "" # Initialize as empty string or suitable default

    processed_count = 0

    # 2. Iterate through the DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        
        source_path = str(row['source_path']).strip()
        uid = str(row['uid']).strip()

        is_windows_path = source_path.lower().startswith("c:") or source_path.lower().startswith("d:")
        is_ubuntu_path = source_path.startswith("/")

        if current_os_mode == 'windows':
            if not is_windows_path:
                continue # Skip Ubuntu paths
        elif current_os_mode == 'ubuntu':
            if not is_ubuntu_path:
                continue # Skip Windows paths
        else:
            print("Invalid OS mode selected.")
            return

        processed_count += 1

        # Create the destination folder based on UID
        dest_folder = os.path.join(target_root_dir, uid)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Check if source exists (Double check)
        if not os.path.exists(source_path):
            # record the error but keep the row in excel
            print(f"Warning: Path not found on this machine: {source_path}")
            # df.at[index, 'metadata_2_existed'] = "Source Not Found" 
            continue

        # --- A. Metadata Logic (Read & Copy) ---
        path_meta_2 = os.path.join(source_path, "metadata_2.json")
        path_meta_1 = os.path.join(source_path, "metadata.json")
        dest_meta_path = os.path.join(dest_folder, "metadata.json")
        
        file_to_copy = None
        used_meta_2 = False
        
        if os.path.exists(path_meta_2):
            file_to_copy = path_meta_2
            used_meta_2 = True
        elif os.path.exists(path_meta_1):
            file_to_copy = path_meta_1
            used_meta_2 = False
        
        # Extract Data
        uids_found = []
        abuns_found = []
        
        if file_to_copy:
            try:
                # 1. Copy and rename to metadata.json in destination
                shutil.copy2(file_to_copy, dest_meta_path)
                
                # 2. Read content for DataFrame update
                with open(file_to_copy, 'r') as f:
                    meta_data_content = json.load(f)
                    
                # 3. Extract points_data
                points_data = meta_data_content.get("points_data", [])
                
                for point in points_data:
                    # Collect UIDs
                    if "uid" in point:
                        uids_found.append(point["uid"])
                    # Collect Abuns
                    if "abun" in point:
                        abuns_found.append(point["abun"])
                        
            except Exception as e:
                print(f"Error processing metadata for {uid}: {e}")

        # Update DataFrame for this row
        df.at[index, 'metadata_2_existed'] = used_meta_2
        df.at[index, 'uid_old'] = str(uids_found)
        df.at[index, 'abun_list'] = str(abuns_found)

        # --- B. Copy other files (.tif, csv, png) ---
        # Copy TIFs
        tif_files = glob.glob(os.path.join(source_path, "*.tif"))
        for tif in tif_files:
            try:
                shutil.copy2(tif, dest_folder)
            except Exception:
                pass 

        # Copy Specific Files
        files_to_copy = ["cyfi_lattice_predictions.csv", "cyfi_prediction_map.png"]
        for fname in files_to_copy:
            src = os.path.join(source_path, fname)
            if os.path.exists(src):
                shutil.copy2(src, dest_folder)

    # 3. Save the new excel
    output_excel = excel_path.replace(".xlsx", f"_updated_{current_os_mode}.xlsx")
    df.to_excel(output_excel, index=False)
    
    print("-" * 50)
    print(f"Process Complete for mode: {current_os_mode}")
    print(f"Processed {processed_count} rows.")
    print(f"Dataset organized in: {target_root_dir}")
    print(f"Updated Excel saved to: {output_excel}")

if __name__ == "__main__":
    
    INPUT_EXCEL = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\from_server\Master_summary_all_parts_with_flags_and_extra_cases_v7_unique_uids_resolved.xlsx" 
    TARGET_DIR = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\Dataset_v1"
    
    # windows or ubuntu
    OS_MODE = 'windows' 
    
    organize_dataset_os_specific(INPUT_EXCEL, TARGET_DIR, OS_MODE)