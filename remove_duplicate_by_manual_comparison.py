# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 00:55:04 2026

@author: K. Pikounis
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

def interactive_duplicate_resolver(file_path):
    # 1. Load the DataFrame
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # 2. Identify Duplicates (Optimized)
    # This finds all UIDs that appear more than once
    duplicate_counts = df['uid'].value_counts()
    duplicate_uids = duplicate_counts[duplicate_counts > 1].index.tolist()

    print(f"Found {len(duplicate_uids)} UIDs with duplicates.")
    
    if len(duplicate_uids) == 0:
        print("No duplicates found. Exiting.")
        return

    # Keep track of whether we need to save
    changes_made = False

    # 3. Iterate through duplicates
    # We iterate over a copy of the list so we can modify the DF safely
    for i, current_uid in enumerate(duplicate_uids):
        
        # Get the rows for this UID
        # We look up by UID again in case previous iterations affected things (though unlikely with unique pairs)
        rows = df[df['uid'] == current_uid]
        
        # Validation: Ensure we strictly have 2 rows as stated in requirements
        if len(rows) != 2:
            print(f"Skipping UID {current_uid}: Expected 2 rows, found {len(rows)}.")
            continue

        # Get indices to modify/drop later
        idx_left = rows.index[0]
        idx_right = rows.index[1]
        
        # Get paths
        path_left = os.path.join(str(rows.loc[idx_left, 'source_path']), "cyfi_prediction_map.png")
        path_right = os.path.join(str(rows.loc[idx_right, 'source_path']), "cyfi_prediction_map.png")

        print("="*60)
        print(f"Processing Pair {i+1}/{len(duplicate_uids)} | UID: {current_uid}")
        print(f"Left Source:  ...{str(rows.loc[idx_left, 'source_path'])}")
        print(f"Right Source: ...{str(rows.loc[idx_right, 'source_path'])}")

        # 4. Plot Images Side-by-Side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Helper to load and show image
        def show_img_on_ax(ax, path, title):
            if os.path.exists(path):
                try:
                    img = mpimg.imread(path)
                    ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading image:\n{e}", ha='center')
            else:
                ax.text(0.5, 0.5, "Image not found", ha='center')
            ax.axis('off')
            ax.set_title(title)

        show_img_on_ax(axes[0], path_left, "LEFT (Option L)")
        show_img_on_ax(axes[1], path_right, "RIGHT (Option R)")
        
        plt.tight_layout()
        plt.show()

        # 5. User Decision
        while True:
            choice = input("Decide: Keep (L)eft, Keep (R)ight, Keep (B)oth? (or type 'exit'): ").strip().lower()
            
            if choice == 'exit':
                save_and_exit(df, file_path)
                return

            if choice.lower() == 'l':
                # Keep Left -> Drop Right
                print(f"Keeping Left. Deleting row index {idx_right}...")
                df.drop(idx_right, inplace=True)
                changes_made = True
                break
            
            elif choice.lower() == 'r':
                # Keep Right -> Drop Left
                print(f"Keeping Right. Deleting row index {idx_left}...")
                df.drop(idx_left, inplace=True)
                changes_made = True
                break
            
            elif choice.lower() == 'b':
                # Keep Both -> Append 'a' and 'b'
                print("Keeping Both. Renaming UIDs...")
                df.at[idx_left, 'uid'] = f"{current_uid}a"
                df.at[idx_right, 'uid'] = f"{current_uid}b"
                changes_made = True
                break
            
            else:
                print("Invalid input. Please enter L, R, B, or exit.")

        # 6. Check if user wants to proceed to next pair
        # (The prompt implies asking to proceed *after* each case)
        cont = input("Proceed to next duplicate? (Press Enter for Yes, type 'exit' to save & quit): ").strip().lower()
        if cont == 'exit':
            save_and_exit(df, file_path)
            return
        
        # Clear previous plots/output if running in a loop-heavy env (optional)
        # plt.close('all') 

    # End of loop
    print("\nAll duplicates processed.")
    if changes_made:
        save_and_exit(df, file_path)
    else:
        print("No changes were made.")

def save_and_exit(df, path):
    print("\nSaving DataFrame...")
    # It is often safer to save to a new file first, but per your request 
    # we will overwrite or save to the specified location.
    # To be safe, let's append "_resolved" so we don't accidentally destroy your source 
    # if you press the wrong button, unless you strictly want to overwrite.
    
    # Overwrite method (Uncomment to overwrite original):
    # save_path = path 
    
    # Safer method:
    save_path = path.replace(".xlsx", "_resolved.xlsx")
    
    try:
        df.to_excel(save_path, index=False)
        print(f"Successfully saved to: {save_path}")
    except PermissionError:
        print("Error: Could not save. Please close the Excel file if it is open.")
        alt_path = "recovered_data.xlsx"
        df.to_excel(alt_path, index=False)
        print(f"Saved to {alt_path} instead.")
    sys.exit()

# --- Execution ---
if __name__ == "__main__":
    # Update this path to your actual file
    FILE_PATH = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\from_server\Master_summary_all_parts_with_flags_and_extra_cases_v7_unique_uids_resolved.xlsx"
    
    interactive_duplicate_resolver(FILE_PATH)