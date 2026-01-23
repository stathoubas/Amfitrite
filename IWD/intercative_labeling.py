# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 16:18:36 2026

@author: K. Pikounis
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

def interactive_labeling(file_path):
    # 1. Read the Excel file
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Loaded {len(df)} rows from {file_path}")
    print("Type 'exit' at any prompt to save and quit.\n")

    # Keep track if we made any changes
    changes_made = False

    # 2. Iterate through rows
    # We use index to update the DataFrame in place
    for index in df.index:
        
        if df.at[index, 'create_image'] == 1:
            print("-" * 60)
            print(f"Processing Row {index + 1}...")
            
            # --- EXTRACT DATA ---
            row_data = df.loc[index]
            source_path = row_data['source_path']
            current_class = str(row_data['Class']).strip()
            # Handle potential NaNs in training_priority gracefully
            try:
                current_priority = int(row_data['training_priority'])
            except (ValueError, TypeError):
                current_priority = None

            # --- PRINT INFO ---
            print("\n--- Current Data ---")
            print(f"Abundance List:     {row_data['abun_list']}")
            print(f"Predictions:        High: {row_data['high_pred']}, Mod: {row_data['mod_pred']}, Low: {row_data['low_pred']}")
            print(f"HAB Severity Index: {row_data['HAB_severity_index']}")
            print(f"HAB Status:         {row_data['HAB_status']}")
            print(f"Flag Class:         {row_data['flag class']}")
            print(f"Current Class:      {current_class}")
            print(f"Current Priority:   {current_priority}")

            # --- SHOW IMAGE ---
            image_path = os.path.join(source_path, "cyfi_prediction_map.png")
            
            if os.path.exists(image_path):
                try:
                    img = mpimg.imread(image_path)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"Row {index+1}: {os.path.basename(source_path)}")
                    plt.show()
                except Exception as e:
                    print(f"Could not load image: {e}")
            else:
                print(f"\n[WARNING] Image not found at: {image_path}")

            # --- USER INPUT ---
            print("\n--- Input New Labels ---")
            
            # Input: Class
            while True:
                new_class = input("Enter new Class (High, Moderate, Low) or 'exit': ").strip()
                if new_class.lower() == 'exit':
                    save_and_exit(df, file_path)
                    return
                
                # Simple validation (optional, can remove if you want free text)
                if new_class in ['High', 'Moderate', 'Low']:
                    break
                print("Invalid input. Please enter High, Moderate, or Low.")

            # Input: Priority
            while True:
                new_prio_str = input("Enter Training Priority (1 or -1) or 'exit': ").strip()
                if new_prio_str.lower() == 'exit':
                    save_and_exit(df, file_path)
                    return
                
                if new_prio_str in ['1', '-1']:
                    new_priority = int(new_prio_str)
                    break
                print("Invalid input. Please enter 1 or -1.")

            # --- LOGIC & UPDATE ---
            
            # Determine suffix
            # We compare strings for class and integers for priority
            match_class = (new_class == current_class)
            match_prio = (new_priority == current_priority)

            suffix = ""
            if match_class and match_prio:
                suffix = " manually verified"
                print(">> Match: Verified.")
            else:
                suffix = " manually revised"
                print(">> Mismatch: Revised.")

            # Update 'flag class'
            current_flag = str(df.at[index, 'flag class'])
            if current_flag.lower() == 'nan':
                current_flag = ""
            df.at[index, 'flag class'] = current_flag + suffix

            # Update Class and Priority with NEW values
            df.at[index, 'Class'] = new_class
            df.at[index, 'training_priority'] = new_priority

            # Update create_image to -1
            df.at[index, 'create_image'] = -1
            
            changes_made = True

            # --- CONTINUE? ---
            cont = input("\nContinue to next? (Press Enter to continue, type 'exit' to stop): ")
            if cont.lower() == 'exit':
                save_and_exit(df, file_path)
                return
            
            # Clear output for next iteration in some environments (optional)
            # from IPython.display import clear_output
            # clear_output(wait=True)

    # End of loop
    if changes_made:
        print("\nNo more rows with create_image == 1.")
        save_and_exit(df, file_path)
    else:
        print("\nNo rows found with create_image == 1.")

def save_and_exit(df, path):
    print(f"\nSaving updated dataset to {path}...")
    try:
        df.to_excel(path, index=False)
        print("Success. Exiting.")
    except PermissionError:
        print("ERROR: Could not save file. Is it open in Excel? Please close it and try running the save command manually.")
        # Fallback save
        new_path = path.replace(".xlsx", "_updated.xlsx")
        df.to_excel(new_path, index=False)
        print(f"Saved to {new_path} instead.")



interactive_labeling(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\from_server\Master_summary_all_parts_with_flags_and_extra_cases_v5.xlsx")