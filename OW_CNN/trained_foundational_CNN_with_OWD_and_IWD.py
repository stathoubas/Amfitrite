# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:05:28 2026

@author: K. Pikounis

train CNNs using both OW and IW datasest
"""
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# ==============================================================================
# MODULE 1: CONFIGURATION & DATA HARMONIZER
# ==============================================================================

class DatasetHarmonizer:
    def __init__(self, iw_dir, ow_dir, iw_excel, ow_csv):
        self.iw_dir = iw_dir
        self.ow_dir = ow_dir
        self.iw_excel = iw_excel
        self.ow_csv = ow_csv

    def process_iw_data(self):
        """Loads and standardizes the Inland Water dataset."""
        print("Loading Inland Water (IW) Dataset...")
        df_iw = pd.read_excel(self.iw_excel)
        
        standardized_rows = []
        for _, row in df_iw.iterrows():
            uid = str(row['uid'])
            folder_path = os.path.join(self.iw_dir, uid)
            
            # Map labels
            indicative_class = str(row['new_indicative_class']).strip().lower()
            if indicative_class in ['high', 'moderate']:
                binary_label = 1
                strat_group = 'iw_hab'
            else:
                binary_label = 0
                strat_group = 'iw_nonhab'
                
            # Verify physical folder exists
            if os.path.isdir(folder_path):
                standardized_rows.append({
                    'uid': uid,
                    'folder_path': folder_path,
                    'binary_label': binary_label,
                    'strat_group': strat_group,
                    'source': 'IW'
                })
                
        print(f" -> Found {len(standardized_rows)} valid IW folders.")
        return pd.DataFrame(standardized_rows)

    def process_ow_data(self):
        """Loads and standardizes the Open Water dataset."""
        print("Loading Open Water (OW) Dataset...")
        df_ow = pd.read_csv(self.ow_csv)
        
        standardized_rows = []
        for _, row in df_ow.iterrows():
            uid = str(row['ID'])
            
            # Determine logic for Open Water suffix and strat group
            if row['tile_is_hab'] == True:
                suffix = "HAB"
                binary_label = 1
                strat_group = "ow_hab"
            else:
                binary_label = 0
                if row['tile_status'] == 'land':
                    suffix = "land"
                    strat_group = "land"
                elif row['tile_status'] == 'clouds':
                    suffix = "clouds"
                    strat_group = "clouds"
                else:
                    suffix = "nonHAB"
                    strat_group = "ow_nonhab"
                    
            folder_path = os.path.join(self.ow_dir, f"{uid}_{suffix}")
            
            # Verify physical folder exists
            if os.path.isdir(folder_path):
                standardized_rows.append({
                    'uid': uid,
                    'folder_path': folder_path,
                    'binary_label': binary_label,
                    'strat_group': strat_group,
                    'source': 'OW'
                })
                
        print(f" -> Found {len(standardized_rows)} valid OW folders.")
        return pd.DataFrame(standardized_rows)

    def create_master_registry(self):
        """Merges both datasets into a single, clean Pandas DataFrame."""
        df_iw = self.process_iw_data()
        df_ow = self.process_ow_data()
        
        master_df = pd.concat([df_iw, df_ow], ignore_index=True)
        print(f"\n[Success] Master Registry created with {len(master_df)} total images.")
        print("Distribution by Stratification Group:")
        print(master_df['strat_group'].value_counts())
        
        return master_df


# ==============================================================================
# MODULE 2: SPLITTER & IMBALANCE CALCULATOR
# ==============================================================================

def create_stratified_split_and_weights(master_df, output_registry_path, random_seed=42):
    """
    Splits the master dataframe 70/15/15 based on strat_group.
    Calculates PyTorch class weights purely from the Training set.
    """
    if os.path.exists(output_registry_path):
        print(f"\nLoading existing split from {output_registry_path}...")
        df = pd.read_csv(output_registry_path)
    else:
        print("\nPerforming 70/15/15 Stratified Split...")
        df = master_df.copy()
        df['split'] = 'junk'
        
        # 1. First split: 70% Train, 30% Temp
        train_idx, temp_idx = train_test_split(
            df.index, test_size=0.30, stratify=df['strat_group'], random_state=random_seed
        )
        # 2. Second split: 15% Val, 15% Test (from the 30% Temp)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, stratify=df.loc[temp_idx, 'strat_group'], random_state=random_seed
        )
        
        df.loc[train_idx, 'split'] = 'training'
        df.loc[val_idx, 'split'] = 'validation'
        df.loc[test_idx, 'split'] = 'test'
        
        df.to_csv(output_registry_path, index=False)
        print(f"Split saved to {output_registry_path}")

    # --- CALCULATE CLASS WEIGHTS ---
    # We strictly use the Training set to calculate weights to prevent data leakage!
    train_df = df[df['split'] == 'training']
    
    count_nonhab = len(train_df[train_df['binary_label'] == 0])
    count_hab = len(train_df[train_df['binary_label'] == 1])
    total = count_nonhab + count_hab
    
    # Inverse Frequency Weighting Formulation
    weight_nonhab = total / (2.0 * count_nonhab)
    weight_hab = total / (2.0 * count_hab)
    
    # Create tensor for PyTorch CrossEntropyLoss
    class_weights = torch.tensor([weight_nonhab, weight_hab], dtype=torch.float32)
    
    print("\n--- Training Set Imbalance Calculated ---")
    print(f"Non-HAB Samples (Class 0): {count_nonhab} -> Weight: {weight_nonhab:.3f}")
    print(f"HAB Samples (Class 1):     {count_hab} -> Weight: {weight_hab:.3f}")
    
    return df, class_weights

# --- TESTING MODULES 1 & 2 ---
if __name__ == "__main__":
    # Define paths based on your inputs
    IW_DIR = "/home/kostas/AMFITRITE/data256"
    OW_DIR = "/home/kostas/AMFITRITE/OWdata"
    IW_EXCEL = "/home/kostas/AMFITRITE/dataset_summary_256x256pixels.xlsx"
    OW_CSV = "/home/kostas/AMFITRITE/OWdata/amfitrite_open_waters_master.csv"
    MASTER_OUTPUT = "./amfitrite_universal_split.csv"

    # Run Module 1
    harmonizer = DatasetHarmonizer(IW_DIR, OW_DIR, IW_EXCEL, OW_CSV)
    master_dataframe = harmonizer.create_master_registry()
    
    # Run Module 2
    split_df, loss_weights = create_stratified_split_and_weights(master_dataframe, MASTER_OUTPUT)
