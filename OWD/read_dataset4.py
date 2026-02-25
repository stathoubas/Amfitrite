# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 12:56:55 2026

@author: K. Pikounis
"""

import pandas as pd
import numpy as np
import os

root_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\4 Harmful algal blooms in South America except Venezuela, Colombia and the Guyanas"
# Define file names
occurrence_file = 'occurrence.txt'
# Depending on the exact extraction, it might be named extendedmeasurementorfact.txt or measurementorfact.txt
mof_file = 'extendedmeasurementorfact.txt' 

output_data = 'homogenized_dataset_4.csv'
output_summary = 'dataset_4_summary.csv'

def process_and_homogenize_d4():
    print(f"Loading {occurrence_file}...")
    df_occ = pd.read_csv(os.path.join(root_path, occurrence_file), sep='\t', on_bad_lines='skip', low_memory=False)
    
    print(f"Loading {mof_file}...")
    df_mof = pd.read_csv(os.path.join(root_path, mof_file), sep='\t', on_bad_lines='skip', low_memory=False)
    
    # 1. MERGE THE DATASETS
    # meta.xml states occurrence.txt has 'id' (index 0) and the extension has 'coreid' (index 0)
    occ_id_col = df_occ.columns[0]
    mof_id_col = df_mof.columns[0]
    
    print(f"Merging Occurrence ({occ_id_col}) and Extension ({mof_id_col})...")
    df_merged = pd.merge(df_occ, df_mof, left_on=occ_id_col, right_on=mof_id_col, how='left')
    
    # 2. HOMOGENIZE COLUMN NAMES
    # Rename columns to match the previous datasets
    rename_mapping = {
        occ_id_col: 'id_x',
        'measurementType': 'eMoF_measurementType',
        'measurementValue': 'eMoF_measurementValue',
        'measurementUnit': 'eMoF_measurementUnit'
    }
    
    df_merged.rename(columns=rename_mapping, inplace=True)
    
    # Drop duplicate 'coreid' from MoF as it's redundant with 'id_x'
    if mof_id_col in df_merged.columns and mof_id_col != occ_id_col:
        df_merged.drop(columns=[mof_id_col], inplace=True)
    
    # 3. HANDLE CONCENTRATION AND UNITS
    print("Standardizing concentration units to per Litre...")
    
    if 'eMoF_measurementUnit' in df_merged.columns and 'eMoF_measurementValue' in df_merged.columns:
        # Clean the numeric values (remove commas used for thousands)
        def clean_numeric(val):
            if pd.isna(val):
                return np.nan
            try:
                return float(str(val).replace(',', '').strip())
            except ValueError:
                return np.nan
                
        df_merged['eMoF_measurementValue'] = df_merged['eMoF_measurementValue'].apply(clean_numeric)
        
        # Identify rows where the unit implies "/ml", "per ml", etc.
        mask_ml = df_merged['eMoF_measurementUnit'].astype(str).str.lower().str.contains(r'/ml|per ml|/ ml', na=False)
        
        # Multiply values by 1000 where unit is per ml
        df_merged.loc[mask_ml, 'eMoF_measurementValue'] *= 1000
        
        # Update the unit text to indicate Litres instead of ml
        df_merged.loc[mask_ml, 'eMoF_measurementUnit'] = df_merged.loc[mask_ml, 'eMoF_measurementUnit'].astype(str).str.lower().str.replace('ml', 'l').str.replace('milliliter', 'litre')
        
        # 4. EXTRACT ORGANISM QUANTITY
        # Create organismQuantity / organismQuantityType columns if they measure concentration/abundance
        mask_abundance = df_merged['eMoF_measurementType'].astype(str).str.lower().str.contains('abundance|concentration|count|quantity', na=False)
        
        if 'organismQuantity' not in df_merged.columns:
            df_merged['organismQuantity'] = np.nan
            df_merged['organismQuantityType'] = np.nan
            
            df_merged.loc[mask_abundance, 'organismQuantity'] = df_merged.loc[mask_abundance, 'eMoF_measurementValue']
            df_merged.loc[mask_abundance, 'organismQuantityType'] = df_merged.loc[mask_abundance, 'eMoF_measurementUnit']

    # Save homogenized dataset
    df_merged.to_csv(os.path.join(root_path, output_data), index=False)
    print(f"Successfully saved homogenized dataset to: {output_data}")
    
    # 5. CREATE SUMMARY FILE
    print(f"Generating summary file: {output_summary}")
    summary_data = []
    for col in df_merged.columns:
        summary_data.append({
            'Column Name': col,
            'Data Type': str(df_merged[col].dtype),
            'Non-Null Count': df_merged[col].notna().sum(),
            'Null Count': df_merged[col].isna().sum(),
            'Unique Values': df_merged[col].nunique(),
            'Sample Value': df_merged[col].dropna().iloc[0] if df_merged[col].notna().sum() > 0 else 'N/A'
        })
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(root_path, output_summary), index=False)
    print("Done.")

if __name__ == "__main__":
    try:
        process_and_homogenize_d4()
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check if the files are correctly named and in the directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")