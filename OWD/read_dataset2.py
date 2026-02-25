# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:39:06 2026

@author: K. Pikounis
"""

import pandas as pd
import numpy as np
import os

# Define file names
root_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Harmful Algal Blooms"
input_file = 'occurrence.txt'
output_data = 'homogenized_occurrence.csv'
output_summary = 'new_dataset_summary.csv'

def process_dataset():
    print(f"Reading data from {input_file}...")
    # Read the occurrence.txt using tab separator as indicated by meta.xml
    df = pd.read_csv(os.path.join(root_path, input_file), sep='\t', low_memory=False, on_bad_lines='skip')

    # 1. HOMOGENIZE COLUMN NAMES
    # Map columns from the new dataset to match the previous dataset
    column_mapping = {
        'id': 'occurrenceID',               # Map record ID
        'individualCount': 'organismQuantity' # Map the concentration/count
    }
    
    # Rename columns explicitly
    df.rename(columns=column_mapping, inplace=True)

    # 2. HANDLE CONCENTRATION UNITS
    # Assuming the current dataset is in number/ml, convert it to number/L (cells per litre)
    if 'organismQuantity' in df.columns:
        print("Converting concentration units from number/ml to number/L...")
        
        # Clean the column in case numbers are saved as strings with commas (e.g., "30,000")
        if df['organismQuantity'].dtype == object:
            df['organismQuantity'] = df['organismQuantity'].astype(str).str.replace(',', '').astype(float)
        
        # Multiply by 1000 to convert number/ml to number/L
        df['organismQuantity'] = df['organismQuantity'] * 1000
        
        # Explicitly add the unit column to match the previous dataset structure
        df['organismQuantityType'] = 'cells per litre'

    # Save the homogenized dataset as CSV
    df.to_csv(os.path.join(root_path, output_data), index=False)
    print(f"Successfully saved homogenized dataset to: {output_data}")

    # 3. CREATE SUMMARY FILE
    print("Generating column summary file...")
    summary_data = []
    
    for col in df.columns:
        summary_data.append({
            'Column Name': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].notna().sum(),
            'Null Count': df[col].isna().sum(),
            'Unique Values': df[col].nunique(),
            'Sample Value': df[col].dropna().iloc[0] if df[col].notna().sum() > 0 else 'N/A'
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(root_path, output_summary), index=False)
    print(f"Successfully saved summary to: {output_summary}")

# Execute the script
if __name__ == "__main__":
    try:
        process_dataset()
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")