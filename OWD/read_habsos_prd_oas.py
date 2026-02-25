# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:55:53 2026

@author: K. Pikounis
"""

import pandas as pd
import os

# Define your file names here (update paths if they are in different folders)
root_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\prd OAS\8.8"
input_csv = 'habsos_20240430.csv' # The most recent CSV mentioned in your journal.txt
input_lonlat = '0120767_lonlat.txt'
output_csv = 'homogenized_habsos.csv'

def process_habsos():
    print(f"Loading data from {input_csv}...")
    # Read the main data
    df = pd.read_csv(os.path.join(root_path, "data", "0-data", input_csv), low_memory=False)
    
    # 1. CORRECT THE COORDINATES
    # The journal.txt states the lonlat.txt has longitude on the left, latitude on the right
    if os.path.exists(os.path.join(root_path,"8.8", "about", input_lonlat)):
        print(f"Loading corrected coordinates from {input_lonlat}...")
        # sep=r'\s+' handles any number of spaces or tabs between the columns
        # header=None tells pandas the first row is data, not column names
        df_lonlat = pd.read_csv(os.path.join(root_path,"8.8", "about", input_lonlat), sep=r'\s+', header=None, names=['corr_lon', 'corr_lat'])
        
        # Verify the row counts match before applying
        if len(df) == len(df_lonlat):
            print("Row counts match! Overwriting original LATITUDE and LONGITUDE...")
            df['LONGITUDE'] = df_lonlat['corr_lon']
            df['LATITUDE'] = df_lonlat['corr_lat']
        else:
            print(f"Warning: Row count mismatch! CSV has {len(df)} rows, txt has {len(df_lonlat)} rows. Skipping coordinate correction.")
    else:
        print(f"Warning: {input_lonlat} not found. Proceeding with original coordinates from the CSV.")
    # 2. FILTER OUT BAD DATA
    # According to the data dictionary, CELLCOUNT_QA == 9 means "no data"
    if 'CELLCOUNT_QA' in df.columns:
        initial_len = len(df)
        df = df[df['CELLCOUNT_QA'] != 9].copy()
        print(f"Filtered out {initial_len - len(df)} rows where CELLCOUNT_QA indicated 'no data'.")

    # 3. CREATE 'scientificName'
    print("Merging GENUS and SPECIES into 'scientificName'...")
    # Fill empty values with blank strings so we don't get literal "NaN" in our text
    df['GENUS'] = df['GENUS'].fillna('')
    df['SPECIES'] = df['SPECIES'].fillna('')
    
    # Combine them and strip any extra whitespace (e.g., if species is missing)
    df['scientificName'] = df['GENUS'].astype(str) + ' ' + df['SPECIES'].astype(str)
    df['scientificName'] = df['scientificName'].str.strip()

    # 4. RENAME COLUMNS (HOMOGENIZATION)
    habsos_to_homogenized_mapping = {
        'OBJECTID': 'id_x',
        'LATITUDE': 'decimalLatitude',
        'LONGITUDE': 'decimalLongitude',
        'SAMPLE_DATE': 'eventDate',
        'SAMPLE_DEPTH': 'minimumDepthInMeters',
        'CELLCOUNT': 'organismQuantity',
        'CELLCOUNT_UNIT': 'organismQuantityType'
    }
    
    df.rename(columns=habsos_to_homogenized_mapping, inplace=True)
    
    # 5. SAVE
    df.to_csv(os.path.join(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\prd OAS", output_csv), index=False)
    print(f"Successfully processed and saved to: {output_csv}")
    print(f"Total columns kept: {len(df.columns)}")

if __name__ == "__main__":
    try:
        process_habsos()
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")