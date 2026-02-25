# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 16:36:00 2026

@author: K. Pikounis
"""

import pandas as pd
import requests
import io
import zipfile
import os

# The Zenodo Record ID for the Arctic Microbial Plankton dataset
zenodo_record_id = "10557176"
api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"
output_csv = "homogenized_arctic_plankton.csv"

def fetch_and_homogenize_zenodo():
    print(f"Querying Zenodo API for record {zenodo_record_id}...")
    response = requests.get(api_url)
    
    if response.status_code != 200:
        print(f"Failed to fetch metadata from Zenodo. Status code: {response.status_code}")
        return
        
    data = response.json()
    files = data.get('files', [])
    
    if not files:
        print("No files found in this Zenodo record.")
        return

    # Look for the primary data file
    target_file = None
    for f in files:
        if f['key'].endswith('.csv') or f['key'].endswith('.txt') or f['key'].endswith('.zip'):
            target_file = f
            break
            
    if not target_file:
        print("Could not find a valid data file (.csv, .txt, .zip) in the repository.")
        return

    file_url = target_file['links']['self']
    file_name = target_file['key']
    print(f"Found data file: {file_name}. Downloading...")
    
    # Download the file
    file_response = requests.get(file_url)
    df = None
    
    # Load the data into Pandas based on file type
    # ADDED encoding='latin1' to handle special characters
    if file_name.endswith('.zip'):
        print("Extracting ZIP file in memory...")
        with zipfile.ZipFile(io.BytesIO(file_response.content)) as z:
            csv_files = [name for name in z.namelist() if name.endswith('.csv') or name.endswith('.txt')]
            main_file = max(csv_files, key=lambda x: z.getinfo(x).file_size)
            print(f"Loading {main_file} from ZIP...")
            separator = '\t' if main_file.endswith('.txt') else ','
            with z.open(main_file) as f:
                df = pd.read_csv(f, sep=separator, low_memory=False, encoding='latin1')
    else:
        separator = '\t' if file_name.endswith('.txt') else ','
        df = pd.read_csv(io.BytesIO(file_response.content), sep=separator, low_memory=False, encoding='latin1')

    print(f"Dataset loaded. Total rows: {len(df)}. Total columns: {len(df.columns)}.")

    # --- HOMOGENIZATION ---
    print("Homogenizing columns...")
    
    homogenize_mapping = {
        'id': 'id_x',
        'eventid': 'id_x',
        'occurrenceid': 'id_x',
        'lat': 'decimalLatitude',
        'latitude': 'decimalLatitude',
        'lon': 'decimalLongitude',
        'longitude': 'decimalLongitude',
        'long': 'decimalLongitude',
        'date': 'eventDate',
        'sample_date': 'eventDate',
        'datetime': 'eventDate',
        'time': 'eventTime',
        'depth': 'minimumDepthInMeters',
        'sample_depth': 'minimumDepthInMeters',
        'species': 'scientificName',
        'taxon': 'scientificName',
        'scientific_name': 'scientificName',
        'abundance': 'organismQuantity',
        'cellcount': 'organismQuantity',
        'concentration': 'organismQuantity',
        'cells_per_liter': 'organismQuantity',
        'units': 'organismQuantityType',
        'unit': 'organismQuantityType'
    }

    original_columns = df.columns.tolist()
    col_lower_map = {col.lower(): col for col in original_columns}
    
    rename_dict = {}
    for lower_col, original_col in col_lower_map.items():
        if lower_col in homogenize_mapping:
            rename_dict[original_col] = homogenize_mapping[lower_col]

    # Apply the renaming
    df.rename(columns=rename_dict, inplace=True)
    
    # Fill in quantity type if missing but we have cell counts
    if 'organismQuantity' in df.columns and 'organismQuantityType' not in df.columns:
        print("Detected organism quantities but no unit column. Setting to 'cells per litre' as default.")
        df['organismQuantityType'] = 'cells per litre'

    # Save to CSV, keeping all extra columns natively
    df.to_csv(os.path.join(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\North American Arctic region", output_csv), index=False)
    print(f"Success! Saved {len(df)} rows and {len(df.columns)} columns to {output_csv}.")

if __name__ == "__main__":
    try:
        fetch_and_homogenize_zenodo()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
