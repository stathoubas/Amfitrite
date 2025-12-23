# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 17:55:45 2025

@author: K. Pikounis
"""

import pandas as pd
import json
import sys
from pathlib import Path
from tqdm import tqdm
from time import time

from compatibility_utils_v2 import calculate_compatibility_score, calculate_spatial_roughness 

INPUT_EXCEL = "Master_summary.xlsx"
RX_METERS = 300
NMIN_PIXELS = 50

# Thresholds for visual flag
ROUGHNESS_THRESH = 0.4
OUTLIER_THRESH = 0.15

# Thresholds for HAB Status
SEVERE_THRESH = 0.10   # >10% High
MODERATE_THRESH = 0.10 # >10% Mod+High

def map_severity_to_class(sev_val):
    """Maps numeric/string severity to High/Moderate/Low."""
    try:
        # Handle string inputs like 'high', 'Low'
        s = str(sev_val).lower().strip()
        if s == 'high': return 'High'
        if s == 'moderate': return 'Moderate'
        if s == 'low': return 'Low'
        
        # Handle numeric inputs
        val = int(float(sev_val))
        if val >= 3: return 'High'
        if val == 2: return 'Moderate'
        if val <= 1: return 'Low'
    except:
        return None
    return None

def process_master_file(input_excel, output_filename, SEVERE_THRESH = 0.10, MODERATE_THRESH = 0.10, 
                        RX_METERS = 300, NMIN_PIXELS = 50, ROUGHNESS_THRESH = 0.4, OUTLIER_THRESH = 0.15):
    '''

    Parameters
    ----------
    input_excel : str
    output_filename : str
    SEVERE_THRESH : float, optional
        DESCRIPTION. percentage of points in the 10x10 pixel grid with class high, above which the label will be High 0,1 => 10%
        The default is 0.10.
    MODERATE_THRESH : float, optional
        DESCRIPTION. percentage of points in the 10x10 pixel grid with class high and moderate, above which the label will be Moderate 0,1 => 10% 
        The default is 0.10.
    RX_METERS : float, optional
        Radius of circle round the measuremnt that the cyfi will run to make predictions, create the log normal distribution to deremine
        if the measrument is compatible with the predictions The default is 300.
    NMIN_PIXELS : int, optional
        Minimum numbr of points in the circle with radius RX_METERS that the measurement - predictions compatibility serach will run. The default is 50.
    ROUGHNESS_THRESH : float, optional
        The limit for the median variability of the entire image above which the image is flages as globally noisy. The default is 0.4.
    OUTLIER_THRESH : float, optional
        The maximum allowable percentage of individual pixels that deviate significantly from their neighbors above which the image is flagged as noisy. 
        The default is 0.15.

    Returns
    -------
    None.

    '''
    print(f"--- Reading {input_excel} ---")
    
    try:
        df = pd.read_csv(input_excel) if input_excel.endswith('.csv') else pd.read_excel(input_excel)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Initialize new columns if they don't exist
    new_cols = [
        "HAB_severity_index", "HAB_status", 
        "compatible_severities", 
        "roughness_median", "outlier_fraction", "num_points", "num_outliers"
    ]
    for col in new_cols:
        df[col] = None

    print(f"--- Processing {len(df)} rows ---")
    t0 = time()
    t1 = time()
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        print("----------------------->  Hello ", idx, round((time() - t0)/60, 1), round(time() - t1))
        t1 = time()
        
        # ---------------------------------------------------------
        # 1. HAB Severity Index & 2. HAB Status
        # ---------------------------------------------------------
        try:
            h = float(row.get('high_pred', 0))
            m = float(row.get('mod_pred', 0))
            l = float(row.get('low_pred', 0))
            total = h + m + l
            
            if total > 0:
                frac_h = h / total
                frac_m = m / total
                
                # Index Formula
                idx_val = frac_h + (0.5 * frac_m)
                df.at[idx, 'HAB_severity_index'] = round(idx_val, 4)
                
                # Status Logic
                if frac_h > SEVERE_THRESH:
                    status = "High" # (User asked for High/Moderate/Low, mapping Severe->High for consistency or strictly Severe?)
                elif (frac_h + frac_m) > MODERATE_THRESH:
                    status = "Moderate"
                else:
                    status = "Low"
                df.at[idx, 'HAB_status'] = status
            else:
                df.at[idx, 'HAB_severity_index'] = 0
                df.at[idx, 'HAB_status'] = "Low"
                
        except Exception as e:
            print(f"Row {idx} Severity Error: {e}")
            df.at[idx, 'HAB_severity_index'] = -1
            df.at[idx, 'HAB_status'] = "Error"

        # ---------------------------------------------------------
        # Path Handling
        # ---------------------------------------------------------
        source_path = row.get('source_path')
        if not source_path or pd.isna(source_path):
            df.at[idx, 'compatible_severities'] = "Path Empty"
            continue
            
        folder_path = Path(source_path)
        if not folder_path.exists():
            df.at[idx, 'compatible_severities'] = "Folder Not Found"
            continue

        # ---------------------------------------------------------
        # 3. Compatibility Score & Metadata Update
        # ---------------------------------------------------------
        try:
            # Run calculations
            comp_result = calculate_compatibility_score(folder_path, Rx=RX_METERS, Nmin=NMIN_PIXELS)
            
            if comp_result and "clusters" in comp_result:
                # A. Find Compatible Severities
                matched_severities = set()
                
                # Deep copy to clean up for metadata_2.json
                clusters_for_json = []
                
                for cluster in comp_result["clusters"]:
                    # Create clean version for JSON
                    clean_cluster = {
                        "centroid_lat": cluster.get("centroid_lat"),
                        "centroid_lon": cluster.get("centroid_lon"),
                        "cluster_incompatibility_score": round(cluster.get("cluster_incompatibility_score"), 4),
                        "most_compatible_uid": cluster.get("most_compatible_uid"),
                        "points": []
                    }
                    
                    for pt in cluster["points"]:
                        # Extract Severity for Excel Column if Compatible (Score < 3.0)
                        if pt.get("compatibility_score", 99) < 3.0:
                            s_class = map_severity_to_class(pt.get("severity"))
                            if s_class: matched_severities.add(s_class)
                        
                        # Strip point data for JSON
                        clean_cluster["points"].append({
                            "uid": pt.get("uid"),
                            "severity": pt.get("severity"),
                            "compatibility_score": round(pt.get("compatibility_score"), 4)
                        })
                    
                    clusters_for_json.append(clean_cluster)
                
                # Write to Excel
                if matched_severities:
                    # Sort for consistency (High, Moderate, Low)
                    order = {"High": 1, "Moderate": 2, "Low": 3}
                    sorted_sev = sorted(list(matched_severities), key=lambda x: order.get(x, 4))
                    df.at[idx, 'compatible_severities'] = ", ".join(sorted_sev)
                else:
                    df.at[idx, 'compatible_severities'] = "None"

                # B. Create metadata_2.json
                meta_path = folder_path / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, 'r') as f: meta_data = json.load(f)
                    
                    # Extend metadata
                    meta_data["compatibility_analysis"] = {
                        "max_case_score": comp_result.get("max_case_score"),
                        "clusters": clusters_for_json
                    }
                    
                    with open(folder_path / "metadata_2.json", 'w') as f:
                        json.dump(meta_data, f, indent=4)
                        
            else:
                df.at[idx, 'compatible_severities'] = "Calc Failed"
                        
        except Exception as e:
            print(f"Row {idx} Compatibility Error: {e}")
            df.at[idx, 'compatible_severities'] = "Error"

        # ---------------------------------------------------------
        # 4. Spatial Roughness
        # ---------------------------------------------------------
        try:
            csv_path = folder_path / "cyfi_lattice_predictions.csv"
            if csv_path.exists():
                rough_res = calculate_spatial_roughness(csv_path)
                
                if rough_res and rough_res.get("status") == "OK":
                    rm = rough_res.get("roughness_median")
                    of = rough_res.get("outlier_fraction")
                    nump = int(rough_res.get("num_points"))
                    numo = int(rough_res.get("num_outliers"))
                    
                    df.at[idx, 'roughness_median'] = rm
                    df.at[idx, 'outlier_fraction'] = of
                    df.at[idx, 'num_points'] = nump
                    df.at[idx, 'num_outliers'] = numo
                    
                    # Update Pred Visual Flag
                    # "put 1 if roughness_median > 0.4 or outlier_fraction > 0.15"
                    if (rm is not None and rm > ROUGHNESS_THRESH) or \
                       (of is not None and of > OUTLIER_THRESH):
                        df.at[idx, 'pred_visual'] = 1
                    else:
                        # Optional: Reset to 0 if clean? Or keep existing?
                        # Usually "Update" implies overwriting.
                        df.at[idx, 'pred_visual'] = 0
                else:
                    # File exists but status not OK (e.g. Too Sparse)
                    df.at[idx, 'roughness_median'] = -2
                    df.at[idx, 'outlier_fraction'] = -2
            else:
                # No CSV found
                df.at[idx, 'roughness_median'] = -1
                df.at[idx, 'outlier_fraction'] = -1

        except Exception as e:
             print(f"Row {idx} Roughness Error: {e}")
             df.at[idx, 'roughness_median'] = -1
             df.at[idx, 'outlier_fraction'] = -1
             df.at[idx, 'num_points'] = -1
             df.at[idx, 'num_outliers'] = -1
             df.at[idx, 'pred_visual'] = 0

    try:
        if not output_filename.endswith('.xlsx'):
            output_filename += ".xlsx"
            
        df.to_excel(output_filename, index=False)
        print(f"--- Success! Saved to {output_filename} ---")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python Master_Wrapper.py <input_file> <output_file>")
        print("Example: python Master_Wrapper.py input_part_1.xlsx out_part_1.xlsx")
        sys.exit(1)
        
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    
    process_master_file(in_file, out_file)