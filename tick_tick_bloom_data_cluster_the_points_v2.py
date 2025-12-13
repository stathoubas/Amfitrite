# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 16:18:42 2025

@author: K. Pikounis
"""


import pandas as pd
import numpy as np
from datetime import timedelta
from time import time

def Haversine(pt_1: np.ndarray, pt_2: np.ndarray) -> float:
    """
    Calculate the great-circle distance between two points
    on the Earth (specified in decimal degrees)
    
    pt_1 and pt_2 are [lon, lat] numpy arrays.
    Returns distance in meters.
    """
    R = 6371.0  # Earth radius in kilometers
    
    # Convert latitude and longitude from degrees to radians
    lon_1, lat_1 = np.radians(pt_1)
    lon_2, lat_2 = np.radians(pt_2)
    
    # Compute differences in coordinates
    dlat = lat_2 - lat_1
    dlon = lon_2 - lon_1
    
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(dlon / 2)**2
    c = 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = (R * c) * 1000 # in meters
    return distance

def perform_clustering_with_size(df: pd.DataFrame, time_window_days: int, max_distance_meters: float) -> pd.DataFrame:

    print("Preparing data...")
    df_proc = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df_proc['date']):
        try:
            df_proc['date'] = pd.to_datetime(df_proc['date'], format='%Y%m%d')
        except ValueError:
            print("Could not parse 'date' column with format YYYYMMDD. Please ensure it's in this format.")
            return df

    df_proc['case'] = pd.NA
    df_proc['case'] = df_proc['case'].astype(pd.Int64Dtype())
    df_proc['case_str'] = pd.NA
    df_proc['case_str'] = df_proc['case_str'].astype(object)
    
    time_delta = pd.Timedelta(days=time_window_days)
    lat_delta = 0.1
    lon_delta = 0.1
    
    counter = 0
    t0 = time()
    #loop in df
    for it in df_proc.index:
        row = df_proc.loc[it]
        counter += 1
        
        if counter/250 in [i for i in range(240)]:
            t1 = time()
            print(counter, "in ", round((t1-t0), 1), "s")
            t0 = t1
            
        #select all other entries
        df_except_current = df_proc.drop(it)
        
        # keep curent lon lat
        pt1 = [row["lon"], row["lat"]]
        
        #filters 
        min_date, max_date = row['date'] - time_delta, row['date'] + time_delta
        min_lat, max_lat = row['lat'] - lat_delta, row['lat'] + lat_delta
        min_lon, max_lon = row['lon'] - lon_delta, row['lon'] + lon_delta
        
        #select df by appling filters
        candidate_rows = df_except_current[
            (df_except_current['date'] >= min_date) & (df_except_current['date'] <= max_date) &
            (df_except_current['lat'] >= min_lat) & (df_except_current['lat'] <= max_lat) &
            (df_except_current['lon'] >= min_lon) & (df_except_current['lon'] <= max_lon)
        ]
        
        # in case nothing is selected
        if len(candidate_rows) == 0:
            # update case and case_str
            if pd.isna(df_proc.at[it, "case"]):
                df_proc.at[it, "case"] = counter
                df_proc.at[it, "case_str"] = str(counter)
            # should not go there
            else:
                print("problem len(candidate_rows) = ", len(candidate_rows), "but case_str = ", df_proc.at[it, "case_str"])
        
        # if at leats 1 entry is selected
        else:
            
            # use Haversine filter
            accepted_its = []
            for it2, row2 in candidate_rows.iterrows():
                pt2 = [row2["lon"], row2["lat"]]
                dist = Haversine(pt1, pt2)
                if dist <= max_distance_meters:
                    accepted_its.append(it2)
            # if nothing is selected just update case and case_str
            if len(accepted_its) == 0:
                df_proc.at[it, "case"] = counter
                df_proc.at[it, "case_str"] = str(counter)
            # some entries are selected
            else:
                # the case of the original point
                orignial_case = df_proc.at[it, "case"]
                
                # all the same cases found from a previous entry
                if not pd.isna(orignial_case) and len(set(df_proc.loc[accepted_its+[it], "case"])) == 1:
                    # do nothing
                    continue
                # all empty cases = 1st time this cluste is encountered
                elif pd.isna(orignial_case) and len(set(df_proc.loc[accepted_its+[it], "case"])) == 1:
                    # assign new case
                    accepted_its.append(it)
                    for i in accepted_its:
                        df_proc.at[i, "case"] = counter
                        df_proc.at[i, "case_str"] = str(counter)
                # new entry not in a cluster, all other entries in the same cluster:
                elif pd.isna(orignial_case) and len(set(df_proc.loc[accepted_its, "case"])) == 1:
                    old_case = list(df_proc.loc[accepted_its, "case"])[0] 
                    # assignt to new entry old clusters case
                    df_proc.at[it, "case"] = old_case
                    df_proc.at[it, "case_str"] = str(old_case)
                # more than one clusters in the final selection
                else:
                    accepted_its.append(it)
                    old_cases = [i for i in df_proc.loc[accepted_its, "case"].to_list() if not pd.isna(i)]
                    old_case_max = max(set(old_cases), key=old_cases.count)
                    for i in accepted_its:
                        if pd.isna(df_proc.at[i, "case"]):
                            df_proc.at[i, "case"] = old_case_max
                            df_proc.at[i, "case_str"] = str(old_case_max)
                        else:
                            old_str = df_proc.at[i, "case_str"]
                            df_proc.at[i, "case"] = old_case_max
                            df_proc.at[i, "case_str"] = old_str+ " , "+ str(old_case_max)

    
    cluster_sizes = df_proc['case'].value_counts()
    df_proc['cluster_size'] = df_proc['case'].map(cluster_sizes)
    df_proc['cluster_size'] = df_proc['cluster_size'].astype(pd.Int64Dtype())
    

    return df_proc



def perform_spatial_clustering_improved(df: pd.DataFrame, edge_length_meters: float) -> pd.DataFrame:
    """
    Performs spatial clustering based on a square window approach with barycenter refinement.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'lat' and 'lon' columns.
        edge_length_meters (float): The edge length of the square window in meters.
        
    Returns:
        pd.DataFrame: A copy of the input dataframe with 'case' and 'cluster_size' columns added.
    """
    print("Preparing data for spatial clustering...")
    df_proc = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df_proc['date']):
        try:
            df_proc['date'] = pd.to_datetime(df_proc['date'], format='%Y%m%d')
        except ValueError:
            print("Could not parse 'date' column with format YYYYMMDD. Please ensure it's in this format.")
            return df
    
    # Initialize columns
    df_proc['case'] = pd.NA
    df_proc['case'] = df_proc['case'].astype('Int64')
    df_proc['cluster_size'] = pd.NA
    df_proc['cluster_size'] = df_proc['cluster_size'].astype('Int64')
    
    # Constants for conversion
    # Approx meters per degree latitude
    METERS_PER_DEG_LAT = 111132.0
    
    case_counter = 1
    
    # Helper function to find points within the square window
    def get_points_in_square(c_lat, c_lon, available_indices):
        # Calculate half-edge in degrees
        # Latitude delta is constant
        d_lat = (edge_length_meters / 2) / METERS_PER_DEG_LAT
        
        # Longitude delta depends on latitude (use center latitude)
        # We use abs(lat) and convert to radians for cosine
        d_lon = (edge_length_meters / 2) / (METERS_PER_DEG_LAT * np.cos(np.radians(c_lat)))
        
        min_lat, max_lat = c_lat - d_lat, c_lat + d_lat
        min_lon, max_lon = c_lon - d_lon, c_lon + d_lon
        
        # Filter the subset of unassigned points
        subset = df_proc.loc[available_indices]
        mask = (
            (subset['lat'] >= min_lat) & (subset['lat'] <= max_lat) &
            (subset['lon'] >= min_lon) & (subset['lon'] <= max_lon)
        )
        return subset[mask].index.tolist()

    # Iterate through every point in the dataframe
    # We iterate over the index to handle non-standard indices correctly
    all_indices = df_proc.index.tolist()
   
    counter = 0
    t0 = time()
    for i in all_indices:     
        counter += 1
        if counter/250 in [i for i in range(100)]:
            t1 = time()
            print(counter, "in ", round((t1-t0), 1), "s")
            t0 = t1
        
        # Step 6: Continue to next point but only process if not yet selected in a cluster
        if not pd.isna(df_proc.at[i, 'case']):
            continue
            
        # ---------------------------------------------------------
        # Step 1: Find points in a square centered at the point
        # ---------------------------------------------------------
        current_lat = df_proc.at[i, 'lat']
        current_lon = df_proc.at[i, 'lon']
        
        # We only search among points that have NO case assigned yet
        unassigned_indices = df_proc.index[df_proc['case'].isna()].tolist()
        
        candidates_1 = get_points_in_square(current_lat, current_lon, unassigned_indices)
        
        if not candidates_1:
            continue

        # ---------------------------------------------------------
        # Step 2: Find barycenter and repeat
        # ---------------------------------------------------------
        subset_1 = df_proc.loc[candidates_1]
        bary_lat_1 = subset_1['lat'].mean()
        bary_lon_1 = subset_1['lon'].mean()
        
        # Search again centered at the barycenter
        candidates_2 = get_points_in_square(bary_lat_1, bary_lon_1, unassigned_indices)
        
        # ---------------------------------------------------------
        # Step 3: Check for changes and do a 3rd time if needed
        # ---------------------------------------------------------
        final_candidates = candidates_2
        
        # Compare sets of indices
        if set(candidates_1) != set(candidates_2):
            if candidates_2: # Ensure we didn't drift into empty space
                subset_2 = df_proc.loc[candidates_2]
                bary_lat_2 = subset_2['lat'].mean()
                bary_lon_2 = subset_2['lon'].mean()
                
                candidates_3 = get_points_in_square(bary_lat_2, bary_lon_2, unassigned_indices)
                final_candidates = candidates_3
        
        # ---------------------------------------------------------
        # Step 4: Assign ascending number (case)
        # ---------------------------------------------------------
        if final_candidates:
            df_proc.loc[final_candidates, 'case'] = case_counter
            case_counter += 1

    # ---------------------------------------------------------
    # Step 5: Count points and assign cluster_size
    # ---------------------------------------------------------
    # Value counts gives size of each case
    cluster_counts = df_proc['case'].value_counts()
    
    # Map the counts back to the dataframe
    df_proc['cluster_size'] = df_proc['case'].map(cluster_counts)
    
    print(f"Clustering complete. Found {case_counter - 1} clusters.")
    return df_proc      



df = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\tick tick bloom data\CAML_cyanobacteria_abundance_20211229_R1_date_extended.csv")


#df_out = perform_clustering_with_size(df, 20, 2000)
#df_out.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\tick tick bloom data\junk4.xlsx", index = False)
        
    
#df_out = perform_clustering_with_size(df, 15, 2560)
#df_out.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\tick tick bloom data\clustered_data_15days_2560m.xlsx", index = False)
 

df_out = perform_spatial_clustering_improved(df, 2560) 
df_out.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\tick tick bloom data\clustered_only_square_2560m.xlsx", index = False)