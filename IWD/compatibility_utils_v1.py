# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 12:50:11 2025

@author: K. Pikounis


compatibility_utils_v1.py

Logic Flow:
1. Cluster Measurement Points -> Define 'Targets' (Centroids).
2. Run CyFi on Water Pixels around Targets (Targeted Densification).
3. Apply Unit Conversion (CyFi / 1000).
4. Calculate Probabilistic Compatibility (Mahalanobis or Wasserstein).
"""

import os
import json
import math
import shutil
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wasserstein_distance

import rasterio
import rioxarray
from rasterio.transform import rowcol
from pyproj import Transformer

# CyFi imports
from cyfi.pipeline import CyFiPipeline
from cyfi.config import FeaturesConfig
from cyfi.data.features import generate_all_features
from cyfi.cli import DEFAULT_MODEL_PATH


# =============================================================================
# HELPER: GEOMETRY & CLUSTERING
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance in meters between two lat/lon points."""
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def cluster_points(points_data, Rx=300):
    """
    Groups points that are within Rx meters of each other.
    Returns a list of 'Targets'. Each Target is a dict with:
      - 'centroid_lat', 'centroid_lon'
      - 'points': [list of original point dicts in this cluster]
    """
    if not points_data: return []
    
    # 1. Simple Greedy Clustering
    clusters = []
    processed_indices = set()
    
    for i, pt_a in enumerate(points_data):
        if i in processed_indices: continue
        
        current_cluster = [pt_a]
        processed_indices.add(i)
        
        # Check all other points against this seed
        for j, pt_b in enumerate(points_data):
            if j in processed_indices: continue
            
            dist = haversine_distance(pt_a['lat'], pt_a['lon'], pt_b['lat'], pt_b['lon'])
            if dist <= Rx:
                current_cluster.append(pt_b)
                processed_indices.add(j)
        
        clusters.append(current_cluster)
        
    # 2. Calculate Centroids
    targets = []
    for cluster in clusters:
        lats = [p['lat'] for p in cluster]
        lons = [p['lon'] for p in cluster]
        
        targets.append({
            "centroid_lat": sum(lats) / len(lats),
            "centroid_lon": sum(lons) / len(lons),
            "points": cluster
        })
        
    return targets


# =============================================================================
# STEP B: TARGETED PREDICTION (Run CyFi on Centroids)
# =============================================================================

def predict_for_targets(input_folder_path, targets, Rx=300, Nmin=30):
    """
    Runs CyFi 'Targeted Densification' around the defined Target Centroids.
    Includes optimization to skip CyFi if insufficient water pixels are found.
    """
    input_dir = Path(input_folder_path).resolve()
    
    # 1. Config & Setup
    features_config = FeaturesConfig()
    required_bands = features_config.use_sentinel_bands 
    
    # Load SCL (Essential for Water Mask)
    try:
        scl_path = input_dir / "SCL_raw.tif"
        if not scl_path.exists(): return None
        
        scl_da = rioxarray.open_rasterio(scl_path).squeeze()
        scl_values = scl_da.values
        height, width = scl_da.shape
        transform = scl_da.rio.transform()
        crs = scl_da.rio.crs
    except Exception as e:
        print(f"Error loading SCL: {e}")
        return None

    # Load Other Bands (Lazy Load)
    data_store = {"SCL": scl_values}
    for band in required_bands:
        if band == "SCL": continue
        path = input_dir / f"{band}_raw.tif"
        if path.exists():
            data_store[band] = rioxarray.open_rasterio(path).squeeze().values
        else:
            data_store[band] = np.full((height, width), np.nan, dtype=np.float32)

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    
    # 2. Prepare Batch for CyFi
    temp_cache = Path(tempfile.mkdtemp(prefix="cyfi_targeted_"))
    cache_subdir = temp_cache / f"sentinel_{features_config.image_feature_meter_window}"
    fake_item_id = "TARGET_ITEM"
    
    valid_samples = [] # For DataFrame
    batch_map = {}     # sample_id -> target_index
    
    sigma = Rx / 2.0   # Gaussian Kernel Sigma

    # Iterate Targets (Centroids)
    for t_idx, target in enumerate(targets):
        c_lat, c_lon = target['centroid_lat'], target['centroid_lon']
        
        # Convert Lat/Lon to Pixel
        cx, cy = transformer.transform(c_lon, c_lat)
        r_center, c_center = rowcol(transform, cx, cy)
        
        # Define Window
        radius_px = int(math.ceil(Rx / 10.0))
        r_min = max(0, r_center - radius_px)
        r_max = min(height, r_center + radius_px + 1)
        c_min = max(0, c_center - radius_px)
        c_max = min(width, c_center + radius_px + 1)
        
        # Slice SCL to speed up loop
        scl_sub = data_store["SCL"][r_min:r_max, c_min:c_max]
        
        local_count = 0
        
        for r_local in range(scl_sub.shape[0]):
            for c_local in range(scl_sub.shape[1]):
                r_global = r_min + r_local
                c_global = c_min + c_local
                
                # WATER CHECK (SCL == 6)
                if data_store["SCL"][r_global, c_global] != 6:
                    continue
                
                # DISTANCE CHECK (Real world meters from Centroid)
                dist_px = math.sqrt((r_global - r_center)**2 + (c_global - c_center)**2)
                dist_m = dist_px * 10.0
                if dist_m > Rx: continue
                
                # --- Valid Pixel ---
                sample_id = f"t{t_idx}_p{local_count}"
                local_count += 1
                
                # Save NPY Context Window
                item_dir = cache_subdir / sample_id / fake_item_id
                item_dir.mkdir(parents=True, exist_ok=True)
                
                ctx_rad = features_config.image_feature_meter_window // 10 // 2
                cr_min = max(0, r_global - ctx_rad)
                cr_max = min(height, r_global + ctx_rad)
                cc_min = max(0, c_global - ctx_rad)
                cc_max = min(width, c_global + ctx_rad)
                
                if (cr_max - cr_min) < (ctx_rad*2) or (cc_max - cc_min) < (ctx_rad*2):
                    continue # Skip edge pixels
                
                for band in required_bands:
                    arr = data_store[band][cr_min:cr_max, cc_min:cc_max]
                    np.save(item_dir / f"{band}.npy", arr[np.newaxis, :, :])
                
                # Calculate Weight (Relative to Centroid)
                weight = math.exp(-(dist_m**2) / (2 * sigma**2))
                
                valid_samples.append({
                    "sample_id": sample_id,
                    "date": "2020-01-01", # Dummy
                    "latitude": 0, "longitude": 0
                })
                
                batch_map[sample_id] = {
                    "target_idx": t_idx,
                    "weight": weight
                }

    # --- OPTIMIZATION CHECK ---
    if len(valid_samples) < Nmin:
        shutil.rmtree(temp_cache)
        return None

    # 3. Run CyFi
    samples_df = pd.DataFrame(valid_samples).set_index("sample_id")
    sat_meta_df = pd.DataFrame({
        "sample_id": samples_df.index,
        "item_id": [fake_item_id]*len(samples_df),
        "days_before_sample": [0]*len(samples_df),
        "datetime": ["2020-01-01"]*len(samples_df),
        "visual_href": [None]*len(samples_df)
    })
    
    try:
        _, features_df = generate_all_features(
            samples=samples_df, satellite_meta=sat_meta_df, 
            config=features_config, cache_dir=temp_cache
        )
        pipeline = CyFiPipeline.from_disk(DEFAULT_MODEL_PATH)
        pipeline.predict_features = features_df
        pipeline.predict_samples = samples_df
        pipeline._predict_model()
        results_df = pipeline.output_df 
    except Exception as e:
        print(f"CyFi Error: {e}")
        shutil.rmtree(temp_cache)
        return None

    # 4. Aggregate Results by Target
    output_data = {} 
    
    for sample_id, row in results_df.iterrows():
        if sample_id not in batch_map: continue
        info = batch_map[sample_id]
        idx = info['target_idx']
        
        if idx not in output_data:
            output_data[idx] = {'predictions': [], 'weights': []}
            
        #UNIT CONVERSION: Divide by 1000
        converted_val = float(row['density_cells_per_ml']) / 1000.0
        
        output_data[idx]['predictions'].append(converted_val)
        output_data[idx]['weights'].append(info['weight'])
        
    shutil.rmtree(temp_cache)
    return output_data


# =============================================================================
# STEP C: MAIN WRAPPER (Probabilistic Logic)
# =============================================================================

def calculate_compatibility_score(input_folder_path, Rx=300, Nmin=30):
    """
    Main function to calculate Incompatibility Score for a Case Folder.
    """
    input_dir = Path(input_folder_path).resolve()
    metadata_path = input_dir / "metadata.json"
    
    if not metadata_path.exists(): return None
    
    with open(metadata_path, 'r') as f: meta = json.load(f)
    points = meta.get("points_data", [])
    if not points: return None

    # 1. CLUSTER POINTS (Step A)
    targets = cluster_points(points, Rx)
    if not targets: return None
    
    # 2. RUN TARGETED PREDICTION (Step B)
    # Returns map: target_idx -> {predictions, weights}
    sat_results = predict_for_targets(input_folder_path, targets, Rx, Nmin)
    
    if not sat_results: return None 
    
    global_scores = []
    
    # 3. CALCULATE METRICS (Step C)
    for idx, target in enumerate(targets):
        
        # Check if we have satellite data for this target
        if idx not in sat_results: continue
        
        preds = np.array(sat_results[idx]['predictions'])
        weights = np.array(sat_results[idx]['weights'])
        
        if len(preds) < Nmin: continue 
        
        # --- Satellite Distribution Stats ---
        # Log-Normal Assumption (Predictions are already /1000 here)
        log_preds = np.log10(preds + 1)
        
        # Weighted Mean & Sigma
        sum_w = np.sum(weights)
        weights_norm = weights / sum_w
        
        mu_sat = np.sum(log_preds * weights_norm)
        var_sat = np.sum(weights_norm * (log_preds - mu_sat)**2)
        sigma_sat = np.sqrt(var_sat)
        sigma_sat = max(sigma_sat, 0.15) # Noise floor
        
        # --- Lab Data ---
        lab_points = target['points']
        lab_log_values = [math.log10(max(0.1, float(p['abun']))) for p in lab_points]
        
        score = 0.0
        
        # LOGIC BRANCH
        if len(lab_log_values) == 1:
            # Single Point -> Mahalanobis (Z-Score)
            score = abs(lab_log_values[0] - mu_sat) / sigma_sat
            
        else:
            # Cluster -> Wasserstein Distance
            WD = wasserstein_distance(
                lab_log_values,
                log_preds,
                u_weights=None,      
                v_weights=weights_norm 
            )
            score = WD / sigma_sat
            
        global_scores.append(score)

    if not global_scores: return None
    
    # Return Max Incompatibility found in this folder
    return max(global_scores)