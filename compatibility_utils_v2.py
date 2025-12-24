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
    except SystemExit as e:
        print(f"   > CyFi triggered SystemExit (likely no valid data): {e}")
        shutil.rmtree(temp_cache)
        return None
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
def calculate_compatibility_score(input_folder_path, Rx=300, Nmin=50):
    """
    Calculates Probabilistic Incompatibility Scores and returns detailed cluster analysis.
    
    Returns:
        dict: {
            "min_case_score": float or None,
            "clusters": [
                {
                    "centroid_lat": float,
                    "centroid_lon": float,
                    "cluster_incompatibility_score": float, # Wasserstein (if >1 pt) or Z-score
                    "most_compatible_uid": str,
                    "most_compatible_abun": float,
                    "most_compatible_score": float, # The Z-score of the best point
                    "points": [
                        {
                            "uid": str, "lat": float, "lon": float, 
                            "date": str, "abun": float, "severity": str,
                            "compatibility_score": float # Individual Z-score
                        },
                        ... # Sorted by compatibility_score (ascending)
                    ]
                },
                ...
            ]
        }
    """
    input_dir = Path(input_folder_path).resolve()
    metadata_path = input_dir / "metadata.json"
    
    # Default consistent return structure
    result_summary = {
        "min_case_score": None,
        "clusters": []
    }
    
    if not metadata_path.exists(): 
        return result_summary
    
    try:
        with open(metadata_path, 'r') as f: meta = json.load(f)
    except:
        return result_summary
        
    points = meta.get("points_data", [])
    if not points: 
        return result_summary

    # 1. CLUSTER POINTS
    targets = cluster_points(points, Rx)
    if not targets: 
        return result_summary
    
    # 2. RUN TARGETED PREDICTION
    sat_results = predict_for_targets(input_folder_path, targets, Rx, Nmin)
    if not sat_results: 
        return result_summary
    
    global_scores = []
    
    # 3. EVALUATE EACH CLUSTER
    for idx, target in enumerate(targets):
        
        # Check if we have satellite data for this target
        if idx not in sat_results: continue
        
        preds = np.array(sat_results[idx]['predictions'])
        weights = np.array(sat_results[idx]['weights'])
        
        if len(preds) < Nmin: continue 
        
        # --- Satellite Distribution Stats ---
        # Log-Normal Assumption (Predictions are already /1000 from predict_for_targets)
        log_preds = np.log10(preds + 1)
        
        sum_w = np.sum(weights)
        weights_norm = weights / sum_w
        
        mu_sat = np.sum(log_preds * weights_norm)
        var_sat = np.sum(weights_norm * (log_preds - mu_sat)**2)
        sigma_sat = np.sqrt(var_sat)
        sigma_sat = max(sigma_sat, 0.15) # Noise floor
        
        # --- Lab Data Processing & Individual Scoring ---
        lab_points_data = target['points']
        lab_log_values = []
        points_detailed = []
        
        for p in lab_points_data:
            p_abun = max(0.1, float(p.get('abun', 0)))
            p_log = math.log10(p_abun)
            lab_log_values.append(p_log)
            
            # Calculate INDIVIDUAL Score (Z-Score against Sat Distribution)
            indiv_score = abs(p_log - mu_sat) / sigma_sat
            
            points_detailed.append({
                "uid": p.get('uid'),
                "lat": p.get('lat'),
                "lon": p.get('lon'),
                "date": p.get('date'),
                "abun": p_abun,
                "severity": p.get('severity'),
                "compatibility_score": indiv_score
            })
            
        # Sort points by compatibility (Lowest score = Best match)
        points_detailed.sort(key=lambda x: x['compatibility_score'])
        
        # --- Cluster-Level Metric ---
        cluster_score = 0.0
        
        if len(lab_log_values) == 1:
            # Single Point -> Cluster score is same as point score
            cluster_score = points_detailed[0]['compatibility_score']
        else:
            # Multi Point -> Wasserstein Distance
            WD = wasserstein_distance(
                lab_log_values,
                log_preds,
                u_weights=None,      
                v_weights=weights_norm 
            )
            cluster_score = WD / sigma_sat
            
        global_scores.append(cluster_score)
        
        # --- Build Cluster Dict ---
        best_pt = points_detailed[0]
        
        cluster_info = {
            "centroid_lat": target['centroid_lat'],
            "centroid_lon": target['centroid_lon'],
            "cluster_incompatibility_score": cluster_score,
            "most_compatible_uid": best_pt['uid'],
            "most_compatible_abun": best_pt['abun'],
            "most_compatible_score": best_pt['compatibility_score'],
            "points": points_detailed
        }
        
        result_summary["clusters"].append(cluster_info)

    if global_scores:
        result_summary["min_case_score"] = min(global_scores)
    
    return result_summary


def calculate_spatial_roughness(csv_path, min_neighbors=3, outlier_threshold=0.5):
    """
    Calculates spatial roughness metrics from a CyFi predictions CSV.
    
    This function assesses if the satellite predictions are spatially smooth (expected for water)
    or "noisy/strange" (unexpected). It works on the sparse lattice grid (100m spacing).

    Args:
        csv_path (str): Path to the cyfi_lattice_predictions.csv file.
        min_neighbors (int): Minimum number of neighbors required to evaluate a point.
                             - Standard 3x3 grid has 8 neighbors.
                             - Narrow rivers might only have 2 (upstream/downstream).
                             - We set default to 3 to ensure we have enough 'context' to judge.
                             - Points with < 3 neighbors are marked 'isolated' and skipped.
        outlier_threshold (float): Difference in Log10 units to consider a point an "Outlier".
                                   - 0.5 log units approx equals a 3.16x difference in raw magnitude.
                                   - e.g. Neighbor Median = 10,000, Point = 32,000 -> Outlier.

    Returns:
        dict: A dictionary containing:
            - "roughness_median": The typical local variance (Lower is better/smoother).
            - "outlier_fraction": Percentage of points that are spatially inconsistent.
            - "status": "OK", "Too Sparse" (if river is too narrow), or "Error".
    """
    try:
        # 1. Load Data
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"status": f"Error: {e}", "roughness_median": None, "outlier_fraction": None}
    
    # Validation
    required_cols = ['pixel_row', 'pixel_col', 'density_cells_per_ml']
    if df.empty or not all(col in df.columns for col in required_cols):
        return {"status": "Invalid CSV Format", "roughness_median": None, "outlier_fraction": None}

    # 2. Log-Transformation
    # We work in Log-space because biological quantities vary by orders of magnitude.
    # A difference of 500 cells is huge if the baseline is 100, but noise if baseline is 1,000,000.
    # Adding +1 avoids log(0) errors.
    # Note: Whether the input is /1000 or raw cells doesn't matter for the *difference* logic,
    # as log(A/1000) - log(B/1000) is the same as log(A) - log(B).
    df['log_val'] = np.log10(df['density_cells_per_ml'] + 1)
    
    # 3. Create Fast Spatial Lookup
    # Since the grid is regular (every 10 pixels), we map (Row, Col) -> Value
    # This allows O(1) lookup of neighbors instead of searching arrays.
    grid_map = dict(zip(zip(df['pixel_row'], df['pixel_col']), df['log_val']))
    
    local_diffs = []
    skipped_isolated_count = 0
    
    # 4. Define Neighborhood (The 10x10 Lattice)
    # The grid stride is 10 pixels. So neighbors are at offsets +/- 10.
    # We look at the 8 surrounding points (Moore Neighborhood).
    offsets = [
        (-10, -10), (-10, 0), (-10, 10),
        (0, -10),             (0, 10),
        (10, -10),  (10, 0),  (10, 10)
    ]
    
    # 5. Iterate over every predicted point
    for (r, c), current_val in grid_map.items():
        
        # Collect values of valid existing neighbors
        neighbor_vals = []
        for dr, dc in offsets:
            neighbor_key = (r + dr, c + dc)
            if neighbor_key in grid_map:
                neighbor_vals.append(grid_map[neighbor_key])
        
        # 6. Check Logic for Sparse Areas (e.g., Narrow Rivers)
        # If a point has 0, 1, or 2 neighbors, it's essentially forming a line or is isolated.
        # Calculating a "median" from 1 neighbor is risky (if that neighbor is wrong, we flag this one).
        # We skip these points to avoid false positives.
        if len(neighbor_vals) < min_neighbors:
            skipped_isolated_count += 1
            continue
            
        # 7. Calculate Local Difference
        # We use Median because it is robust to outliers.
        # If we used Mean, one bad neighbor would drag the mean and make the *good* center point look bad.
        local_median = np.median(neighbor_vals)
        diff = abs(current_val - local_median)
        local_diffs.append(diff)
        
    # 8. Aggregation & Metrics
    total_valid_comparisons = len(local_diffs)
    
    # Case: Narrow River / Disconnected Puddles
    if total_valid_comparisons < 5:
        # If we have almost no points with enough neighbors, we declare the geometry "Too Sparse".
        # This prevents returning a metric based on just 1 or 2 lucky points.
        return {
            "roughness_median": 0.0,
            "outlier_fraction": 0.0,
            "num_points":total_valid_comparisons,
            "num_outliers":0,
            "status": "Too Sparse / Narrow Geometry"
        }

    local_diffs = np.array(local_diffs)
    
    # Metric A: Median Roughness
    # The "typical" difference between a point and its neighbors.
    # < 0.2 is very smooth. > 0.4 is generally noisy.
    median_roughness = np.median(local_diffs)
    
    # Metric B: Outlier Fraction
    # How many points deviate by more than the threshold (0.5 log units)?
    num_outliers = np.sum(local_diffs > outlier_threshold)
    fraction_outliers = num_outliers / total_valid_comparisons
    
    return {
        "roughness_median": round(median_roughness, 4),
        "outlier_fraction": round(fraction_outliers, 4),
        "num_points":total_valid_comparisons,
        "num_outliers":num_outliers,
        "status": "OK"
    }

# --- Usage Example ---
if __name__ == "__main__":
    # Replace with your actual file path
    path = "cyfi_lattice_predictions.csv" 
    
    results = calculate_spatial_roughness(path)
    
    print("-" * 30)
    print("SPATIAL ROUGHNESS REPORT")
    print("-" * 30)
    print(f"Status:           {results['status']}")
    if results['status'] == "OK":
        print(f"Median Roughness: {results['roughness_median']} (Log10 units)")
        print(f"Outlier Fraction: {results['outlier_fraction']*100:.1f}%")
        print(f"Details:          {results.get('details')}")
        
        # Interpretation
        if results['outlier_fraction'] > 0.15:
            print("\n[FLAG] WARNING: High spatial inconsistency detected.")
        elif results['roughness_median'] > 0.4:
            print("\n[FLAG] WARNING: Image appears globally noisy.")
        else:
            print("\n[OK] Image is spatially smooth.")