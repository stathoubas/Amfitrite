# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 17:35:50 2025

@author: K. Pikounis


differences compared to v2
1) extract_and_save_tile: considers only the points provided by the points_df
   it does not expand the window to search for other additioanl poins from the
   whole dataset
2) update_excel_report: creates an additiona column to Master_Reoprt excel "abun_list"
   with all abun from all points in the matadata json file  
   
"""

import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import tempfile
import shutil
from tqdm import tqdm

import rasterio
import rioxarray
from rasterio.transform import rowcol
import geopandas as gpd

from pyproj import CRS, Transformer, Geod

from PIL import Image as PILImage

# CyFi imports
from cyfi.pipeline import CyFiPipeline
from cyfi.config import FeaturesConfig
from cyfi.data.features import generate_all_features
from cyfi.cli import DEFAULT_MODEL_PATH

# images in excel
import io

import planetary_computer as pc
from pystac_client import Client
# Need the Planetary Computer client (pc) for signing assets, assuming it's imported elsewhere
# If not, add: import pystac_client; import planetary_computer as pc 


def get_bounding_box(latitude, longitude, meter_buffer=50000):
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box [min_lon, min_lat, max_lon, max_lat] around the point.
    """
    g = Geod(ellps='WGS84')
    # Forward calculation: (lon, lat, back_az)
    lon_west, _, _ = g.fwd(longitude, latitude, 270, meter_buffer)
    _, lat_south, _ = g.fwd(longitude, latitude, 180, meter_buffer)
    lon_east, _, _ = g.fwd(longitude, latitude, 90, meter_buffer)
    _, lat_north, _ = g.fwd(longitude, latitude, 0, meter_buffer)
    return [lon_west, lat_south, lon_east, lat_north]

def get_date_range(date, time_buffer_days=15):
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will include the sample date
    and time_buffer_days days prior

    Returns a string"""
    datetime_format = "%Y-%m-%d"
    range_start = pd.to_datetime(date) - timedelta(days=time_buffer_days)
    range_end = pd.to_datetime(date) + timedelta(days=time_buffer_days)
    date_range = f"{range_start.strftime(datetime_format)}/{range_end.strftime(datetime_format)}"

    return date_range

def predict_using_cyfi_pipeline(input_folder_path, 
                                date_str, 
                                metadata_filename="metadata.json",
                                print_images = False):
    input_dir = Path(input_folder_path).resolve()
    print(f"--- Starting CyFi Pipeline Integration (Cloud Threshold 7.5%): {input_dir} ---")

    # 1. Configuration
    features_config = FeaturesConfig()
    CLOUD_THRESHOLD = 0.075 
    features_config.max_cloud_percent = CLOUD_THRESHOLD

    WINDOW_METERS = features_config.image_feature_meter_window 
    PIXEL_SIZE = 10
    WINDOW_PIXELS = WINDOW_METERS // PIXEL_SIZE 
    RADIUS_PIXELS = WINDOW_PIXELS // 2 

    # 2. Load Raw Raster Data & Metadata
    required_bands = features_config.use_sentinel_bands 
    data_store = {}
    
    try:
        scl_da = rioxarray.open_rasterio(input_dir / "SCL_raw.tif").squeeze()
        data_store["SCL"] = scl_da.values
        height, width = scl_da.shape
        transform = scl_da.rio.transform()
        crs = scl_da.rio.crs
    except FileNotFoundError:
        print("Error: SCL_raw.tif not found. Run Part 1 first.")
        return (False, "")

    # Load Metadata (to retrieve reference points)
    metadata_path = input_dir / metadata_filename
    metadata = {}
    all_reference_points = []
    
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            all_reference_points.extend(metadata.get("points_data", []))
            all_reference_points.extend(metadata.get("additional_points", []))
            
    except FileNotFoundError:
        print(f"Warning: {metadata_filename} not found.")

    # Load other bands
    for band in required_bands:
        if band == "SCL": continue
        path = input_dir / f"{band}_raw.tif"
        if path.exists():
            data_store[band] = rioxarray.open_rasterio(path).squeeze().values
        else:
            data_store[band] = np.full((height, width), np.nan, dtype=np.float32)

    # 3. Generate Lattice Grid
    print("Generating 100m Lattice Grid...")
    GRID_STEP = 10 
    
    rows = np.arange(0, height, GRID_STEP)
    cols = np.arange(0, width, GRID_STEP)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
    grid_rows = grid_rows.flatten()
    grid_cols = grid_cols.flatten()
    
    valid_mask = (grid_rows < height) & (grid_cols < width)
    grid_rows = grid_rows[valid_mask]
    grid_cols = grid_cols[valid_mask]
    
    # Check water mask
    water_mask = (data_store["SCL"][grid_rows, grid_cols] == 6)
    target_rows = grid_rows[water_mask]
    target_cols = grid_cols[water_mask]
    
    initial_samples = len(target_rows)
    print(f"Identified {initial_samples} potential water points.")
    if initial_samples == 0: return (False, "")

    # 4. Create Mock Cache Structure
    temp_cache = Path(tempfile.mkdtemp(prefix="cyfi_lattice_"))
    
    cache_subdir = temp_cache / f"sentinel_{WINDOW_METERS}"
    fake_item_id = "LATTICE_ITEM" 

    valid_sample_ids = []
    valid_lats = []
    valid_lons = []
    valid_rows = []
    valid_cols = []

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # 5. Filter & Slice Data
    print("Checking Cloud Cover & Slicing Data...")
    
    skipped_clouds = 0
    
    for idx, (r, c) in enumerate(tqdm(zip(target_rows, target_cols), total=initial_samples)):
        s_id = f"sample_{idx}"
        
        r_min = max(0, r - RADIUS_PIXELS)
        r_max = min(height, r + RADIUS_PIXELS)
        c_min = max(0, c - RADIUS_PIXELS)
        c_max = min(width, c + RADIUS_PIXELS)
        
        if (r_max - r_min) == 0 or (c_max - c_min) == 0:
            continue

        scl_window = data_store["SCL"][r_min:r_max, c_min:c_max]
        cloud_pixel_count = ((scl_window >= 7) & (scl_window <= 10)).sum()
        total_pixels = scl_window.size
        cloud_ratio = cloud_pixel_count / total_pixels
        
        if cloud_ratio > CLOUD_THRESHOLD:
            skipped_clouds += 1
            continue

        item_dir = cache_subdir / s_id / fake_item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        
        x, y = rasterio.transform.xy(transform, r, c)
        lon, lat = transformer.transform(x, y)
        
        valid_sample_ids.append(s_id)
        valid_lats.append(lat)
        valid_lons.append(lon)
        valid_rows.append(r)
        valid_cols.append(c)

        for band in required_bands:
            arr_window = data_store[band][r_min:r_max, c_min:c_max]
            arr_reshaped = arr_window[np.newaxis, :, :] 
            np.save(item_dir / f"{band}.npy", arr_reshaped)

    print("\nProcessing Summary:")
    print(f" - Total Potential Points: {initial_samples}")
    print(f" - Skipped (Cloud > {CLOUD_THRESHOLD*100}%): {skipped_clouds}")
    print(f" - Valid Points to Predict: {len(valid_sample_ids)}")

    if len(valid_sample_ids) == 0:
        print("No valid points remained after cloud filtering.")
        shutil.rmtree(temp_cache)
        return (False, "")

    # 6. Prepare DataFrames for CyFi
    samples_df = pd.DataFrame({
        "sample_id": valid_sample_ids,
        "date": [date_str] * len(valid_sample_ids),
        "latitude": valid_lats,
        "longitude": valid_lons
    }).set_index("sample_id")

    satellite_meta_df = pd.DataFrame({
        "sample_id": valid_sample_ids,
        "item_id": [fake_item_id] * len(valid_sample_ids),
        "days_before_sample": [0] * len(valid_sample_ids), 
        "datetime": [date_str] * len(valid_sample_ids), 
        "visual_href": [None] * len(valid_sample_ids) 
    })

    # 7. Run CyFi Feature Generation
    print("Running CyFi Feature Generation...")
    try:
        _, features_df = generate_all_features(
            samples=samples_df,
            satellite_meta=satellite_meta_df,
            config=features_config,
            cache_dir=temp_cache
        )
    except Exception as e:
        print(f"Feature generation failed: {e}")
        shutil.rmtree(temp_cache)
        return (False, "")

    # 8. Run CyFi Prediction
    print("Running Prediction...")
    pipeline = CyFiPipeline.from_disk(DEFAULT_MODEL_PATH)
    pipeline.predict_features = features_df
    pipeline.predict_samples = samples_df
    pipeline._predict_model()
    
    results_df = pipeline.output_df.reset_index()
    results_df["pixel_row"] = valid_rows
    results_df["pixel_col"] = valid_cols
    
    csv_path = input_dir / "cyfi_lattice_predictions.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

    # --- UPDATE METADATA JSON ---
    severity_list = results_df.severity.astype(str).str.lower().to_list()
    
    # [FIX] Define these variables explicitly!
    count_high = severity_list.count("high")
    count_moderate = severity_list.count("moderate")
    count_low = severity_list.count("low")

    metadata["High counts"] = count_high
    metadata["Moderate counts "] = count_moderate
    metadata["Low counts "] = count_low
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Updated metadata saved to: {metadata_path}")

    # 9. Visualization
    # ---------------------------------------------------------
    print("Generating Visualization...")
    try:
        raw_vis_path = input_dir / "visual_raw.tif"
        if raw_vis_path.exists():
            raw_vis = rioxarray.open_rasterio(raw_vis_path).squeeze()
            bg_img = np.moveaxis(raw_vis.values, 0, -1)
            
            vmin, vmax = np.nanpercentile(bg_img, [2, 98])
            bg_img = np.clip((bg_img - vmin) / (vmax - vmin), 0, 1)

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(bg_img)
            
            # 1. PREDICTIONS (Dots)
            severity_colors_pred = {
                'low': 'green', 'moderate': 'orange', 'high': 'red',
                '1': 'green', '2': 'orange', '3': 'red',
                1: 'green', 2: 'orange', 3: 'red'
            }
            results_df['severity'] = results_df['severity'].astype(str)
            colors_pred = results_df['severity'].map(lambda x: severity_colors_pred.get(x.lower(), 'gray'))
            
            ax.scatter(results_df['pixel_col'], results_df['pixel_row'], 
                       c=colors_pred, s=20, alpha=0.9, edgecolors='black', linewidth=0.5, label='Prediction')
            
            # 2. REFERENCE POINTS (X)
            if all_reference_points:
                ref_lats = [p['lat'] for p in all_reference_points]
                ref_lons = [p['lon'] for p in all_reference_points]
                
                ref_colors = []
                for p in all_reference_points:
                    sev = p.get('severity', 3)
                    try:
                        val = int(sev)
                        if val == 1: ref_colors.append('green')
                        elif val == 2: ref_colors.append('orange')
                        elif val == 3: ref_colors.append('red')
                        else: ref_colors.append('red')
                    except:
                        s = str(sev).lower()
                        if 'low' in s: ref_colors.append('green')
                        elif 'mod' in s: ref_colors.append('orange')
                        elif 'high' in s: ref_colors.append('red')
                        else: ref_colors.append('red')

                transformer_inv = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                ox, oy = transformer_inv.transform(ref_lons, ref_lats)
                orows, ocols = rowcol(transform, ox, oy)
                
                ax.scatter(ocols, orows, c=ref_colors, s=150, marker='x', linewidths=3, label='Reference Point')

            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Low', markersize=8),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Moderate', markersize=8),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='High', markersize=8),
                Line2D([0], [0], marker='x', color='red', label='Reference', markersize=8, linestyle='None')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            ax.set_title(f"CyFi Predictions: {date_str} (Cloud < {CLOUD_THRESHOLD*100}%)")
            ax.set_axis_off()
            
            final_img_path = input_dir / "cyfi_prediction_map.png"
            plt.savefig(final_img_path, bbox_inches='tight', dpi=150)
            print(f"Visualization SAVED to: {final_img_path}")
            
            if print_images:
                plt.show()
            plt.close(fig)
            
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

    print("Cleaning up temporary cache...")
    shutil.rmtree(temp_cache)
    
    # 10. Rename Folder
    try:
        new_folder_name = f"{input_dir.name}_H{count_high}_M{count_moderate}_L{count_low}"
        if not input_dir.name.endswith(f"_H{count_high}_M{count_moderate}_L{count_low}"):
            new_folder_path = input_dir.parent / new_folder_name
            input_dir.rename(new_folder_path)
            print(f"Successfully renamed folder to: {new_folder_path}")
            return (True, new_folder_path)
        return (True, input_dir)
        
    except Exception as e:
        print(f"Could not rename folder: {e}")
        # Even if rename fails, return True for folder_ok but empty string for new path
        return (True, str(input_dir))

def compress_image_to_bytes(image_path, target_img_size = (200,200)):
    """
    Reads an image, resizes it, and returns the byte stream.
    """
    try:
        with PILImage.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # Resize (Thumbnail)
            img.thumbnail(target_img_size, PILImage.LANCZOS)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', optimize=True)
            img_byte_arr.seek(0)
            
            return img_byte_arr, img.width, img.height
    except Exception as e:
        return None, 0, 0
      
def find_satelite_images(p_lat, p_lon, sel_date, meter_buffer = 3000):
    catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    
    bbox = get_bounding_box(p_lat, p_lon, meter_buffer)
    print(bbox)
    date_range = get_date_range(sel_date)

    search = catalog.search(
    collections=["sentinel-2-l2a"], bbox=bbox, datetime=date_range)

    items = [item for item in search.get_all_items()]
    
    item_details = pd.DataFrame(
        [
            {
                "datetime": item.datetime.strftime("%Y-%m-%d"),
                "platform": item.properties["platform"],
                "min_long": item.bbox[0],
                "max_long": item.bbox[2],
                "min_lat": item.bbox[1],
                "max_lat": item.bbox[3],
                "bbox": item.bbox,
                "item_obj": item,
            }
            for item in items
        ]
    )

    # check which rows actually contain the sample location
    item_details["contains_sample_point"] = (
        (item_details.min_lat < p_lat)
        & (item_details.max_lat > p_lat)
        & (item_details.min_long < p_lon)
        & (item_details.max_long > p_lon)
    )

    print(
        f"Filtering from {len(item_details)} returned to {item_details.contains_sample_point.sum()} items that contain the sample location"
    )
    
    item_details = item_details[item_details["contains_sample_point"]]
    item_details[["datetime", "platform", "contains_sample_point", "bbox"]].sort_values(
        by="datetime"
    )

    item_details["per_clouds"] = -1
    # the commented code below finds the cloud coverage of the whole image
    '''
    for it, row in item_details.iterrows():
        ar = rioxarray.open_rasterio(pc.sign(row.item_obj.assets["SCL"].href)).to_numpy()
        mask = (ar == 3) | ((ar >= 7) & (ar <= 10))
        cloud_per = round(100 * np.sum(mask) / ar.size, 2)
        item_details.at[it, "per_clouds"] = cloud_per
    '''
    
    # the code below finds the cloud coverage of the box of interest
    for it, row in item_details.iterrows():
        try:
            # 1. Lazy load the SCL asset URL
            scl_href = pc.sign(row.item_obj.assets["SCL"].href)
            
            # 2. Open and Clip to the bounding box (Lazy loading)
            # This ensures we don't download the whole tile, just the area of interest
            ds = rioxarray.open_rasterio(scl_href)
            ds_clip = ds.rio.clip_box(
                minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326"
            )
            
            # 3. Load values into memory (now it's a small 2D array)
            ar = ds_clip.values.squeeze()
            
            # 4. Calculate Cloud Percentage
            # We filter for: 
            #   3: Cloud Shadows
            #   7: Unclassified
            #   8: Cloud Medium Probability
            #   9: Cloud High Probability
            #   10: Cirrus
            # Create a boolean mask
            cloud_mask = (ar == 3) | ((ar >= 7) & (ar <= 10))
            
            # Calculate percentage: (Count of True / Total Pixels) * 100
            cloud_per = round(100 * np.sum(cloud_mask) / ar.size, 2)
            
            item_details.at[it, "per_clouds"] = cloud_per            
        except Exception as e:
            print(f"Error calculating clouds for item {it}: {e}")
    
    item_details['date_difference'] = (pd.to_datetime(item_details['datetime']) - pd.to_datetime(sel_date)).dt.days
    
    return item_details

def select_item(item_details):
    low_clouds = item_details[item_details.per_clouds < 7.5]
    if len(low_clouds) == 0:
        return False, None
    
    low_clouds["abs_date_difference"] = np.abs(low_clouds["date_difference"])
    if len(low_clouds) == 1:
        return low_clouds.iloc[0], True
    
    low_clouds.sort_values(by = "per_clouds", inplace = True)
    if low_clouds.iloc[0]["per_clouds"] < 2.0:
        sel_items = low_clouds[low_clouds["per_clouds"] < 2.0].copy()
        sel_items.sort_values(by = "abs_date_difference", inplace = True)
    elif low_clouds.iloc[0]["per_clouds"] < 5.0:
        sel_items = low_clouds[low_clouds["per_clouds"] < 5.0].copy()
        sel_items.sort_values(by = "abs_date_difference", inplace = True)
    else:
        sel_items = low_clouds.copy()
        sel_items.sort_values(by = "abs_date_difference", inplace = True)
        
    return sel_items.iloc[0], True

def extract_and_save_tile(
    item,
    points_df,
    initial_pixel_size=100,
    save_data=False,
    output_path=None, 
    print_images = True):
    
    print("--- Starting Tile Extraction and Processing (Cluster Only) ---")
    
    # 1. Setup CRS and Center
    target_crs = CRS.from_string(item.properties["proj:code"])
    center_lat = points_df['lat'].mean()
    center_lon = points_df['lon'].mean()
    print(f"Center Point (Lat/Lon): ({center_lat:.4f}, {center_lon:.4f})")
    
    # 2. Calculate Clipping Box in UTM
    points_gdf = gpd.GeoDataFrame(
        points_df, 
        geometry=gpd.points_from_xy(points_df.lon, points_df.lat), 
        crs=CRS.from_string("EPSG:4326")
    ).to_crs(target_crs)
    
    minx_points, miny_points, maxx_points, maxy_points = points_gdf.total_bounds
    center_x = (minx_points + maxx_points) / 2
    center_y = (miny_points + maxy_points) / 2
    
    initial_side_meters = initial_pixel_size * 10
    required_margin_meters = max(maxx_points - minx_points, maxy_points - miny_points) / 2
    final_margin_meters = max(initial_side_meters / 2, required_margin_meters)
    
    min_x_final = center_x - final_margin_meters
    max_x_final = center_x + final_margin_meters
    min_y_final = center_y - final_margin_meters
    max_y_final = center_y + final_margin_meters
    
    final_pixel_side = int((max_x_final - min_x_final) / 10)
    print(f"Final Tile Dimension (10m pixels): {final_pixel_side} x {final_pixel_side}")
    
    clip_bbox_utm = (min_x_final, min_y_final, max_x_final, max_y_final)
    transformer_latlon_to_utm = Transformer.from_crs(CRS.from_string("EPSG:4326"), target_crs, always_xy=True)

    # --- 3. PREPARE METADATA (Cluster Points Only) ---
    cols_to_keep = ['uid', 'lat', 'lon', 'date', 'abun', 'severity']
    if 'severity' not in points_df.columns:
        points_df['severity'] = 3

    def clean_df_for_json(df, cols):
        temp = df[cols].copy()
        if 'date' in temp.columns:
            temp['date'] = temp['date'].astype(str)
        return temp.to_dict('records')

    points_data_list = clean_df_for_json(points_df, cols_to_keep)
    # additional_points_list REMOVED

    # 5. Asset Clipping
    band_gsd_map = {
        "B02": 10, "B03": 10, "B04": 10, "B08": 10, "visual": 10, "AOT": 10, "WVP": 60,
        "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "SCL": 20, "B11": 20, "B12": 20,
        "B01": 60, "B09": 60
    }
    
    clipped_data = {}
    print("--- Clipping and Aligning Rasters ---")
    
    b04_href = pc.sign(item.assets["B04"].href)
    b04_ds = rioxarray.open_rasterio(b04_href)
    b04_clip = b04_ds.rio.clip_box(
        minx=clip_bbox_utm[0], miny=clip_bbox_utm[1], 
        maxx=clip_bbox_utm[2], maxy=clip_bbox_utm[3],
        crs=target_crs
    )
    b04_clip = b04_clip.isel(y=slice(0, final_pixel_side), x=slice(0, final_pixel_side))
    clipped_data["B04"] = b04_clip.squeeze() 
    
    for asset_key, gsd in band_gsd_map.items():
        if asset_key == "B04": continue
        
        asset_href = pc.sign(item.assets[asset_key].href)
        ds = rioxarray.open_rasterio(asset_href)
        
        ds_clip = ds.rio.clip_box(
            minx=clip_bbox_utm[0], miny=clip_bbox_utm[1], 
            maxx=clip_bbox_utm[2], maxy=clip_bbox_utm[3],
            crs=target_crs
        )

        resampling_method = rasterio.enums.Resampling.nearest if asset_key == "SCL" else rasterio.enums.Resampling.bilinear
        ds_aligned = ds_clip.rio.reproject_match(b04_clip, resampling=resampling_method)
        
        if asset_key != "visual":
            clipped_data[asset_key] = ds_aligned.squeeze()
        else:
             clipped_data[asset_key] = ds_aligned
    
    if save_data:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\n--- Saving RAW TIFFs ---")
        for key, da in clipped_data.items():
            da.rio.to_raster(output_dir / f"{key}_raw.tif")

    # 6. Calculate Indices
    SCALE_FACTOR = 10000.0 
    b04_ref = (clipped_data["B04"] / SCALE_FACTOR).astype(np.float32)
    b05_ref = (clipped_data["B05"] / SCALE_FACTOR).astype(np.float32)
    b07_ref = (clipped_data["B07"] / SCALE_FACTOR).astype(np.float32)
    b08_ref = (clipped_data["B08"] / SCALE_FACTOR).astype(np.float32)
    
    sum_b8_b4 = b08_ref + b04_ref
    ndvi = (b08_ref - b04_ref) / sum_b8_b4.where(sum_b8_b4 != 0, np.nan) 
    sum_b7_b5 = b07_ref + b05_ref
    ndci = (b07_ref - b05_ref) / sum_b7_b5.where(sum_b7_b5 != 0, np.nan) 
    
    vis_data = clipped_data.copy()
    vis_data["NDVI"] = ndvi
    vis_data["NDCI"] = ndci

    # 7. Prepare Plotting Coordinates (UTM)
    plot_x, plot_y = transformer_latlon_to_utm.transform(points_df.lon.values, points_df.lat.values)
    
    def get_color(sev):
        try:
            val = int(sev)
            if val == 1: return 'green'
            if val == 2: return 'orange'
            if val == 3: return 'red'
        except:
            s = str(sev).lower()
            if 'low' in s: return 'green'
            if 'mod' in s: return 'orange'
            if 'high' in s: return 'red'
        return 'red'
        
    plot_colors = points_df['severity'].apply(get_color).tolist()

    # 8. Save Visual Previews
    if save_data:
        print("--- Saving Visual Previews ---")
        
        colors = ['white', 'white', 'black', 'black', 'green', 'saddlebrown', 'lightblue', 'grey', 'grey', 'grey', 'grey']
        cmap_scl = ListedColormap(colors)
        bounds = np.arange(12)
        norm_scl = BoundaryNorm(bounds, cmap_scl.N)

        def save_plot_with_points(da, key, cmap=None, norm=None, vmin=None, vmax=None, rgb=False):
            fig, ax = plt.subplots(figsize=(10, 10))
            extent_utm = [da.x.min(), da.x.max(), da.y.min(), da.y.max()]
            
            if rgb:
                arr = da.transpose('y', 'x', 'band').values
                vmin, vmax = np.nanpercentile(arr, [2, 98])
                arr_scaled = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
                ax.imshow(arr_scaled, extent=extent_utm, origin='upper')
            elif key == "SCL":
                ax.imshow(da.values, cmap=cmap, norm=norm, extent=extent_utm, origin='upper')
            else:
                robust = True if vmin is None else False
                arr_2d = da.values.squeeze()
                if robust:
                   vmin, vmax = np.nanpercentile(arr_2d, [2, 98])
                ax.imshow(arr_2d, cmap=cmap, extent=extent_utm, origin='upper', vmin=vmin, vmax=vmax)
            
            ax.scatter(plot_x, plot_y, c=plot_colors, s=150, marker='x', linewidths=3, zorder=10)
            ax.set_axis_off() 
            plt.savefig(output_dir / f"{key}_preview.png", bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        for key in vis_data.keys():
            da = vis_data[key]
            if key == "visual": save_plot_with_points(da, key, rgb=True)
            elif key == "SCL": save_plot_with_points(da, key, cmap=cmap_scl, norm=norm_scl)
            elif key == "NDVI": save_plot_with_points(da, key, cmap="RdYlGn", vmin=-1, vmax=1)
            elif key == "NDCI": save_plot_with_points(da, key, cmap="jet", vmin=-1, vmax=1)
            else: save_plot_with_points(da, key, cmap="gray")

        metadata = {
            "item_id": item.id,
            "per_clouds": item.properties.get("per_clouds", "N/A"),
            "uid": points_df.iloc[0]["uid"], 
            "abun": max(points_df["abun"].to_list()),
            "tile_size_10m_pixels": final_pixel_side,
            "date": item.properties["datetime"].split('T')[0],
            "center_lat": center_lat,
            "center_lon": center_lon,
            "points_data": points_data_list
            # "additional_points" REMOVED
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Data successfully saved to: {output_dir.resolve()}")

def update_excel_report(folder_path, excel_path="Master_Report.xlsx", uids_tracker_path="uids_processes.xlsx"):
    folder = Path(folder_path).resolve()
    excel_file = Path(excel_path).resolve()
    uids_file = Path(uids_tracker_path).resolve()
    
    print(f"--- Updating Report & UID Tracker ---")
    
    try:
        with open(folder / "metadata.json", "r") as f:
            meta = json.load(f)
    except FileNotFoundError:
        print(f"Error: metadata.json not found in {folder}")
        return

    # --- PART A: UID TRACKER ---
    print(f"--- Updating UID Tracker: {uids_file} ---")
    current_uids = set()
    points_data = meta.get("points_data", [])
    
    # Extract Abundance List for Report
    abun_list_raw = []
    
    if points_data:
        for p in points_data:
            if "uid" in p:
                current_uids.add(str(p["uid"]))
            if "abun" in p:
                abun_list_raw.append(p["abun"])

    # Sort abundance from Largest to Smallest
    abun_list_raw.sort(reverse=True)
    abun_list_str = str(abun_list_raw)

    if current_uids:
        if uids_file.exists():
            try:
                df_uids = pd.read_excel(uids_file)
                existing_uids = set(df_uids['uid'].astype(str))
            except Exception as e:
                df_uids = pd.DataFrame(columns=['uid'])
                existing_uids = set()
        else:
            df_uids = pd.DataFrame(columns=['uid'])
            existing_uids = set()

        new_uids_list = [uid for uid in current_uids if uid not in existing_uids]
        
        if new_uids_list:
            new_df = pd.DataFrame({'uid': new_uids_list})
            df_final = pd.concat([df_uids, new_df], ignore_index=True)
            df_final.to_excel(uids_file, index=False)
            print(f"Saved updated UIDs to {uids_file}")

    # --- PART B: MASTER REPORT ---
    new_row = {
        "uid": meta.get("uid", "N/A"),
        "case": int(folder.name.split("_")[0][4:]) if folder.name.startswith("case") else "N/A",
        "item_id": meta.get("item_id", "N/A"),
        "date": meta.get("date", "N/A"),
        "lat": meta.get("center_lat", "N/A"),
        "lon": meta.get("center_lon", "N/A"),
        "abun": meta.get("abun", "N/A"),
        "abun_list": abun_list_str,
        "per_clouds": meta.get("per_clouds", "N/A"),
        "high_pred": meta.get("High counts", -1),
        "mod_pred": meta.get("Moderate counts ", -1),
        "low_pred": meta.get("Low counts ", -1),
        "NDVI": "", 
        "NCVI": "",
        "pred_vislual": "",
        "source_path": str(folder)
    }
    
    if excel_file.exists():
        df = pd.read_excel(excel_file)
        if 'source_path' not in df.columns: df['source_path'] = ""
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    try:
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Report')
            workbook = writer.book
            worksheet = writer.sheets['Report']
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1})
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_fmt)

            image_map = {
                "NDVI": "NDVI_preview.png",
                "NCVI": "NDCI_preview.png",
                "pred_vislual": "cyfi_prediction_map.png"
            }
            columns_list = list(df.columns)
            
            for index, row in df.iterrows():
                excel_row_idx = index + 1
                row_folder_str = row.get('source_path')
                if not row_folder_str or pd.isna(row_folder_str): continue
                row_path = Path(row_folder_str)
                worksheet.set_row_pixels(excel_row_idx, 150)

                for col_name, filename in image_map.items():
                    if col_name in columns_list:
                        col_idx = columns_list.index(col_name)
                        img_path = row_path / filename
                        if img_path.exists():
                            img_data, width, height = compress_image_to_bytes(img_path, (200, 200))
                            if img_data:
                                worksheet.embed_image(excel_row_idx, col_idx, filename, {
                                    'image_data': img_data, 'object_position': 1
                                })
                                worksheet.set_column_pixels(col_idx, col_idx, width + 10)

        print(f"Report Rebuilt Successfully: {excel_file}")
    except Exception as e:
        print(f"An error occurred while writing the report: {e}")