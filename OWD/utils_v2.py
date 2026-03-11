# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:36:39 2026

@author: K. Pikounis

based on utils_v5 of IWD folder
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import shutil
from tqdm import tqdm

import rasterio
import rioxarray
from rioxarray.exceptions import NoDataInBounds

from pyproj import CRS, Transformer

# CyFi imports
from cyfi.pipeline import CyFiPipeline
from cyfi.config import FeaturesConfig
from cyfi.data.features import generate_all_features
from cyfi.cli import DEFAULT_MODEL_PATH

# images in excel
import planetary_computer as pc

import os

import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as L
import timm

import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from PIL import Image, ImageDraw, ImageFont

import xarray as xr

def extract_master_scene(
    item,
    lat,
    lon,
    case_id,
    output_path=None, 
):
    """
    Downloads a single massive 11,680m x 11,680m bounding box around the center point.
    (7680m core for the 9 tiles + 4000m buffer for CyFi context).
    Saves the aligned TIFFs to a 'master_scene' subfolder.
    """
    print(f"--- Extracting Master Scene for {case_id} ---")
    
    # 1. Setup CRS and Calculate Center in UTM
    target_crs = CRS.from_string(item.properties["proj:code"])
    print(f"Center Point (Lat/Lon): ({lat:.4f}, {lon:.4f})")
    
    transformer_latlon_to_utm = Transformer.from_crs(CRS.from_string("EPSG:4326"), target_crs, always_xy=True)
    center_x, center_y = transformer_latlon_to_utm.transform(lon, lat)

    # 2. Define Master Bounding Box
    # Total width = 7680m (core) + 4000m (buffer) = 11680m
    # Half side = 5840m
    half_side = 5840.0
    
    min_x = center_x - half_side
    max_x = center_x + half_side
    min_y = center_y - half_side
    max_y = center_y + half_side
    
    clip_bbox_utm = (min_x, min_y, max_x, max_y)

    # 3. Open B04 as the reference grid
    try:
        b04_href = pc.sign(item.assets["B04"].href)
        b04_ds_full = rioxarray.open_rasterio(b04_href)
    except Exception as e:
        print(f"Error opening B04: {e}")
        return False

    band_gsd_map = {
        "B02": 10, "B03": 10, "B04": 10, "B08": 10, "visual": 10, "AOT": 10, "WVP": 60,
        "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "SCL": 20, "B11": 20, "B12": 20,
        "B01": 60, "B09": 60
    }
    
    # Pre-open all assets to save HTTP request overhead
    opened_assets = {}
    for asset_key in band_gsd_map.keys():
        if asset_key == "B04": continue
        href = pc.sign(item.assets[asset_key].href)
        opened_assets[asset_key] = rioxarray.open_rasterio(href)

    base_output_dir = Path(output_path) if output_path else None
    if not base_output_dir:
        print("Error: No output path provided.")
        return False
        
    master_dir = base_output_dir / "master_scene"
    master_dir.mkdir(parents=True, exist_ok=True)
    
    clipped_data = {}
    
    # 4. Clip Reference Band (B04)
    try:
        print("Clipping reference band (B04)...")
        b04_clip = b04_ds_full.rio.clip_box(
            minx=clip_bbox_utm[0], miny=clip_bbox_utm[1], 
            maxx=clip_bbox_utm[2], maxy=clip_bbox_utm[3],
            crs=target_crs
        )
        # Ensure exact pixel dimensions if necessary (11680m / 10m = 1168 pixels)
        b04_clip = b04_clip.isel(y=slice(0, 1168), x=slice(0, 1168))
        clipped_data["B04"] = b04_clip.squeeze() 
        
        # 5. Clip and Align all other bands to B04
        print("Clipping and aligning remaining bands...")
        for asset_key, ds in opened_assets.items():
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
                
    except NoDataInBounds:
        print(f"Error: Bounding box outside bounds for master scene. Skipping.")
        return False
    except Exception as e:
        print(f"An error occurred during clipping master scene: {e}")
        return False

    # 6. Save Data
    print(f"Saving TIFFs to {master_dir}...")
    for key, da in clipped_data.items():
        da.rio.to_raster(master_dir / f"{key}_raw.tif")
        
    # Save a lightweight metadata file so the slicer knows where the exact center is
    master_meta = {
        "item_id": item.id,
        "center_lat": lat,
        "center_lon": lon,
        "center_x_utm": center_x,
        "center_y_utm": center_y,
        "crs": str(target_crs),
        "date": item.properties["datetime"].split('T')[0]
    }
    with open(master_dir / "master_metadata.json", "w") as f:
        json.dump(master_meta, f, indent=4)
            
    print("Master Scene extraction complete!")
    return True

def run_cyfi_and_slice_scene(
    base_folder_path, 
    date_str, 
    metadata_filename="master_metadata.json",
    print_images=True
):
    """
    Runs CyFi on the Master Scene, ensuring no edge effects by utilizing the 4000m buffer.
    After prediction, slices the arrays into 9 tiles, computes H/M/L scores,
    and saves them directly into correctly named folders (e.g., tile_1_H10_M5_L0).
    """
    base_dir = Path(base_folder_path).resolve()
    master_dir = base_dir / "master_scene"
    
    print(f"--- Starting Master Scene CyFi Prediction & Slicing in: {base_dir} ---")
    
    if not master_dir.exists():
        print(f"Error: master_scene folder not found in {base_dir}. Skipping.")
        return []

    # 1. Load Master Metadata
    with open(master_dir / metadata_filename, "r") as f:
        master_meta = json.load(f)
        
    master_crs = CRS.from_string(master_meta["crs"])
    master_center_x = master_meta["center_x_utm"]
    master_center_y = master_meta["center_y_utm"]
    
    # 2. Setup CyFi Configuration
    features_config = FeaturesConfig()
    CLOUD_THRESHOLD = 0.075 
    features_config.max_cloud_percent = CLOUD_THRESHOLD

    WINDOW_METERS = features_config.image_feature_meter_window # usually 2000m
    PIXEL_SIZE = 10
    RADIUS_PIXELS = (WINDOW_METERS // PIXEL_SIZE) // 2 # 100 pixels
    required_bands = features_config.use_sentinel_bands 
    
    # 3. Load Master TIFFs into Memory
    print("Loading Master TIFFs into memory...")
    data_store = {}
    master_height, master_width = 1168, 1168  # 11680m / 10m
    
    for band in required_bands:
        path = master_dir / f"{band}_raw.tif"
        if path.exists():
            ds = rioxarray.open_rasterio(path).squeeze()
            data_store[band] = ds.values
            if band == "SCL":
                transform = ds.rio.transform()
        else:
            data_store[band] = np.full((master_height, master_width), np.nan, dtype=np.float32)
            
    # Also load visual for plotting later
    vis_path = master_dir / "visual_raw.tif"
    if vis_path.exists():
        data_store["visual"] = rioxarray.open_rasterio(vis_path).squeeze().values

    # 4. Generate 100m Lattice Grid strictly inside the 7680m core
    # The master array is 1168x1168. The core is 768x768.
    # Therefore, the core starts at pixel 200 and ends at pixel 968.
    print("Generating 100m Lattice Grid in the core area...")
    GRID_STEP = 10 
    
    core_min_idx = 200
    core_max_idx = 968
    
    rows = np.arange(core_min_idx, core_max_idx, GRID_STEP)
    cols = np.arange(core_min_idx, core_max_idx, GRID_STEP)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
    grid_rows = grid_rows.flatten()
    grid_cols = grid_cols.flatten()
    
    water_mask = (data_store["SCL"][grid_rows, grid_cols] == 6)
    target_rows = grid_rows[water_mask]
    target_cols = grid_cols[water_mask]
    
    initial_samples = len(target_rows)
    print(f"Identified {initial_samples} potential water points in the core.")

    temp_cache = Path(tempfile.mkdtemp(prefix="cyfi_master_lattice_"))
    cache_subdir = temp_cache / f"sentinel_{WINDOW_METERS}"
    fake_item_id = "LATTICE_ITEM" 

    valid_sample_ids, valid_lats, valid_lons, valid_rows, valid_cols = [], [], [], [], []
    transformer = Transformer.from_crs(master_crs, "EPSG:4326", always_xy=True)

    # 5. Extract Windows for CyFi Prediction
    if initial_samples > 0:
        print("Checking Cloud Cover & Slicing Data for CyFi...")
        for idx, (r, c) in enumerate(tqdm(zip(target_rows, target_cols), total=initial_samples)):
            s_id = f"sample_{idx}"
            r_min = r - RADIUS_PIXELS
            r_max = r + RADIUS_PIXELS
            c_min = c - RADIUS_PIXELS
            c_max = c + RADIUS_PIXELS
            
            scl_window = data_store["SCL"][r_min:r_max, c_min:c_max]
            cloud_pixel_count = ((scl_window >= 7) & (scl_window <= 10)).sum()
            if (cloud_pixel_count / scl_window.size) > CLOUD_THRESHOLD:
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
                np.save(item_dir / f"{band}.npy", arr_window[np.newaxis, :, :])

    # 6. Run CyFi Pipeline on Master Scene
    master_results_df = pd.DataFrame()
    if len(valid_sample_ids) > 0:
        print(f" - Valid Points to Predict: {len(valid_sample_ids)}")
        samples_df = pd.DataFrame({
            "sample_id": valid_sample_ids, "date": [date_str] * len(valid_sample_ids),
            "latitude": valid_lats, "longitude": valid_lons
        }).set_index("sample_id")

        satellite_meta_df = pd.DataFrame({
            "sample_id": valid_sample_ids, "item_id": [fake_item_id] * len(valid_sample_ids),
            "days_before_sample": [0] * len(valid_sample_ids), "datetime": [date_str] * len(valid_sample_ids), 
            "visual_href": [None] * len(valid_sample_ids) 
        })

        try:
            print("Running CyFi Feature Generation...")
            _, features_df = generate_all_features(
                samples=samples_df, satellite_meta=satellite_meta_df, config=features_config, cache_dir=temp_cache
            )
            print("Running Prediction Model...")
            pipeline = CyFiPipeline.from_disk(DEFAULT_MODEL_PATH)
            pipeline.predict_features = features_df
            pipeline.predict_samples = samples_df
            pipeline._predict_model()
            
            master_results_df = pipeline.output_df.reset_index()
            master_results_df["pixel_row"] = valid_rows
            master_results_df["pixel_col"] = valid_cols
        except Exception as e:
            print(f"CyFi execution failed: {e}")
            
    shutil.rmtree(temp_cache, ignore_errors=True)

    # 7. Physical Slicing of the 9 Tiles
    print("\n--- Slicing Arrays into 9 Tiles ---")
    offsets = {
        0: (0, 0), 1: (0, 1), 2: (-1, 1), 3: (-1, 0), 4: (-1, -1),
        5: (0, -1), 6: (1, -1), 7: (1, 0), 8: (1, 1)
    }
    
    processed_folders = []
    # In the master array, the absolute center pixel is (584, 584)
    MASTER_CENTER_PIXEL = 584
    TILE_SIZE = 256
    HALF_TILE = 128
    
    for tile_idx, (dx, dy) in offsets.items():
        # Calculate bounds in the master array
        # dx relates to columns. dy relates to rows (Numpy row 0 is North, so +dy goes up/smaller row)
        c_center = MASTER_CENTER_PIXEL + (dx * TILE_SIZE)
        r_center = MASTER_CENTER_PIXEL - (dy * TILE_SIZE)
        
        r_min, r_max = r_center - HALF_TILE, r_center + HALF_TILE
        c_min, c_max = c_center - HALF_TILE, c_center + HALF_TILE
        
        # Extract CyFi Predictions for THIS specific tile
        if not master_results_df.empty:
            tile_preds = master_results_df[
                (master_results_df.pixel_row >= r_min) & (master_results_df.pixel_row < r_max) &
                (master_results_df.pixel_col >= c_min) & (master_results_df.pixel_col < c_max)
            ].copy()
        else:
            tile_preds = pd.DataFrame()
            
        # Count Severity
        if not tile_preds.empty:
            severity_list = tile_preds.severity.astype(str).str.lower().to_list()
            count_h = severity_list.count("high")
            count_m = severity_list.count("moderate")
            count_l = severity_list.count("low")
        else:
            count_h, count_m, count_l = 0, 0, 0
            
        # Define immediate output folder name
        new_folder_name = f"tile_{tile_idx}_H{count_h}_M{count_m}_L{count_l}"
        tile_dir = base_dir / new_folder_name
        tile_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving {new_folder_name}...")
        
        # Calculate water and cloud stats from SCL slice
        scl_slice = data_store["SCL"][r_min:r_max, c_min:c_max]
        water_pixels = int(np.sum(scl_slice == 6))
        cloud_mask = (scl_slice == 3) | ((scl_slice >= 7) & (scl_slice <= 10))
        per_clouds = round((np.sum(cloud_mask) / scl_slice.size) * 100, 2)
        
        # Save Tiffs
        # To maintain georeferencing, we calculate the bounds of the new tile
        # Center of tile in UTM
        tile_x_utm = master_center_x + (dx * 2560)
        tile_y_utm = master_center_y + (dy * 2560)
        tile_lon, tile_lat = transformer.transform(tile_x_utm, tile_y_utm)
        
        tile_transform = rasterio.transform.from_origin(
            tile_x_utm - 1280, tile_y_utm + 1280, 10, 10
        )
        
        for band in required_bands:
            if band in data_store:
                arr_slice = data_store[band][r_min:r_max, c_min:c_max]
                # Write to disk using rioxarray framework structure
                da = xr.DataArray(
                    arr_slice[np.newaxis, :, :],
                    coords={"band": [1], "y": np.linspace(tile_y_utm+1280, tile_y_utm-1280, 256), 
                            "x": np.linspace(tile_x_utm-1280, tile_x_utm+1280, 256)},
                    dims=("band", "y", "x")
                )
                da.rio.write_crs(master_crs, inplace=True)
                da.rio.write_transform(tile_transform, inplace=True)
                da.rio.to_raster(tile_dir / f"{band}_raw.tif")
                
        # Save Predictions CSV mapping coordinates relative to the tile!
        if not tile_preds.empty:
            tile_preds['pixel_row'] = tile_preds['pixel_row'] - r_min
            tile_preds['pixel_col'] = tile_preds['pixel_col'] - c_min
            tile_preds.to_csv(tile_dir / "cyfi_lattice_predictions.csv", index=False)
            
        # Save Metadata
        metadata = {
            "item_id": master_meta["item_id"], "per_clouds": per_clouds,
            "water_pixels": water_pixels, "num_pixels": 65536,
            "uid": f"{base_dir.name}_tile_{tile_idx}", "abun": "N/A", 
            "tile_size_10m_pixels": 256, "date": master_meta["date"],
            "center_lat": tile_lat, "center_lon": tile_lon,
            "High counts": count_h, "Moderate counts ": count_m, "Low counts ": count_l,
            "points_data": [{'case': base_dir.name.split("_")[0], 'lat': tile_lat, 'lon': tile_lon, 'date': master_meta["date"]}]
        }
        with open(tile_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        # Generate Visualization
        if "visual" in data_store:
            try:
                # visual is RGB, shape (3, H, W)
                vis_slice = data_store["visual"][:, r_min:r_max, c_min:c_max]
                bg_img = np.moveaxis(vis_slice, 0, -1)
                vmin, vmax = np.nanpercentile(bg_img, [2, 98])
                bg_img = np.clip((bg_img - vmin) / (vmax - vmin), 0, 1)

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(bg_img)
                
                if not tile_preds.empty:
                    severity_colors = {'low': 'green', 'moderate': 'orange', 'high': 'red', '1': 'green', '2': 'orange', '3': 'red'}
                    colors_pred = tile_preds['severity'].astype(str).map(lambda x: severity_colors.get(x.lower(), 'gray'))
                    ax.scatter(tile_preds['pixel_col'], tile_preds['pixel_row'], 
                               c=colors_pred, s=30, alpha=0.9, edgecolors='black', linewidth=0.5)
                               
                ax.set_title(f"Tile {tile_idx} | H:{count_h} M:{count_m} L:{count_l}")
                ax.set_axis_off()
                
                plt.savefig(tile_dir / "cyfi_prediction_map.png", bbox_inches='tight', dpi=150)
                if print_images:
                    plt.show()
                plt.close(fig)
            except Exception as e:
                print(f"Visualization failed for tile {tile_idx}: {e}")
                
        processed_folders.append((True, tile_dir))

    # 8. Cleanup Master Scene to save disk space
    print("Cleaning up Master Scene...")
    shutil.rmtree(master_dir, ignore_errors=True)
    
    return processed_folders

def load_tensor_from_folder(folder_path, use_mask=False, img_size=256):
    """Loads bands from a single folder, applies mask if needed, and returns a model-ready tensor."""
    band_names = [
        "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", "B05_raw.tif", "B06_raw.tif",
        "B07_raw.tif", "B08_raw.tif", "B8A_raw.tif", "B11_raw.tif", "B12_raw.tif"
    ]
    
    band_data = []
    for b_name in band_names:
        p = os.path.join(folder_path, b_name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required band file: {p}")
            
        with rasterio.open(p) as src:
            band_data.append(src.read(1).astype(np.float32))
            
    bands_stack = np.stack(band_data, axis=0)
    
    # Apply Mask (If requested)
    if use_mask:
        scl_path = os.path.join(folder_path, "SCL_raw.tif")
        if os.path.exists(scl_path):
            with rasterio.open(scl_path) as src:
                scl = src.read(1)
            # SCL 6 is water
            mask = (scl == 6).astype(np.float32)
            bands_stack = bands_stack * mask
        else:
            print("  [Warning] SCL_raw.tif not found in folder. Proceeding without mask.")
            
    # Convert to Tensor and add Batch Dimension -> Shape: (1, 10, H, W)
    tensor = torch.from_numpy(bands_stack).unsqueeze(0)
    
    # Resize to expected dimensions
    tensor = torch.nn.functional.interpolate(
        tensor, size=(img_size, img_size), 
        mode='bilinear', align_corners=False
    )
    
    # Normalize
    tensor = tensor / 10000.0
    
    return tensor


class HABLightningModel(L.LightningModule):
    def __init__(self, arch_name='resnet18', num_classes=2, in_chans=10):
        super().__init__()
        self.model = self._build_model(arch_name, num_classes, in_chans)

    def _build_model(self, arch, num_classes, in_chans):
        if arch == 'resnet18':
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        elif arch == 'convnextv2_base':
            return timm.create_model('convnextv2_base', pretrained=False, num_classes=num_classes, in_chans=in_chans)
        elif arch == 'rdnet_base':
            return timm.create_model('rdnet_base', pretrained=False, num_classes=num_classes, in_chans=in_chans)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, x):
        return self.model(x)


def run_single_folder(base_folder_path, model_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # Save a master results file in the root
    output_txt_path = os.path.join(base_folder_path, "cnn_results_all_tiles.txt")
    
    with open(output_txt_path, "w") as f:
        header = f"Analyzing Base Folder: {base_folder_path}\n" + "-" * 50
        print(header)
        f.write(header + "\n")
        
        scenarios = [
            ("res18_scl",      "resnet18",        True,  256),
            ("res18_no_scl",   "resnet18",        False, 256),
            ("convnext_scl",   "convnextv2_base", True,  256),
            ("rdnet_no_scl",   "rdnet_base",      False, 256)
        ]
        
        # Dynamically find all subdirectories starting with "tile_" to account for CyFi's folder renaming
        subdirs = [d for d in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, d)) and d.startswith("tile_")]
        subdirs.sort(key=lambda x: int(x.split('_')[1]))
        
        for tile_dir_name in subdirs:
            tile_folder = os.path.join(base_folder_path, tile_dir_name)
            
            tile_header = f"\n--- Results for {tile_dir_name} ---"
            print(tile_header)
            f.write(tile_header + "\n")
            
            for scenario_name, arch, use_mask, img_size in scenarios:
                try:
                    input_tensor = load_tensor_from_folder(tile_folder, use_mask=use_mask, img_size=img_size)
                    input_tensor = input_tensor.to(device)
                    
                    model_wrapper = HABLightningModel(arch_name=arch)
                    ckpt_path = model_paths.get(scenario_name)
                    
                    if not ckpt_path or not os.path.exists(ckpt_path):
                        error_msg = f"{scenario_name:<15} | ERROR: Weights not found at {ckpt_path}"
                        print(error_msg)
                        f.write(error_msg + "\n")
                        continue
                        
                    if ckpt_path.endswith('.ckpt'):
                        checkpoint = torch.load(ckpt_path, map_location='cpu')
                        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
                        model_wrapper.model.load_state_dict(state_dict, strict=False)
                    else:
                        state_dict = torch.load(ckpt_path, map_location='cpu')
                        model_wrapper.model.load_state_dict(state_dict, strict=False)
                        
                    model_wrapper.to(device)
                    model_wrapper.eval()
                    
                    with torch.no_grad():
                        outputs = model_wrapper(input_tensor)
                        _, preds = torch.max(outputs, 1)
                        pred_class = preds.item()
                        
                    str_pred = "YES" if pred_class == 1 else "NO"
                    result_line = f"Model: {scenario_name:<15} | HAB Detected: {str_pred}"
                    
                    print(result_line)
                    f.write(result_line + "\n")
                    
                except Exception as e:
                    err_line = f"Model: {scenario_name:<15} | ERROR: {str(e)}"
                    print(err_line)
                    f.write(err_line + "\n")
                    
    print(f"\nResults have been successfully saved to: {output_txt_path}")
    

def create_collage(base_dir):
    """
    Creates a 3x3 collage of the 9 tiles with black borders.
    Draws the tile number in white in the top-left corner of each tile.
    Saves the output as 'colage.png' in the base directory.
    """
    print("--- Generating 3x3 Collage ---")
    
    # Configuration for the collage
    tile_size = 500  # Resize all images to 500x500 pixels for perfect alignment
    border = 10      # 10 pixel black line between tiles
    total_size = (tile_size * 3) + (border * 4)
    
    # Map the tile index to a 3x3 grid (row, col)
    idx_to_pos = {
        2: (0, 0), 1: (0, 1), 8: (0, 2),
        3: (1, 0), 0: (1, 1), 7: (1, 2),
        4: (2, 0), 5: (2, 1), 6: (2, 2)
    }
    
    # Try to load a large font if supported by the Pillow version, otherwise use default
    try:
        font = ImageFont.load_default(size=50)
    except TypeError:
        font = ImageFont.load_default()
    
    # Create a completely black canvas
    collage = Image.new("RGB", (total_size, total_size), "black")
    draw = ImageDraw.Draw(collage)
    
    # Loop through the 9 expected tiles
    for i in range(9):
        row, col = idx_to_pos[i]
        
        folders = list(base_dir.glob(f"tile_{i}*"))
        if not folders or not folders[0].is_dir():
            continue  
            
        tile_dir = folders[0]
        
        # 1. Try to find the CyFi Prediction Map
        img_path = tile_dir / "cyfi_prediction_map.png"
        
        # 2. If it doesn't exist, fallback to the Visual Preview
        if not img_path.exists():
            img_path = tile_dir / "visual_preview.png"
            
        # If an image was found, process, paste, and label it
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((tile_size, tile_size))
                
                # Calculate coordinates
                x = border + col * (tile_size + border)
                y = border + row * (tile_size + border)
                
                # Paste into the collage
                collage.paste(img, (x, y))
                
                # Draw the number with a black outline for visibility
                text = str(i)
                text_x, text_y = x + 15, y + 15
                
                # Draw black outline
                outline_color = "black"
                draw.text((text_x - 2, text_y - 2), text, font=font, fill=outline_color)
                draw.text((text_x + 2, text_y - 2), text, font=font, fill=outline_color)
                draw.text((text_x - 2, text_y + 2), text, font=font, fill=outline_color)
                draw.text((text_x + 2, text_y + 2), text, font=font, fill=outline_color)
                
                # Draw white text over it
                draw.text((text_x, text_y), text, font=font, fill="white")
                
            except Exception as e:
                print(f"  [Warning] Could not process image for tile_{i}: {e}")
                
    # Save the final collage
    out_path = base_dir / "colage.png"
    collage.save(out_path)
    print(f"Collage successfully saved to: {out_path}")


def folder_summary(base_folder_path, add_images_to_excel=True):
    """
    Summarizes CyFi and CNN results, calculates totals, creates a collage,
    and optionally embeds prediction map thumbnails into the Excel file.
    """
    base_dir = Path(base_folder_path)
    print(f"--- Generating Summary for {base_dir.name} ---")
    
    # 1. Generate Collage
    create_collage(base_dir)
    
    # 2. Parse CNN results
    cnn_results_file = base_dir / "cnn_results_all_tiles.txt"
    cnn_data = {}
    
    if cnn_results_file.exists():
        with open(cnn_results_file, 'r') as f:
            lines = f.readlines()
        
        current_tile = None
        for line in lines:
            line = line.strip()
            if line.startswith("--- Results for "):
                current_tile = line.replace("--- Results for ", "").replace(" ---", "")
                cnn_data[current_tile] = {}
            elif line.startswith("Model:") and current_tile is not None:
                parts = line.split("|")
                model_name = parts[0].replace("Model:", "").strip()
                prediction = parts[1].replace("HAB Detected:", "").strip()
                cnn_data[current_tile][model_name] = prediction
    else:
        print(f"  [Warning] CNN results file not found at {cnn_results_file}")

    # 3. Gather CyFi Stats
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d) and d.startswith("tile_")]
    
    try:
        subdirs.sort(key=lambda x: int(x.split('_')[1]))
    except Exception:
        subdirs.sort()
        
    summary_list = []
    
    for tile_dir_name in subdirs:
        tile_path = base_dir / tile_dir_name
        meta_path = tile_path / "metadata.json"
        
        h_count, m_count, l_count, water_pixels = 0, 0, 0, 0
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                h_count = meta.get("High counts", 0)
                m_count = meta.get("Moderate counts ", meta.get("Moderate counts", 0)) 
                l_count = meta.get("Low counts ", meta.get("Low counts", 0))
                water_pixels = meta.get("water_pixels", 0)
                all_pixels = meta.get("num_pixels", 0)
                
        total_cyfi_points = h_count + m_count + l_count
        
        row_data = {
            "Tile_Folder": tile_dir_name,
            "CyFi_Map": "", 
            "Total_pixels": all_pixels,
            "Total_CyFi_Points": total_cyfi_points,
            "SCL_Water_Pixels": water_pixels,
            "CyFi_High": h_count,
            "CyFi_Moderate": m_count,
            "CyFi_Low": l_count
        }
        
        tile_cnn = cnn_data.get(tile_dir_name, {})
        row_data["res18_scl"] = tile_cnn.get("res18_scl", "N/A")
        row_data["res18_no_scl"] = tile_cnn.get("res18_no_scl", "N/A")
        row_data["convnext_scl"] = tile_cnn.get("convnext_scl", "N/A")
        row_data["rdnet_no_scl"] = tile_cnn.get("rdnet_no_scl", "N/A")
        
        summary_list.append(row_data)
        
    # 4. Create DataFrame and inject images into Excel
    if summary_list:
        df = pd.DataFrame(summary_list)
        output_excel = base_dir / "folder_summary.xlsx"
        
        df.to_excel(output_excel, index=False)
        
        if add_images_to_excel:
            print("--- Adding Image Thumbnails to Excel ---")
            wb = openpyxl.load_workbook(output_excel)
            ws = wb.active
            
            ws.column_dimensions['B'].width = 18
            
            for idx, tile_dir_name in enumerate(df['Tile_Folder']):
                row_num = idx + 2  
                ws.row_dimensions[row_num].height = 80
                
                img_path = base_dir / tile_dir_name / "cyfi_prediction_map.png"
                
                if img_path.exists():
                    try:
                        img = OpenpyxlImage(str(img_path))
                        img.width = 100
                        img.height = 100
                        ws.add_image(img, f"B{row_num}")
                    except Exception as e:
                        print(f"  [Warning] Could not insert image for {tile_dir_name}: {e}")

            wb.save(output_excel)
            
        print(f"Summary successfully saved to: {output_excel}")
        #display(df.drop(columns=["CyFi_Map"]))
        return df
    else:
        print("No tile subfolders found to summarize.")
        return None
    
