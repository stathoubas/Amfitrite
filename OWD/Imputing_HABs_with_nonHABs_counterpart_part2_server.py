# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:30:54 2026

@author: K. Pikounis
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

import planetary_computer as pc
import os
import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as L
import timm
import xarray as xr
import argparse
import logging
import traceback
import csv
from pystac_client import Client

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

torch.set_num_threads(6)

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def fetch_specific_stac_item(item_id ,catalog):
    """Fetches the exact STAC item object from Planetary Computer using its ID."""
    search = catalog.search(collections=["sentinel-2-l2a"], ids=[item_id])
    items = list(search.items())
    if not items:
        return None
    return items[0]

def extract_buffered_scene(
    item,
    lat,
    lon,
    case_id,
    output_path=None, 
):
    """
    Downloads a 6560m x 6560m bounding box around the center point.
    (2560m core for the single tile + 2000m buffer on each side for CyFi context).
    Saves the aligned TIFFs to a 'buffered_scene' subfolder.
    """
    print(f"--- Extracting Buffered Scene for {case_id} ---")
    
    # 1. Setup CRS and Calculate Center in UTM
    target_crs = CRS.from_string(item.properties["proj:code"])
    print(f"Center Point (Lat/Lon): ({lat:.4f}, {lon:.4f})")
    
    transformer_latlon_to_utm = Transformer.from_crs(CRS.from_string("EPSG:4326"), target_crs, always_xy=True)
    center_x, center_y = transformer_latlon_to_utm.transform(lon, lat)

    # 2. Define Buffered Bounding Box
    # CHANGE: Total width = 2560m (core) + 4000m (2000m buffer * 2) = 6560m
    # Half side = 3280m
    half_side = 3280.0
    
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
        
    buffered_dir = base_output_dir / "buffered_scene"
    buffered_dir.mkdir(parents=True, exist_ok=True)
    
    clipped_data = {}
    
    # 4. Clip Reference Band (B04)
    try:
        print("Clipping reference band (B04)...")
        b04_clip = b04_ds_full.rio.clip_box(
            minx=clip_bbox_utm[0], miny=clip_bbox_utm[1], 
            maxx=clip_bbox_utm[2], maxy=clip_bbox_utm[3],
            crs=target_crs
        )
																				 
        b04_clip = b04_clip.isel(y=slice(0, 656), x=slice(0, 656))
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
        print(f"Error: Bounding box outside bounds for buffered scene. Skipping.")
        return False
    except Exception as e:
        print(f"An error occurred during clipping buffered scene: {e}")
        return False

    # 6. Save Data
    print(f"Saving TIFFs to {buffered_dir}...")
    for key, da in clipped_data.items():
        da.rio.to_raster(buffered_dir / f"{key}_raw.tif")
        
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
    with open(buffered_dir / "buffered_metadata.json", "w") as f:
        json.dump(master_meta, f, indent=4)
            
    print("Buffered Scene extraction complete!")
    return True

def run_cyfi_and_slice_core(
    base_folder_path, 
    date_str, 
    metadata_filename="buffered_metadata.json",
    print_images=False
):
    """
    Runs CyFi on the buffered scene, generating predictions ONLY for the 256x256 core.
    Slices the core arrays, counts water pixels, and saves the final TIFFs/PNG directly
    into the base folder. Returns CyFi counts and water pixels.
    """
    base_dir = Path(base_folder_path).resolve()
    buffered_dir = base_dir / "buffered_scene"
    
    print(f"--- Starting CyFi Prediction & Core Slicing in: {base_dir} ---")
    
    if not buffered_dir.exists():
        print(f"Error: buffered_scene folder not found in {base_dir}. Skipping.")
        return 0, 0, 0, 0, 0

    # 1. Load Buffered Metadata
    with open(buffered_dir / metadata_filename, "r") as f:
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
 
    
    # 3. Load Buffered TIFFs into Memory
    print("Loading Buffered TIFFs into memory...")
    data_store = {}
    
    # CHANGE: Array is now 656x656 (6560m / 10m)
    master_height, master_width = 656, 656  
    
    for band in required_bands:
        path = buffered_dir / f"{band}_raw.tif"
        if path.exists():
            ds = rioxarray.open_rasterio(path).squeeze()
            data_store[band] = ds.values
            if band == "SCL":
                transform = ds.rio.transform()
        else:
            data_store[band] = np.full((master_height, master_width), np.nan, dtype=np.float32)
            
    # Also load visual for plotting later
    vis_path = buffered_dir / "visual_raw.tif"
    if vis_path.exists():
        data_store["visual"] = rioxarray.open_rasterio(vis_path).squeeze().values

    # 4. Generate 100m Lattice Grid strictly inside the 2560m core
														 
																	
    print("Generating 100m Lattice Grid in the core area...")
    GRID_STEP = 10 
    
    # Core sits exactly in the middle of the 656 array (200 to 456)
    core_min_idx = 200
    core_max_idx = 456
    
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

    temp_cache = Path(tempfile.mkdtemp(prefix="cyfi_core_lattice_"))
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

    # 6. Run CyFi Pipeline on the Core
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

    # 7. Extract the Core Tile & Save Outputs
    print("\n--- Saving Final Core Tile ---")
    
    # Evaluate only the exact core
    tile_preds = pd.DataFrame()
    if not master_results_df.empty:
        tile_preds = master_results_df[
            (master_results_df.pixel_row >= core_min_idx) & (master_results_df.pixel_row < core_max_idx) &
            (master_results_df.pixel_col >= core_min_idx) & (master_results_df.pixel_col < core_max_idx)
        ].copy() 
        
    # Count Severity
    if not tile_preds.empty:
        severity_list = tile_preds.severity.astype(str).str.lower().to_list()
        count_h = severity_list.count("high")
        count_m = severity_list.count("moderate")
        count_l = severity_list.count("low")
    else:
        count_h, count_m, count_l = 0, 0, 0
        
    # Calculate water pixels from SCL core slice
    scl_slice = data_store["SCL"][core_min_idx:core_max_idx, core_min_idx:core_max_idx]
    water_pixels = int(np.sum(scl_slice == 6))
    cloud_mask = (scl_slice == 3) | ((scl_slice >= 7) & (scl_slice <= 10))
    per_clouds = round((np.sum(cloud_mask) / scl_slice.size) * 100, 2)
    
    # Save TIFFs directly to base_dir
    tile_transform = rasterio.transform.from_origin(
        master_center_x - 1280, master_center_y + 1280, 10, 10
    )
    
    for band in required_bands:
        if band in data_store:
            arr_slice = data_store[band][core_min_idx:core_max_idx, core_min_idx:core_max_idx]
																   
            da = xr.DataArray(
                arr_slice[np.newaxis, :, :],
                coords={"band": [1], "y": np.linspace(master_center_y+1280, master_center_y-1280, 256), 
                        "x": np.linspace(master_center_x-1280, master_center_x+1280, 256)},
                dims=("band", "y", "x")
            )
            da.rio.write_crs(master_crs, inplace=True)
            da.rio.write_transform(tile_transform, inplace=True)
            da.rio.to_raster(base_dir / f"{band}_raw.tif")

    # Save Predictions CSV mapping coordinates relative to the tile!
    if not tile_preds.empty:
        tile_preds['pixel_row'] = tile_preds['pixel_row'] - core_min_idx
        tile_preds['pixel_col'] = tile_preds['pixel_col'] - core_min_idx
        tile_preds.to_csv(base_dir / "cyfi_lattice_predictions.csv", index=False)
        
    # Generate Visualization (Saved directly to base_dir)
    if "visual" in data_store:
        try:
												
            vis_slice = data_store["visual"][:, core_min_idx:core_max_idx, core_min_idx:core_max_idx]
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
                           
            ax.set_title(f"CyFi Map | H:{count_h} M:{count_m} L:{count_l}")
            ax.set_axis_off()
            
            plt.savefig(base_dir / "cyfi_prediction_map.png", bbox_inches='tight', dpi=150)
            if print_images:
                plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"Visualization failed: {e}")

    # 8. Cleanup Buffered Scene to save disk space
    print("Cleaning up Buffered Scene...")
    shutil.rmtree(buffered_dir, ignore_errors=True)
    
    return count_h, count_m, count_l, water_pixels, per_clouds


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
    


def setup_logger(output_root, base_name):
    """Sets up a text logger for detailed debugging."""
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    log_file = os.path.join(output_root, f"execution_log_{base_name}.log")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    return log_file

def run_cnns_on_tile(base_folder_path, model_paths):
    """
    Evaluates the single 256x256 tile residing directly in base_folder_path
    using all 4 CNN scenarios. Returns a dictionary of the predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running CNN Inference in: {base_folder_path} ---")
    print(f"Device: {device}")
    
    scenarios = [
        ("res18_scl",      "resnet18",        True,  256),
        ("res18_no_scl",   "resnet18",        False, 256),
        ("convnext_scl",   "convnextv2_base", True,  256),
        ("rdnet_no_scl",   "rdnet_base",      False, 256)
    ]
    
    # Initialize dictionary to hold results for the CSV
    results = {}
    
    for scenario_name, arch, use_mask, img_size in scenarios:
        try:
            # Load tensor directly from the base folder (no subdirectories!)
            input_tensor = load_tensor_from_folder(base_folder_path, use_mask=use_mask, img_size=img_size)
            input_tensor = input_tensor.to(device)
            
            model_wrapper = HABLightningModel(arch_name=arch)
            ckpt_path = model_paths.get(scenario_name)
            
            if not ckpt_path or not os.path.exists(ckpt_path):
                error_msg = f"{scenario_name:<15} | ERROR: Weights not found at {ckpt_path}"
                print(error_msg)
                results[scenario_name] = "ERROR"
                continue
                
            # Load weights safely
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
            print(f"Model: {scenario_name:<15} | HAB Detected: {str_pred}")
            
            # Store in dictionary instead of text file
            results[scenario_name] = str_pred
            
        except Exception as e:
            print(f"Model: {scenario_name:<15} | ERROR: {str(e)}")
            results[scenario_name] = "ERROR"
            
    return results

def run_heavy_pipeline(input_path, output_root, model_paths):
    # 1. Setup Tracking and Logging
    base_name = os.path.splitext(os.path.basename(input_path))[0]
        
    log_txt_path = setup_logger(output_root, base_name)
    log_csv_path = os.path.join(output_root, f"summary_report_{base_name}.csv")
    
    logging.info(f"Starting Phase 2 Pipeline. Reading input from: {input_path}")
    
    # --- RESUME FUNCTIONALITY ---
    processed_cases = set()
    
    if os.path.exists(log_csv_path):
        logging.info(f"Found existing summary report: {log_csv_path}. Checking for processed cases...")
        try:
            existing_log_df = pd.read_csv(log_csv_path)
            processed_cases = set(existing_log_df['case_id'].astype(str))
            logging.info(f"Found {len(processed_cases)} previously processed cases. These will be skipped.")
            
            log_file_mode = 'a'
            write_header = False
        except Exception as e:
            logging.warning(f"Could not parse existing summary report. Starting fresh. Error: {e}")
            log_file_mode = 'w'
            write_header = True
    else:
        log_file_mode = 'w'
        write_header = True

    csv_headers = [
        "case_id", "date", "status", "failed_stage", "error_message", "output_folder",
        "SCL_Water_Pixels", "Cloud_Percentage", "CyFi_High", "CyFi_Moderate", "CyFi_Low",
        "res18_scl", "res18_no_scl", "convnext_scl", "rdnet_no_scl",
        "Overall_Class", "Total_Index"
    ]
    with open(log_csv_path, mode=log_file_mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(csv_headers)
        
    # 2. Read the Input Data (Robust Loader)
    try:
        if input_path.endswith('.xlsx') or input_path.endswith('.xls'):
            df = pd.read_excel(input_path)
        else:
            try:
                df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(input_path, encoding='latin-1', sep=';', low_memory=False)
    except Exception as e:
        logging.error(f"Failed to read input file {input_path}. Error: {e}")
        return

    # 3. Process Row by Row
    for index, row in df.iterrows():
        
        # --- ID Fallback Logic ---
        raw_id = row.get('ID')
            
        case_id = str(raw_id).replace(":", "_")
        
        sat_item_id = str(row.get('sat_item', "")).strip()
        if not sat_item_id or sat_item_id.lower() == 'nan':
            logging.info(f"Row {index+1}: Case {case_id} has no valid satellite item. Skipping.")
            continue
        if case_id in processed_cases:
            logging.info(f"Row {index+1}: Case {case_id} already processed. Skipping.")
            continue
        
        lat_val = row.get('center_lat')        
        lon_val = row.get('center_lon')
        try:
            lat = float(str(lat_val).replace(',', '.'))
            lon = float(str(lon_val).replace(',', '.'))
        except (ValueError, TypeError):
            logging.error(f"Row {index} ({case_id}): Invalid Lat/Lon. Skipping.")
            continue
            
        raw_date = row.get('date')
        try:
            parsed_date = pd.to_datetime(str(raw_date), dayfirst=True)
            date_str = parsed_date.strftime("%Y-%m-%d")
        except Exception:
            logging.error(f"Row {index} ({case_id}): Invalid Date format. Skipping.")
            continue
            
        uid = f"{case_id}_{date_str}"
        logging.info(f"\n{'='*60}\nProcessing Case: {uid} | Item: {sat_item_id} (Row {index+1}/{len(df)})\n{'='*60}")
        
        status = "SUCCESS"
        failed_stage = "None"
        error_msg = ""
        
        # CHANGE 2: Output folder is now uniquely named with UID (ID + Date)
        out_folder_name = uid
        out_folder_path = os.path.join(output_root, out_folder_name)
        
        # Single tile tracking variables
        overall_class = "Unknown"
        total_index = 0.0
        water_px, per_clouds, h_count, m_count, l_count = 0, 0.0, 0, 0, 0
        cnn_results = {"res18_scl": "N/A", "res18_no_scl": "N/A", "convnext_scl": "N/A", "rdnet_no_scl": "N/A"}
        
        try:
            # --- STAGE 1: Fetch Pre-Determined Item ---
            failed_stage = "1_Fetch_Item"
            selected_item = fetch_specific_stac_item(sat_item_id, catalog)
            if not selected_item:
                raise ValueError(f"Could not retrieve item {sat_item_id}")
                
            final_date = selected_item.datetime.strftime("%Y-%m-%d")
            os.makedirs(out_folder_path, exist_ok=True)

            # --- STAGE 2: Extract Buffered Scene ---
            failed_stage = "2_Extract_Buffered_Scene"
            extract_ok = extract_buffered_scene(
                item=selected_item, lat=lat, lon=lon, case_id=uid, output_path=out_folder_path
            )
            if not extract_ok:
                raise ValueError("Scene extraction failed.")

            # --- STAGE 3: CyFi Pipeline & Core Slicing ---
            failed_stage = "3_CyFi_Pipeline_And_Slicing"
            h_count, m_count, l_count, water_px, per_clouds = run_cyfi_and_slice_core(
                base_folder_path=out_folder_path, date_str=final_date, print_images=False
            )

            # --- STAGE 4: CNN Inference & Filters ---
            failed_stage = "4_CNN_Inference"
            if water_px < 32000:
                overall_class = "Filters removed all tiles"
                logging.info(f"Skipping CNNs: Insufficient water pixels ({water_px} < 32000).")
            else:
                cnn_results = run_cnns_on_tile(out_folder_path, model_paths)

                # --- STAGE 5: INLINE CLASSIFICATION & SCORING ---
                failed_stage = "5_Classification"
                
                # Helper for CNN "YES" checks
                def is_yes(model_key):
                    return cnn_results.get(model_key) == "YES"
                    
                rdnet_yes = is_yes('rdnet_no_scl')
                cnn_cols = ['rdnet_no_scl', 'convnext_scl', 'res18_scl', 'res18_no_scl']
                yes_count = sum(1 for col in cnn_cols if is_yes(col))
                
                # Score CNN (Max 60)
                cnn_score = 0
                if is_yes('rdnet_no_scl'): cnn_score += 30
                if is_yes('convnext_scl'): cnn_score += 10
                if is_yes('res18_scl'): cnn_score += 10
                if is_yes('res18_no_scl'): cnn_score += 10
                
                # Score CyFi (Max 40)
                cyfi_mass = h_count + (0.5 * m_count)
                cyfi_score = (cyfi_mass / 676.0) * 40.0
                
                total_index = round(cnn_score + cyfi_score, 2) 
                
                # Apply Classification Logic
                is_hab = rdnet_yes and (yes_count >= 2) and (h_count >= 20)
                is_clean = (not rdnet_yes) and (yes_count <= 1) and (h_count < 5) and (m_count < 15)
                
                if is_hab:
                    overall_class = "HAB Detected"
                elif is_clean:
                    overall_class = "Clean"
                else:
                    overall_class = "To Be Checked"

            failed_stage = "None" 
            logging.info(f"Successfully completed {uid} | Class: {overall_class} | Score: {total_index}")

        except Exception as e:
            status = "FAILED"
            error_msg = str(e).replace('\n', ' ')
            logging.error(f"Failed at {failed_stage} for {uid}. Error: {error_msg}")
            logging.debug(traceback.format_exc()) 
            
        finally:
            
            buffered_cleanup_path = os.path.join(out_folder_path, "buffered_scene")
            if os.path.exists(buffered_cleanup_path):
                shutil.rmtree(buffered_cleanup_path, ignore_errors=True)
            
            # Write everything immediately to CSV
            row_data = [
                case_id, date_str, status, failed_stage, error_msg, out_folder_path,
                water_px, per_clouds, h_count, m_count, l_count,
                cnn_results.get("res18_scl", "N/A"), cnn_results.get("res18_no_scl", "N/A"), 
                cnn_results.get("convnext_scl", "N/A"), cnn_results.get("rdnet_no_scl", "N/A"),
                overall_class, total_index
            ]
            with open(log_csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
                f.flush()

    logging.info(f"\nAll rows processed. Summary report saved to: {log_csv_path}")


if __name__ == "__main__":
    MODEL_PATHS = {
        "res18_scl":    "/vol/Amfitrite/CNNs_for_annotation/res18_scl/best_epoch_16.pth",
        "res18_no_scl": "/vol/Amfitrite/CNNs_for_annotation/res18_no_scl/best_epoch_26.pth",
        "convnext_scl": "/vol/Amfitrite/CNNs_for_annotation/convnext_scl/best_epoch_15.pth",
        "rdnet_no_scl": "/vol/Amfitrite/CNNs_for_annotation/rdnet_no_scl/best_epoch_35.pth"
    }
    
    parser = argparse.ArgumentParser(description="Phase 2: Heavy Execution Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV/Excel file")
    parser.add_argument("--output", type=str, required=True, help="Path to the root output folder")
    args = parser.parse_args()
    
    run_heavy_pipeline(args.input, args.output, MODEL_PATHS)

