# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 00:26:37 2026

@author: K. Pikounis
"""

import rioxarray
import os
import rasterio

# Point this to one of your saved bands (e.g., B04_raw.tif)
tif_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\v10_to_100\case741_abun1_365_date2018-06-17_H0_M88_L301/B01_raw.tif"

if os.path.exists(tif_path):
    # 1. Open the raster
    da = rioxarray.open_rasterio(tif_path)

    # 2. Reproject the bounds to WGS84 (Lat/Lon)
    # We use .rio.reproject() to calculate the transformation
    da_latlon = da.rio.reproject("EPSG:4326")

    # 3. Get the bounds
    min_lon, min_lat, max_lon, max_lat = da_latlon.rio.bounds()

    print(f"Lat Range: {min_lat:.5f} to {max_lat:.5f}")
    print(f"Lon Range: {min_lon:.5f} to {max_lon:.5f}")
else:
    print("File not found.")

    

def get_tif_dimensions_fast(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            width = src.width
            height = src.height
            bands = src.count
            total_pixels = width * height
            
            print(f"Dimensions: {width} x {height}")
            print(f"Total Pixels: {total_pixels}")
            
            return total_pixels, width, height
            
    except Exception as e:
        print(f"Error: {e}")
        return None