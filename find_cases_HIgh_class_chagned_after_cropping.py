# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 01:29:35 2026

@author: K. Pikounis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

import pandas as pd
import os
import rasterio


def get_tif_dimensions_fast(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            width = src.width
            height = src.height
            bands = src.count
            total_pixels = width * height
            
            #print(f"Dimensions: {width} x {height}")
            #print(f"Total Pixels: {total_pixels}")
            
            return total_pixels, width, height
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def show_image_with_relative_box(folder_path):
    """
    Reads 'cyfi_prediction_map.png', draws a red square corresponding 
    to the central 256x256 crop of the original 365x365 TIF data.
    """
    image_name = "cyfi_prediction_map.png"
    image_path = os.path.join(folder_path, image_name)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    try:
        img = mpimg.imread(image_path)
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    # 1. Get PNG dimensions
    img_h, img_w = img.shape[:2]
    
    # 2. Define the Ratio based on TIF dimensions
    TIF_FULL_SIZE = 365.0
    TIF_CROP_SIZE = 256.0
    
    ratio = TIF_CROP_SIZE / TIF_FULL_SIZE  # ~0.701
    
    # 3. Calculate Box Size in PNG pixels
    # We assume the PNG displays the full 365x365 extent
    box_w_px = img_w * ratio
    box_h_px = img_h * ratio
    
    # 4. Calculate Center Position
    x_start = (img_w - box_w_px) / 2
    y_start = (img_h - box_h_px) / 2
    
    print(f"PNG Size: {img_w}x{img_h}")
    print(f"Drawing box representing {int(TIF_CROP_SIZE)} crop (Ratio: {ratio:.2f})")
    print(f"Box Pixels on PNG: {box_w_px:.1f}x{box_h_px:.1f}")

    # 5. Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    rect = patches.Rectangle(
        (x_start, y_start), 
        box_w_px, 
        box_h_px, 
        linewidth=2, 
        edgecolor='red', 
        facecolor='none'
    )
    
    ax.add_patch(rect)
    ax.set_title(f"Red Box = Central {int(TIF_CROP_SIZE)}x{int(TIF_CROP_SIZE)} of Original TIF\nUID: {os.path.basename(folder_path)}")
    plt.axis('off')
    plt.show()


def find_edge_case_highs(master_excel_path, root_data_path):
    """
    Reads the master excel, selects 'High' class cases, and checks their 
    CyFi CSVs to see if they have < 10 high points in the center 256x256 crop.
    """

    # 1. Load Master Excel
    print(f"Loading Master Excel: {master_excel_path}")
    try:
        df = pd.read_excel(master_excel_path)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return []

    # 2. Filter for High Class Cases
    # We normalize to title case (High) or lower case to be safe
    if 'Class' not in df.columns:
        print("Error: Column 'Class' not found in Excel.")
        return []

    high_cases_df = df[df['Class'].astype(str).str.lower() == 'high']
    print(f"Found {len(high_cases_df)} cases labeled as 'High'. Checking point distribution...")

    flagged_cases = []

    # Constants for Cropping
    # Assuming original image is 365x365 (standard for this dataset)
    ORIGINAL_SIZE = 365 
    CROP_SIZE = 256

    # Calculate crop bounds (Center Crop)
    # Margin = (365 - 256) / 2 = 54.5 -> Round to 54
    # Range: 54 to 310
    margin = int((ORIGINAL_SIZE - CROP_SIZE) / 2)
    min_idx = margin
    max_idx = margin + CROP_SIZE

    print(f"Defining Center Crop: Rows/Cols from {min_idx} to {max_idx} (Original size: {ORIGINAL_SIZE})")

    # 3. Iterate through cases
    for index, row in high_cases_df.iterrows():
        uid = str(row['uid']).strip()

        total_pixels, width, height = get_tif_dimensions_fast(os.path.join(root_data_path, uid, "B04_raw.tif"))
        if width != ORIGINAL_SIZE or height != ORIGINAL_SIZE:
            continue

        # Construct path to the CSV
        csv_path = os.path.join(root_data_path, uid, "cyfi_lattice_predictions.csv")

        if not os.path.exists(csv_path):
            # Optional: print warning if file is missing
            # print(f"Warning: CSV not found for {uid}")
            continue

        try:
            # Read CyFi Predictions
            cyfi_df = pd.read_csv(csv_path)

            # 4. Filter for Center Crop (256x256)
            # Check row and col indices
            center_points = cyfi_df[
                (cyfi_df['pixel_row'] >= min_idx) & 
                (cyfi_df['pixel_row'] < max_idx) & 
                (cyfi_df['pixel_col'] >= min_idx) & 
                (cyfi_df['pixel_col'] < max_idx)
            ]

            # 5. Count 'high' severity points
            # Ensure case-insensitive comparison (CyFi output is usually lowercase 'high')
            high_point_count = len(center_points[center_points['severity'].astype(str).str.lower() == 'high'])

            # 6. Check Threshold (< 10)
            if high_point_count < 10:
                print(f"-> Found Case: {uid} | Center High Points: {high_point_count}")
                flagged_cases.append({
                    'uid': uid,
                    'original_class': row['Class'],
                    'center_high_points': high_point_count,
                    'csv_path': csv_path
                })

        except Exception as e:
            print(f"Error processing {uid}: {e}")

    # 7. Create Result DataFrame
    result_df = pd.DataFrame(flagged_cases)

    print("-" * 50)
    print(f"Processing Complete.")
    print(f"Found {len(result_df)} 'High' cases with < 10 high points in the center 256x256 crop.")

    return result_df

master_excel_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\from_server/Master_summary_all_parts_with_flags_and_extra_cases_v8_updated.xlsx"
root_data_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\Dataset_v1"

junk = find_edge_case_highs(master_excel_path, root_data_path)