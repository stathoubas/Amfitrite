# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:42:45 2026

@author: K. Pikounis
"""

import os
import pandas as pd
import numpy as np

def strip_images_from_excel(summary_excel_path):
    """
    Reads a heavy Excel file (with embedded images), ignores the images,
    and saves a lightweight '_v2.xlsx' version. Returns the clean DataFrame.
    """
    if not os.path.exists(summary_excel_path):
        return None
        
    try:
        df = pd.read_excel(summary_excel_path)
        v2_path = summary_excel_path.replace(".xlsx", "_v2.xlsx")
        df.to_excel(v2_path, index=False)
        
        return df
    except Exception as e:
        print(f"Error stripping images from {summary_excel_path}: {e}")
        return None
    
    
if __name__ == "__main__":
    root_directory = "/vol2/Amfitrite/OWD/California/HAB" 
    
    # Αναζήτηση σε όλους τους υποφακέλους
    for root, dirs, files in os.walk(root_directory):
        if "folder_summary.xlsx" in files:
            file_path = os.path.join(root, "folder_summary.xlsx")
            print(f"Processing: {file_path}")
            
            result = strip_images_from_excel(file_path)
            
            if result is not None:
                print(f"Successfully created _v2 for: {root}")
            else:
                print(f"Skipped or failed: {root}")
    