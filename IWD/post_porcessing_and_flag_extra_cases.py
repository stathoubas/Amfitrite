# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 13:32:55 2026

@author: K. Pikounis
"""

import pandas as pd
import os
from tqdm import tqdm

SEVERE_THRESH = 0.10
MODERATE_THRESH = 0.10

#df0 = pd.read_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\extra_low_cases\results\Master_summary.xlsx")
#out_file = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\extra_low_cases\results\Master_summary_v1.xlsx"

#df0 = pd.read_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\extra_high_cases\results\Master_summary.xlsx")
#out_file = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\extra_high_cases\results\Master_summary_v1.xlsx"

df0 = pd.read_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\extra_cases_not_previously_processed_before_and_after_2017\results\Master_summary.xlsx")
out_file = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\extra_cases_not_previously_processed_before_and_after_2017\results\Master_summary_v1.xlsx"


df = df0.copy()
for idx, row in tqdm(df.iterrows(), total=len(df)):
    h = float(row.get('high_pred', 0))
    m = float(row.get('mod_pred', 0))
    l = float(row.get('low_pred', 0))
    total = h + m + l
    
    if total > 0:
        frac_h = h / total
        frac_m = m / total
        frac_l = l / total
        df.at[idx, 'total_pred'] = total
        df.at[idx, 'high_frac'] = frac_h
        df.at[idx, 'mod_frac'] = frac_m
        df.at[idx, 'low_frac'] = frac_l
        
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
        print(idx)
        
df.to_excel(out_file, index = False)

#df_1 = df[df.water_pixels >= 6554].copy()