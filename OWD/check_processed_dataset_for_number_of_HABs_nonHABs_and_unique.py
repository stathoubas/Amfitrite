# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:47:28 2026

@author: K. Pikounis
"""

import pandas as pd

df3 = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\prd OAS\homogenized_habsos_after_Florida_v3.1_FL_and_noFL_decoupled_FINAL_v2_with_results_processed.csv")
temp_df = df3[df3.usable == 1]
all_HAB_tiles = []
all_nonHAB_tiles = []
for it, row in temp_df.iterrows():
    if not pd.isna(row["tiles_HABs"]):
        if len(row["tiles_HABs"]) > 3:
            all_HAB_tiles.extend([str(int(row["cluster_ID"])) + "_"+ i.strip().split("_")[1] for i in  row["tiles_HABs"].split(",")])
            
    if not pd.isna(row["tiles_nonHABs"]):
        if len(row["tiles_nonHABs"]) > 3:
            all_nonHAB_tiles.extend([str(int(row["cluster_ID"])) + "_"+ i.strip().split("_")[1] for i in  row["tiles_nonHABs"].split(",")])

print(len(all_HAB_tiles))
print(len(set(all_HAB_tiles)))
print(len(set(temp_df[temp_df.is_HAB == "Yes"].cluster_ID)))

print("-----------------------")

print(len(all_nonHAB_tiles))
print(len(set(all_nonHAB_tiles)))
print(len(set(temp_df[temp_df.is_HAB != "Yes"].cluster_ID)))