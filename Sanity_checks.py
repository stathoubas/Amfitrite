# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 11:53:18 2025

@author: K. Pikounis
"""

import os
import pandas as pd
import json

####################################################
#### Part 1 compare excels with folder contents ####
####################################################

folders = os.listdir(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\v_all")
real_folders = [i for i in folders if "." not in i]
set_f = set(real_folders)



df = pd.read_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\v_all\Master_summary - Copy.xlsx")
folders_df = [i.split("\\")[-1] for i in df.source_path.to_list()]
set_f_df = set(folders_df)


print(len(set_f))
print(len(set_f_df))
print(len(set_f.intersection(set_f_df)))
print(len(set_f_df.difference(set_f)))
print(len(set_f.difference(set_f_df)))


root = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\v_all"
c = 0
uids = []
all_uids = []
for f in  set_f.difference(set_f_df):
    if os.path.exists(os.path.join(root, f, "cyfi_lattice_predictions.csv")):
        print("cyfi_lattice_predictions.csv exists in ", f)
    else:
        metadata_path = os.path.join(root, f, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f: 
                meta = json.load(f)
            uids.append(meta.get("uid"))
            for p_d in meta.get("points_data"):
                uid = p_d.get("uid")
                all_uids.append(uid)
                
prob_uids_df = pd.read_csv(os.path.join(root, "problematic_uids.csv"))

uids_reasons = {}
all_uids_reasons = {}

for ui in uids:
    if not ui in prob_uids_df.uid.to_list():
        reason = "not in csv"
        print(ui)
    else:
        reason = prob_uids_df[prob_uids_df.uid == ui].iloc[0]["reason"]
    if reason in uids_reasons:
        uids_reasons[reason] += 1
    else:
        uids_reasons[reason] = 1
        
print(uids_reasons)


for ui in all_uids:
    if not ui in prob_uids_df.uid.to_list():
        reason = "not in csv"
        print(ui)
    else:
        reason = prob_uids_df[prob_uids_df.uid == ui].iloc[0]["reason"]
    if reason in all_uids_reasons:
        all_uids_reasons[reason] += 1
    else:
        all_uids_reasons[reason] = 1
        
print(all_uids_reasons)


####################################################
#### Part 1 compare wondows to cluster          ####
####################################################