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

root = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test"

f_l = [i for i in os.listdir(os.path.join(root, "v_all")) if "." not in i]
f_s = [i for i in os.listdir(os.path.join(root, "from_server", "v4_3")) if "." not in i]

s_l = set(f_l)
s_s = set(f_s)

print(len(s_l))
print(len(s_s))
print(len(s_l.intersection(s_s)))
print(len(s_l.difference(s_s)))
print(len(s_s.difference(s_l)))

more_in_s_l_noH = set([i.split("_H")[0] for i in s_l.difference(s_s)])
more_in_s_s_noH = set([i.split("_H")[0] for i in s_s.difference(s_l)])

print(len(s_l.difference(s_s)), len(more_in_s_l_noH))
print(len(s_s.difference(s_l)), len(more_in_s_s_noH))
print(len(more_in_s_l_noH.intersection(more_in_s_s_noH)))
print(len(more_in_s_l_noH.difference(more_in_s_s_noH)))
print(len(more_in_s_s_noH.difference(more_in_s_l_noH)))

print(more_in_s_l_noH.difference(more_in_s_s_noH))
print(more_in_s_s_noH.difference(more_in_s_l_noH))

"""
Τελικά τα lapto έχει 21 παραπάνω ενω ο server 1 παρπάνω.
Το 1 παραπάνω του server ταιριάζει με ενα απο ατ 21 παραπάω του laptop:
από server: ase361_abun30_365_date2019-08-25_H23_M51_L0  από laptop: case361_abun30_365_date2019-08-20_H140_M0_L0
καλύτερη η εικόνα το laptop!!!
"""


more_in_lap_tup = []
for i in s_l.difference(s_s):
    more_in_lap_tup.append((i, i.split("_H")[0]))
    

more_in_sev_tup = []
more_in_sev_full = []
more_in_sev_no_H = []
for i in s_s.difference(s_l):
    more_in_sev_tup.append((i, i.split("_H")[0]))
    more_in_sev_full.append(i)
    more_in_sev_no_H.append(i.split("_H")[0])
    
c = 0
c_s1_l1 = 0
c_s1_l0 = 0
c_s0_l1 = 0
c_s0_l0 = 0
c_weird = 0
for full, no_h in more_in_lap_tup:
    if no_h in more_in_sev_no_H:
        s_name = more_in_sev_full[more_in_sev_no_H.index(no_h)]
        csv_in_l = os.path.exists(os.path.join(root, "v_all", full, "cyfi_lattice_predictions.csv"))
        csv_in_s = os.path.exists(os.path.join(root, "from_server", "v4_3", s_name, "cyfi_lattice_predictions.csv"))
        
        if csv_in_l and csv_in_s:
            c_s1_l1 += 1
        elif not csv_in_l and csv_in_s:
            c_s1_l0 += 1
        elif csv_in_l and not csv_in_s:
            c_s0_l1 += 1
        elif not csv_in_l and not csv_in_s:
            c_s0_l0 += 1
        else:
            c_weird += 1
        
    else:
        c += 1


print("in both:", c_s1_l1)
print("in laptop only:", c_s0_l1)
print("in server only:", c_s1_l0)
print("in neither:", c_s0_l0)
print("weird????:", c_weird)
print("no match:", c)


########################################
#### lets do the same for 10 to 100 ####
########################################


root = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test"

f_l = [i for i in os.listdir(os.path.join(root, "v10_to_100")) if "." not in i]
f_s = [i for i in os.listdir(os.path.join(root, "from_server", "v4_3_10_to_100")) if "." not in i]

s_l = set(f_l)
s_s = set(f_s)

print(len(s_l))
print(len(s_s))
print(len(s_l.intersection(s_s)))
print(len(s_l.difference(s_s)))
print(len(s_s.difference(s_l)))

more_in_s_l_noH = set([i.split("_H")[0] for i in s_l.difference(s_s)])
more_in_s_s_noH = set([i.split("_H")[0] for i in s_s.difference(s_l)])

print(len(s_l.difference(s_s)), len(more_in_s_l_noH))
print(len(s_s.difference(s_l)), len(more_in_s_s_noH))
print(len(more_in_s_l_noH.intersection(more_in_s_s_noH)))
print(len(more_in_s_l_noH.difference(more_in_s_s_noH)))
print(len(more_in_s_s_noH.difference(more_in_s_l_noH)))

print(more_in_s_l_noH.difference(more_in_s_s_noH))
print(more_in_s_s_noH.difference(more_in_s_l_noH))

"""
Τελικά o server έχει 1 παραπάνω. Το άβαλα και στο laptop, άρα όλα κομπλέ!

"""


more_in_lap_tup = []
for i in s_l.difference(s_s):
    more_in_lap_tup.append((i, i.split("_H")[0]))
    

more_in_sev_tup = []
more_in_sev_full = []
more_in_sev_no_H = []
for i in s_s.difference(s_l):
    more_in_sev_tup.append((i, i.split("_H")[0]))
    more_in_sev_full.append(i)
    more_in_sev_no_H.append(i.split("_H")[0])
    
c = 0
c_s1_l1 = 0
c_s1_l0 = 0
c_s0_l1 = 0
c_s0_l0 = 0
c_weird = 0
for full, no_h in more_in_lap_tup:
    if no_h in more_in_sev_no_H:
        s_name = more_in_sev_full[more_in_sev_no_H.index(no_h)]
        csv_in_l = os.path.exists(os.path.join(root, "v10_to_100", full, "cyfi_lattice_predictions.csv"))
        csv_in_s = os.path.exists(os.path.join(root, "from_server", "v4_3_10_to_100", s_name, "cyfi_lattice_predictions.csv"))
        
        if csv_in_l and csv_in_s:
            c_s1_l1 += 1
        elif not csv_in_l and csv_in_s:
            c_s1_l0 += 1
        elif csv_in_l and not csv_in_s:
            c_s0_l1 += 1
        elif not csv_in_l and not csv_in_s:
            c_s0_l0 += 1
        else:
            c_weird += 1
        
    else:
        c += 1


print("in both:", c_s1_l1)
print("in laptop only:", c_s0_l1)
print("in server only:", c_s1_l0)
print("in neither:", c_s0_l0)
print("weird????:", c_weird)
print("no match:", c)
