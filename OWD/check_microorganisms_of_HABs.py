# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:47:44 2026

@author: K. Pikounis
"""

import pandas as pd
import numpy as np

gen = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\tiles\amfitrite_open_waters_master_with_microorg.csv")

gen2 = gen.copy()
gen2["microorg"] = ""
   

#HABSOS_NOAA
HABSOS_NOAA = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\prd OAS\homogenized_habsos_after_Florida_v3.1_FL_and_noFL_decoupled_FINAL_v2_with_results_processed.csv")
idxs = set([int(i) for i in gen[gen.dataset == "HABSOS_NOAA"].id_x])

for i in idxs:
    temp = HABSOS_NOAA[HABSOS_NOAA.id_x == i]
    li = []
    for it, row in temp.iterrows():
        li.append(row["GENUS"]+"&"+row["SPECIES"]+":"+str(row["organismQuantity"]))
    st = "_".join(li)
    print(st)
    gen2.loc[(gen2.dataset == "HABSOS_NOAA") & (gen2.id_x == str(i)), "microorg"] = st


#HAEDAT
hadeat = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS dataset v2\datasets_1_3_4_6_v5_decoupled_FINAL_v2_v3_with_results_processed.csv")
idxs = set(gen[gen.dataset == "HAEDAT"].id_x)
for i in idxs:
    temp = hadeat[hadeat.id_x == i]
    li = []
    for it, row in temp.iterrows():
        li.append(row["scientificName"]+":"+str(row["organismQuantity"]))
    st = "_".join(li)
    print(st)
    gen2.loc[(gen2.dataset == "HAEDAT") & (gen2.id_x == i), "microorg"] = st


#Florida_HABs
flor = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\Florida\Historic_Harmful_Algal_Bloom_Events_2015_-_2023_homogenised_v3_decoupled_FINAL_with_results_processed.csv")
idxs = set([int(i) for i in gen[gen.dataset == "Florida_HABs"].id_x])

for i in idxs:
    temp = flor[flor.id_x == i]
    li = []
    for it, row in temp.iterrows():
        li.append(row["scientificName"]+":"+str(row["organismQuantity"]))
    st = "_".join(li)
    print(st)
    gen2.loc[(gen2.dataset == "Florida_HABs") & (gen2.id_x == str(i)), "microorg"] = st


#CalHABMAP
cal = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\California\homogenized_calhabmap_decoupled_FINAL_v3.1_with_results_processed.csv")
idxs = set(gen[gen.dataset == "CalHABMAP"].SampleID)

for i in idxs:
    temp = cal[cal.SampleID == str(int(i))]
    li = []
    for it, row in temp.iterrows():
        li.append(row["scientificName"]+":"+str(row["organismQuantity"]))
    st = "_".join(li)
    print(st)
    gen2.loc[(gen2.dataset == "CalHABMAP") & (gen2.SampleID == i), "microorg"] = st

#OBIS_HAB_Med
idxs = set(gen[gen.dataset == "OBIS_HAB_Med"].id_x)
for i in idxs:
    temp = hadeat[hadeat.id_x == i]
    li = []
    for it, row in temp.iterrows():
        li.append(row["scientificName"]+":"+str(row["organismQuantity"]))
    st = "_".join(li)
    print(st)
    gen2.loc[(gen2.dataset == "OBIS_HAB_Med") & (gen2.id_x == i), "microorg"] = st
   

#NAA_plankton
naa = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\North American Arctic region\homogenized_arctic_plankton_v1_decoupled_FINAL_v2_v3_with_results_processed.csv")
idxs = set(gen[gen.dataset == "NAA_plankton"].sat_item)

for i in idxs:
    temp = naa[naa.sat_item == i]
    li = []
    for it, row in temp.iterrows():
        li.append(row["verbatimScientificName"]+":"+str(row["individualCount"]))
    st = "_".join(li)
    print(st)
    gen2.loc[(gen2.dataset == "NAA_plankton") & (gen2.sat_item == i), "microorg"] = st
   


#REPHY
re1 = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\SEANOE REPHY\REPHY_Manche_Atlantique_1987-2022_decoupled_FINAL_v2_with_results_processed.csv")
re2 = pd.read_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\SEANOE REPHY\REPHY_Med_1987-2022_decoupled_FINAL_v2_with_results_processed.csv")
re= pd.concat([re1, re2])
idxs = set(gen[gen.dataset == "REPHY"]["Lieu de surveillance : Identifiant"])

for i in idxs:
    temp = re[re["Lieu de surveillance : Identifiant"] == i]
    li = []
    for it, row in temp.iterrows():
        if not pd.isna(row["scientificName"]) and not pd.isna(row["scientificName"]):
            li.append(row["scientificName"]+":"+str(row["organismQuantity"]))
    st = "_".join(li)
    print(st)
    gen2.loc[(gen2.dataset == "REPHY") & (gen2["Lieu de surveillance : Identifiant"] == i), "microorg"] = st
 
    
def get_top_microorg(cell_str):
    if pd.isna(cell_str) or not str(cell_str).strip():
        return None
    
    parts = str(cell_str).split("_")
    max_qty = -1.0
    top_name = None
    
    for part in parts:
        if ":" in part:
            separator = ":"
        elif "&" in part:
            separator = "&"
        else:
            continue 
            
        try:
            name, qty_str = part.split(separator, 1)
            qty = float(qty_str)
            
            if qty > max_qty:
                max_qty = qty
                top_name = name
        except ValueError:
            continue
            
    return top_name

da = ['CalHABMAP', 'Florida_HABs', 'HABSOS_NOAA', 'HAEDAT', 'NAA_plankton', 'OBIS_HAB_Med', 'REPHY']

gen2 = gen2[gen2.dataset.apply(lambda x: x in da)]

gen2['top_microorg'] = gen2['microorg'].apply(get_top_microorg)

# 3. Aggregate overall names (INCLUDING blanks/NaNs)
print("=== OVERALL DATASET ===")
# dropna=False ensures NaNs are counted and factored into the percentage
overall_counts = gen2['top_microorg'].value_counts(dropna=False)
overall_pct = gen2['top_microorg'].value_counts(normalize=True, dropna=False) * 100

overall_summary = pd.DataFrame({
    'Count': overall_counts, 
    'Percentage (%)': overall_pct.round(2)
})
# Rename the empty index strictly for cleaner printing
overall_summary.index = overall_summary.index.fillna('NaN (Blank/No Data)')
print(overall_summary)
print("\n" + "="*50 + "\n")

# 4. Aggregate by specific datasets
datasets_to_check = [
    "HABSOS_NOAA", "HAEDAT", "CalHABMAP", 
    "Florida_HABs", "NAA_plankton", "OBIS_HAB_Med", "REPHY"
]

for ds in datasets_to_check:
    print(f"=== DATASET: {ds} ===")
    
    # Filter for the specific dataset (do NOT drop NAs here)
    ds_data = gen2[gen2['dataset'] == ds]
    
    if ds_data.empty:
        print("No rows found for this dataset.\n")
        continue
        
    # Calculate counts and percentages including NaNs
    ds_counts = ds_data['top_microorg'].value_counts(dropna=False)
    ds_pct = ds_data['top_microorg'].value_counts(normalize=True, dropna=False) * 100
    
    ds_summary = pd.DataFrame({
        'Count': ds_counts, 
        'Percentage (%)': ds_pct.round(2)
    })
    
    ds_summary.index = ds_summary.index.fillna('NaN (Blank/No Data)')
    
    print(ds_summary)
    print("-" * 50 + "\n")    