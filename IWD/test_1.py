# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 13:04:23 2025

@author: K. Pikounis
"""

import pandas as pd
from utils import find_satelite_images, update_excel_report, predict_using_cyfi_pipeline, extract_and_save_tile

df = pd.read_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\tick tick bloom data\junk4.xlsx")
after2017 = df[(df.date >= "2017-01-01")]
#high_sev = after2017[(after2017.severity > 4)]
high_sev = after2017[(after2017.severity <= 1)]
print(df.columns)
print(high_sev.shape)


high_sev[high_sev.case == 317]
print(high_sev[high_sev.case == 317]["lat"].mean())
print(high_sev[high_sev.case == 317]["lon"].mean())
print(high_sev[high_sev.case == 317]["date"])

item_details = find_satelite_images(high_sev[high_sev.case == 317]["lat"].mean(), high_sev[high_sev.case == 317]["lon"].mean(), "2017-08-30")
item_details["abs_date_difference"] = np.abs(item_details["date_difference"])
item_details.sort_values(by = "per_clouds", inplace = True)

if item_details.iloc[0]["per_clouds"] < 2.0:
    sel_items = item_details[item_details["per_clouds"] < 2.0].copy()
    sel_items.sort_values(by = "abs_date_difference", inplace = True)
    sel_item = sel_items.iloc[0]
elif item_details.iloc[0]["per_clouds"] < 5.0:
    sel_items = item_details[item_details["per_clouds"] < 5.0].copy()
    sel_items.sort_values(by = "abs_date_difference", inplace = True)
    sel_item = sel_items.iloc[0]
elif item_details.iloc[0]["per_clouds"] < 10.0:
    sel_items = item_details[item_details["per_clouds"] < 5.0].copy()
    sel_items.sort_values(by = "abs_date_difference", inplace = True)
    sel_item = sel_items.iloc[0]
else:
    sel_item = sel_items.iloc[0]

