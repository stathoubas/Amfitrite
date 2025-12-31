# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 11:53:16 2025

@author: K. Pikounis
"""

import pandas as pd
import numpy as np
from datetime import timedelta

def generate_balanced_dates(date_list_str):
    """
    Takes a list of date strings, calculates average/range, and returns 
    12 dates covering at least a 1-year window, applying a lower limit shift.
    
    Args:
        date_list_str (list): List of date strings (e.g. ['2017-04-26'])
        
    Returns:
        list: 12 date strings (YYYY-MM-DD)
    """
    LOWER_LIMIT_STR = "2017-03-01"
    
    # 1. Parse dates
    dt_dates = pd.to_datetime(date_list_str)
    
    # FIX: Convert to Series to ensure .mean() exists and cast to int64
    # (timestamps are stored as int64 nanoseconds in pandas)
    avg_ts = pd.Series(dt_dates).astype(np.int64).mean()
    avg_date = pd.to_datetime(avg_ts)
    
    min_date = dt_dates.min()
    max_date = dt_dates.max()
    date_range = max_date - min_date
    
    # 2. Determine Window Start and End
    one_year = timedelta(days=365)
    
    if len(dt_dates) == 1 or date_range < one_year:
        # Scenario A: Short range or single date -> Create 1 Year Window centered on Average
        half_window = timedelta(days=182) # approx 6 months
        
        window_start = avg_date - half_window
        window_end = avg_date + half_window
        
        # Ensure window is exactly at least 365 days for the 12 points
        if (window_end - window_start) < one_year:
             window_end = window_start + one_year
             
    else:
        # Scenario B: Long range -> Cover the existing range
        window_start = min_date
        window_end = max_date

    # 3. Apply Lower Limit Constraint (2017-03-01)
    limit_date = pd.Timestamp(LOWER_LIMIT_STR)
    
    if window_start < limit_date:
        # Calculate how much we need to shift
        shift_delta = limit_date - window_start
        
        # Shift both start and end forward to preserve the window size
        window_start = limit_date
        window_end = window_end + shift_delta

    # 4. Generate exactly 12 dates
    # 'periods=12' ensures we get exactly 12 points evenly spaced
    generated_dates = pd.date_range(start=window_start, end=window_end, periods=12)
    
    # 5. Format and return
    return [d.strftime("%Y-%m-%d") for d in generated_dates]


############## excels with cases after 2017 both v_all and v_10_to_100  #####################
df_t = pd.read_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\tick tick bloom data\clustered_only_square_2560m.xlsx")
after2017 = df_t[(df_t.date >= "2017-01-01")]
after2017_near_water = after2017[after2017["distance_to_water_m"] <= 10]
after2017_far_from_water = after2017[(after2017["distance_to_water_m"] > 10) & (after2017["distance_to_water_m"] <= 100)]

all_cases_near_water = list(set(after2017_near_water.case.to_list()))
all_cases_far_from_water = list(set(after2017_far_from_water.case.to_list()))

df = pd.read_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\from_server\Master_summary_all_parts_with_flags.xlsx")
cases_done = list(set(df.case.to_list()))

rest_cases_near_water = [i for i in all_cases_near_water if i not in cases_done]

print("so, ", len(rest_cases_near_water), "rest_cases_near_water")
cases = []
lats = []
lons = []
dates_outer = []
ver = []
for c in rest_cases_near_water:
    cases.append(c)
    temp_df = after2017[after2017.case == c]
    lats.append(temp_df.lat.mean())
    lons.append(temp_df.lon.mean())
    target_dates = generate_balanced_dates([str(i).split(" ")[0] for i in temp_df.date.to_list()])
    dates_outer.append(",".join(target_dates))
    ver.append("v_all")
    

cases_done.extend(cases)    
rest_cases_far_from_water = [i for i in all_cases_far_from_water if i not in cases_done]

print("and, ", len(rest_cases_far_from_water), "rest_cases_far_from_water")

for c in rest_cases_far_from_water:
    cases.append(c)
    temp_df = after2017[after2017.case == c]
    lats.append(temp_df.lat.mean())
    lons.append(temp_df.lon.mean())
    target_dates = generate_balanced_dates([str(i).split(" ")[0] for i in temp_df.date.to_list()])
    dates_outer.append(",".join(target_dates))
    ver.append("v10_to_100")

    
df_cases_not_yet_processes_after_2017 = pd.DataFrame({'case': cases, 'lat': lats, 'lon': lons, "counts":None, "dates":None, "target_dates":dates_outer, "version":ver})
df_cases_not_yet_processes_after_2017.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\extra_cases_not_previously_processed_after_2017\cases_not_not_previously_processed_after_2017.xlsx", index=False)

parts = np.array_split(df_cases_not_yet_processes_after_2017, 2)
for i in parts:
    print(i.shape)
count = 0
for i in parts:
    count += 1
    i.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\extra_cases_not_previously_processed_after_2017\cases_not_not_previously_processed_after_2017_part"+str(count)+".xlsx", index = False)
 
cases_done.extend(cases)
cases_done = list(set(cases_done))

############## excels with cases before 2017 both v_all and v_10_to_100  #####################
pre_defined_dates = ["2019-01-15", "2019-03-15", "2019-05-15", "2019-07-15", "2019-09-15", "2019-11-15",
                     "2020-02-15", "2020-04-15", "2020-06-15", "2020-08-15", "2020-10-15", "2020-12-15"]
pre_defined_dates_str = ",".join(pre_defined_dates)

before2017 = df_t[(df_t.date < "2017-01-01")]
before2017_near_water = before2017[before2017["distance_to_water_m"] <= 10]
before2017_far_from_water = before2017[(before2017["distance_to_water_m"] > 10) & (before2017["distance_to_water_m"] <= 100)]

all_cases_near_water = list(set(before2017_near_water.case.to_list()))
all_cases_far_from_water = list(set(before2017_far_from_water.case.to_list()))

rest_cases_near_water = [i for i in all_cases_near_water if i not in cases_done]

print("so, ", len(rest_cases_near_water), "rest_cases_near_water before 2017")

cases = []
lats = []
lons = []
dates_outer = []
ver = []
for c in rest_cases_near_water:
    cases.append(c)
    temp_df = before2017[before2017.case == c]
    lats.append(temp_df.lat.mean())
    lons.append(temp_df.lon.mean())
    dates_outer.append(pre_defined_dates_str)
    ver.append("v_all")
    

cases_done.extend(cases)    
rest_cases_far_from_water = [i for i in all_cases_far_from_water if i not in cases_done]

print("and, ", len(rest_cases_far_from_water), "rest_cases_far_from_water before 2017")

for c in rest_cases_far_from_water:
    cases.append(c)
    temp_df = before2017[before2017.case == c]
    lats.append(temp_df.lat.mean())
    lons.append(temp_df.lon.mean())
    dates_outer.append(pre_defined_dates_str)
    ver.append("v10_to_100")

cases_done.extend(cases)    
cases_done = list(set(cases_done))
    
df_cases_not_yet_processes_before_2017 = pd.DataFrame({'case': cases, 'lat': lats, 'lon': lons, "counts":None, "dates":None, "target_dates":dates_outer, "version":ver})
df_cases_not_yet_processes_before_2017.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\extra_cases_not_previously_processed_before_2017\cases_not_not_previously_processed_before_2017.xlsx", index=False)


############## excels with cases v_100_to_500  #####################


far_from_water_100_to_500 = df_t[(df_t["distance_to_water_m"] > 100) & (df_t["distance_to_water_m"] <= 500)]

all_cases_far_from_water_100_to_500 = list(set(far_from_water_100_to_500.case.to_list()))
rest_cases_far_from_water_100_to_500 = [i for i in all_cases_far_from_water_100_to_500 if i not in cases_done]

print("and, ", len(rest_cases_far_from_water_100_to_500), "rest_cases_far_from_water_100_to_500")

cases = []
lats = []
lons = []
dates_outer = []
ver = []
for c in rest_cases_far_from_water_100_to_500:
    cases.append(c)
    temp_df = far_from_water_100_to_500[far_from_water_100_to_500.case == c]
    lats.append(temp_df.lat.mean())
    lons.append(temp_df.lon.mean())
    dates_outer.append(pre_defined_dates_str)
    ver.append("v100_to_500")

cases_done.extend(cases)    
cases_done = list(set(cases_done))
    
df_cases_not_yet_processes_before_2017 = pd.DataFrame({'case': cases, 'lat': lats, 'lon': lons, "counts":None, "dates":None, "target_dates":dates_outer, "version":ver})
df_cases_not_yet_processes_before_2017.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\extra_cases_not_previously_processed_100_to_500\cases_not_not_previously_processed_100_to_500.xlsx", index=False)


