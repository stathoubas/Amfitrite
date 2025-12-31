# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 11:16:49 2025

@author: K. Pikounis
"""

import pandas as pd
import numpy as np
import os
import json
from collections import Counter
from dateutil.relativedelta import relativedelta


def generate_missing_dates_only(date_input_str):
    """
    Parses a string of dates, identifies the start date, and generates
    only the MISSING dates required to ensure one date per month
    for a full year.
    
    Args:
        date_input_str (str): "2021-09-11" or "2021-09-11,2020-06-23"
        
    Returns:
        list: A sorted list of ONLY the newly created date strings (YYYY-MM-DD).
    """
    # 1. Parse Input
    raw_dates = [d.strip() for d in date_input_str.split(',')]
    try:
        # distinct original dates sorted
        original_dt_list = sorted(list(set([pd.to_datetime(d) for d in raw_dates])))
    except Exception as e:
        return f"Error parsing dates: {e}"

    if not original_dt_list:
        return []
    
    start_date = original_dt_list[0]
    
    # List to store ONLY the new dates we create
    new_generated_dates = []
    
    # Track the current reference date for day-of-month continuity
    # (e.g. if we have June 23, July should be 23. If we find July 18, Aug should be 18)
    current_ref_date = start_date

    # 2. Iterate for the next 11 months to complete a 1-year cycle
    for i in range(1, 12):
        # Calculate which month/year we are looking for relative to the absolute start
        target_step_date = start_date + relativedelta(months=i)
        target_year = target_step_date.year
        target_month = target_step_date.month
        
        # 3. Check if we already have a date provided for this specific Month/Year
        # We search the original input list
        existing_date_in_month = None
        for d in original_dt_list:
            if d.year == target_year and d.month == target_month:
                existing_date_in_month = d
                break
        
        if existing_date_in_month:
            # We HAVE a date. Do not generate one.
            # Update reference so future gaps follow this date's day-of-month
            current_ref_date = existing_date_in_month
        else:
            # GAP DETECTED. Generate a new date.
            # We base it on the current_ref_date + 1 month
            new_date = current_ref_date + relativedelta(months=1)
            
            new_generated_dates.append(new_date)
            
            # Update reference to this newly created date
            current_ref_date = new_date

    # 4. Format Output
    return [d.strftime("%Y-%m-%d") for d in new_generated_dates]


df = pd.read_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\from_server\Master_summary_all_parts_with_flags_v3.xlsx")
'''
cases = list(set(df.case))
cases_low = list(set(df[df.Class == "Low"].case))

cases_not_low = [i for i in cases if i not in cases_low]
cases_not_low.sort()
'''
cases = list(set(df.case))
cases_high = list(set(df[df.Class == "High"].case))

cases_not_high = [i for i in cases if i not in cases_high]
cases_not_high.sort()

cs = []
lats = []
lons = []
dates_outer = []
target_dates = []
vs = []
all_counts = []

for c in cases_not_high:
    df_c = df[df.case == c]
    paths = list(set(df_c.source_path))
    
    junk = []
    for p in paths:
        if "v_all" in p:
            junk.append("v_all")
        if "v10_to_100" in p:
            junk.append("v10_to_100")
    if len(set(junk)) > 1:
        paths = [p for p in paths if "v_all" in p]
    
    lats_lons = []
    dates = []
    counts = []
    
    for p in paths:
        meta_path = os.path.join(p , "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f: 
                meta_data = json.load(f)
                lats_lons.append( (meta_data.get('center_lat'), meta_data.get('center_lon')) )
                dates.append(meta_data.get('date'))
                counts.append(meta_data.get('High counts')+meta_data.get('Moderate counts ')+meta_data.get('Low counts '))
    lat, lon = Counter(lats_lons).most_common(1)[0][0]
    cs.append(c)
    lats.append(lat)
    lons.append(lon)
    dates_outer.append(",".join(dates))
    new_dates = generate_missing_dates_only(",".join(dates))
    target_dates.append(",".join(new_dates))
    if "v_all" in paths[0]:
        vs.append("v_all")
    else:
        vs.append("v10_to_100")
    all_counts.append(np.mean(counts))
        
    
#df_cases_not_low_lat_lon = pd.DataFrame({'case': cs, 'lat': lats, 'lon': lons, "counts":all_counts, "dates":dates_outer, "target_dates":target_dates, "version":vs})
#df_cases_not_low_lat_lon.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\from_server\cases_not_low_lat_lon.xlsx", index=False)

df_cases_not_high_lat_lon = pd.DataFrame({'case': cs, 'lat': lats, 'lon': lons, "counts":all_counts, "dates":dates_outer, "target_dates":target_dates, "version":vs})
df_cases_not_high_lat_lon.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\sat data test\from_server\cases_not_high_lat_lon.xlsx", index=False)

