# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:03:55 2026

@author: K. Pikounis
"""

import pandas as pd

def extract_satellite_cases_exploded(file_path, n):
    # 1. Define thresholds
    thresholds = {
        'Akashiwo sanguinea': {'high': 100000, 'low': 50000},
        'Alexandrium spp': {'high': 100000, 'low': 10000},
        'Ceratium spp': {'high': 50000, 'low': 10000},
        'Cochlodinium spp': {'high': 100000, 'low': 50000},
        'Dinophysis spp': {'high': 500000, 'low': 100000},
        'Gymnodinium spp': {'high': 100000, 'low': 50000},
        'Lingulodinium polyedra': {'high': 100000, 'low': 50000},
        'Prorocentrum spp': {'high': 100000, 'low': 50000},
        'Pseudo nitzschia delicatissima group': {'high': 500000, 'low': 100000},
        'Pseudo nitzschia seriata group': {'high': 100000, 'low': 50000},
        'Other Diatoms': {'high': 100000, 'low': 50000},
        'Other Dinoflagellates': {'high': 100000, 'low': 50000},
        'Total Phytoplankton': {'high': 100000, 'low': 50000}
    }
    default_thresh = {'high': 100000, 'low': 50000}

    # 2. Load the data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    df['eventDate'] = pd.to_datetime(df['eventDate']).dt.date 
    all_selected_events = []
    count =0

    # 3. Process location by location to find the target dates
    for location, loc_data in df.groupby('Location_Code'):
        lat = loc_data['decimalLatitude'].iloc[0]
        lon = loc_data['decimalLongitude'].iloc[0]
        sid = loc_data['SampleID'].iloc[0]
        
        daily_data = loc_data.pivot_table(
            index='eventDate', 
            columns='scientificName', 
            values='organismQuantity', 
            aggfunc='sum'
        ).fillna(0)
        
        loc_hab_candidates = []
        loc_non_hab_candidates = []
        
        for date, row in daily_data.iterrows():
            is_hab = False
            is_non_hab = True
            triggering_organisms = []
            
            for species, qty in row.items():
                thresh = thresholds.get(species, default_thresh)
                if qty >= thresh['high']:
                    is_hab = True
                    triggering_organisms.append(f"{species} ({qty:,.0f})")
                if qty >= thresh['low']:
                    is_non_hab = False
            
            # Store the event metadata + the raw row data for exploding later
            case_record = {
                'Location_Code': location,
                'decimalLatitude': lat,
                'decimalLongitude': lon,
                'Event_Date': date,
                'Classification': 'HAB' if is_hab else 'Non-HAB',
                'Organisms_Above_Threshold': ", ".join(triggering_organisms) if is_hab else "None",
                'Event_Total_Biomass': row.sum(),
                '_species_data': row.to_dict() # Hidden dictionary to hold counts
            }
            
            if is_hab:
                loc_hab_candidates.append(case_record)
            elif is_non_hab:
                loc_non_hab_candidates.append(case_record)
                
        # Get Top/Bottom n
        loc_hab_candidates.sort(key=lambda x: x['Event_Total_Biomass'], reverse=True)
        top_habs = loc_hab_candidates[:n]
        
        loc_non_hab_candidates.sort(key=lambda x: x['Event_Total_Biomass'])
        bottom_non_habs = loc_non_hab_candidates[:n]
        
        all_selected_events.extend(top_habs)
        all_selected_events.extend(bottom_non_habs)
        
        print(location, " top :", len(top_habs), " bottom :", len(bottom_non_habs))
        
    # 4. EXPLODE THE DATA (Create one row per microorganism per event)
    exploded_rows = []
    rows = []
    for event in all_selected_events:
        count += 1
        base_info = {
            'Event_count': count,
            'Location_Code': event['Location_Code'],
            'Event_Date': event['Event_Date'],
            'decimalLatitude': event['decimalLatitude'],
            'decimalLongitude': event['decimalLongitude'],
            'Classification': event['Classification'],
            'Organisms_Above_Threshold': event['Organisms_Above_Threshold'],
            'Event_Total_Biomass': event['Event_Total_Biomass']
        }
        rows.append(base_info)
        
        # Add a new row for every species present on this date
        for species, qty in event['_species_data'].items():
            if qty > 0: # Skip the zeroes to keep the Excel file clean
                row_dict = base_info.copy()
                row_dict['scientificName'] = species
                row_dict['organismQuantity'] = qty
                exploded_rows.append(row_dict)
                
    # 5. Build Final DataFrame
    summary_df = pd.DataFrame(rows)
    results_df = pd.DataFrame(exploded_rows)
    return summary_df, results_df

# --- Run the code ---
# We use 5 here, but change it to however many events you want to extract
summary_df, final_exploded_df = extract_satellite_cases_exploded(r'C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\California\homogenized_calhabmap.csv', n=10)
final_exploded_df.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\California\top_and_bottom_10_HABs.xlsx", index = False)
summary_df.to_excel(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\California\summary_top_and_bottom_10_HABs.xlsx", index = False)