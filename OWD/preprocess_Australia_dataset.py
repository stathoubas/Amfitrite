# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 22:47:47 2026

@author: K. Pikounis
"""

import pandas as pd
import numpy as np

def categorize_hab(species_name):
    """
    Explicitly categorizes every organism from the IMOS dataset into Yes, Probably, or No.
    """
    name = str(species_name).lower()
    
    # =========================================================================
    # YES: Known highly toxic species, severe high-biomass bloomers, 
    # and cyanobacteria that cause hypoxia or fish kills.
    # =========================================================================
    hab_yes = [
        'alexandrium', 'amphidinium', 'dinophysis', 'dictyocha', 'gonyaulax',
        'gymnodinium', 'gyrodinium', 'karenia', 'karlodinium', 'lyngbya', 
        'margalefidinium', 'noctiluca', 'phalacroma', 'phaeocystis', 
        'prorocentrum', 'pseudo-nitzschia', 'ptychodiscus', 'trichodesmium'
    ]
    
    # =========================================================================
    # PROBABLY: Organisms that are usually benign but can cause harm under 
    # certain conditions (e.g., massive oxygen-depleting red tides, or having 
    # serrated physical spines that damage fish gills).
    # =========================================================================
    hab_prob = [
        'asterionellopsis', 'cerataulina', 'chaetoceros', 'cylindrotheca', 
        'dactyliosolen', 'detonula', 'eucampia', 'guinardia', 'leptocylindrus', 
        'melosira', 'rhizosolenia', 'scrippsiella', 'skeletonema', 
        'thalassiosira', 'tripos', 'ceratium'
    ]
    
    # =========================================================================
    # NO: Explicitly benign diatoms, heterotrophic zooplankton (ciliates/
    # tintinnids), fungi, radiolarians, and harmless dinoflagellates.
    # =========================================================================
    hab_no = [
        'acanthoica', 'acanthostomella', 'actinocyclus', 'amphilithium', 
        'amphisolenia', 'amphora', 'amphorellopsis', 'amphorides', 
        'ascampbelliella', 'aspergillus', 'asterionella', 'asteromphalus', 
        'azpeitia', 'bacillaria', 'bacteriastrum', 'bellerochea', 'biddulphia', 
        'braarudosphaera', 'ceratocorys', 'climacocylis', 'climacodium', 
        'cocconeis', 'codonella', 'codonellopsis', 'corethron', 'coscinodiscus', 
        'cymatocylis', 'cyttarocylis', 'dadayiella', 'daturella', 'diaphanoeca', 
        'dictyocysta', 'diploneis', 'ditylum', 'ebria', 'entomoneis', 'ephemera', 
        'epiplocylis', 'eutintinnus', 'favella', 'fragilaria', 'fragilariopsis', 
        'gazelletta', 'globotoralia', 'goniodoma', 'grammatophora', 'haslea', 
        'helicostomella', 'hemiaulus', 'lauderia', 'licmophora', 'lioloma', 
        'lithodesmium', 'mastogloia', 'membraneis', 'mesoporos', 'navicula', 
        'neostreptotheca', 'nitzschia', 'octactis', 'odontella', 'ornithocercus', 
        'oxyphysis', 'oxytoxum', 'palmerina', 'paralia', 'parundella', 
        'peridinium', 'pinnularia', 'plagiotropis', 'planktoniella', 
        'pleurosigma', 'podolampas', 'porosira', 'proboscia', 'proplectella', 
        'protoperidinium', 'protorhabdonella', 'pseudosolenia', 'pterosperma', 
        'pyrocystis', 'pyrophacus', 'rhabdonella', 'rhabdonellopsis', 'richelia', 
        'rotalia', 'sagenoarium', 'salpingella', 'shionodiscus', 
        'steenstrupiella', 'stephanopyxis', 'striatella', 'thalassionema', 
        'thalassiothrix', 'tintinnopsis', 'torodinium', 'triceratium', 
        'trichotoxon', 'trigonium', 'undella', 'xystonella', 'xystonellopsis',
        'ciliate', 'radiolarian', 'foraminifera', 'zooplankton', 'egg', 
        'fungal', 'spine', 'pollen', 'paint', 'cyanobacteria', 'autotroph', 
        'heterotroph', 'diatom'
    ]

    # Check lists (using genus-level matching)
    for genus in hab_yes:
        if genus in name: return 'Yes'
        
    for genus in hab_prob:
        if genus in name: return 'Probably'
        
    for genus in hab_no:
        if genus in name: return 'No'
        
    # If the script finds a completely new column we didn't account for:
    return 'UNKNOWN - CHECK'

def prep_australia_data(input_csv, output_csv):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Identify metadata columns 
    metadata_cols = [
        'TripCode', 'Sample_ID', 'Region', 'Latitude', 'Longitude', 
        'SampleTime_UTC', 'SampleTime_Local', 'Year_Local', 'Month_Local', 
        'Day_Local', 'Time_Local24hr', 'SatSST_degC', 'SatChlaSurf_mgm3', 
        'PCI', 'SampleVolume_m3'
    ]
    
    metadata_cols = ["Project","StationName","StationCode","Latitude","Longitude",
                     "TripCode","SampleTime_UTC","SampleTime_Local","Year_Local",
                     "Month_Local","Day_Local","Time_Local24hr","SampleDepth_m",
                     "CTDSST_degC","CTDChlaSurf_mgm3","CTDSalinity_psu","Method"]

    
    id_vars = [col for col in metadata_cols if col in df.columns]
    
    print("Melting data from Wide to Long format...")
    # 2. MELT the dataframe
    df_long = pd.melt(
        df, 
        id_vars=id_vars, 
        var_name='scientificName', 
        value_name='organismQuantity'
    )
    
    # UNIT CONVERSION: Convert Cells/m3 to Cells/L so part1.py doesn't break 🚨
    df_long['organismQuantity'] = df_long['organismQuantity'] / 1000.0
    
    # 3. Drop rows where cell count is 0
    initial_rows = len(df_long)
    df_long = df_long[df_long['organismQuantity'] > 0].copy()
    print(f"Dropped {initial_rows - len(df_long)} rows with zero counts.")
    
    # 4. Standardize columns for part1.py
    df_long['eventDate'] = pd.to_datetime(df_long['SampleTime_UTC'], errors='coerce').dt.strftime('%Y-%m-%d')
    df_long.rename(columns={
        'Sample_ID': 'id_x',
        'Latitude': 'decimalLatitude',
        'Longitude': 'decimalLongitude'
    }, inplace=True)
    
    # 5. APPLY THE EXPLICIT HAB FILTER
    print("Classifying species for HAB potential...")
    df_long['can_create_HAB'] = df_long['scientificName'].apply(categorize_hab)
    
    # Verify if any UNKNOWNs popped up
    unknowns = df_long[df_long['can_create_HAB'] == 'UNKNOWN - CHECK']['scientificName'].unique()
    if len(unknowns) > 0:
        print(f"\nWARNING: Found {len(unknowns)} species not in the dictionary:")
        for u in unknowns:
            print(f"  - {u}")
    
    # 6. Save it
    df_long.to_csv(output_csv, index=False)
    print(f"\nSuccess! Saved formatted data to {output_csv}")


if __name__ == "__main__":
    # Change these to match your actual file names
    INPUT_FILE = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\AODN - australia\IMOS_-_Phytoplankton_Abundance_and_Biovolume_(CPR)-abundance_-_species_data.csv"
    OUTPUT_FILE = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\AODN - australia\Australia_IMOS_Ready_for_Pipeline.csv"
    
    prep_australia_data(INPUT_FILE, OUTPUT_FILE)