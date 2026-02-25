# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 22:37:28 2026

@author: K. Pikounis
"""

import os
import pandas as pd


root_folder = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS datasets"
df1 = pd.read_csv(os.path.join(root_folder, "1 Harmful Algal Event Database (HAEDAT)", "all_data.csv"))
df3 = pd.read_csv(os.path.join(root_folder, "3 Occurrences of HAB species in the Mediterranean", "homogenized_dataset_3.csv"))
df4 = pd.read_csv(os.path.join(root_folder, "4 Harmful algal blooms in South America except Venezuela, Colombia and the Guyanas", "homogenized_dataset_4.csv"))
df6 = pd.read_csv(os.path.join(root_folder, "6 Global distribution of the genera Gambierdiscus and Fukuyoa", "homogenized_dataset_6.csv"))

df1["dataset"] = "1"
df3["dataset"] = "3"
df4["dataset"] = "4"
df6["dataset"] = "6"

final_df = pd.concat([df1, df3, df4, df6], ignore_index=True, sort=False)

final_df.to_csv(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS datasets\datasets_1_3_4_6.csv", index = False)


#################################################

df = pd.read_csv(r'C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS datasets\datasets_1_3_4_6_v2.csv')

mask = df['date_ok'] == 2

def get_middle_date(date_str):
    try:
        # Χωρισμός του string στο "/"
        start_str, end_str = date_str.split('/')

        # Μετατροπή σε datetime objects
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        # Υπολογισμός μέσης ημερομηνίας: Αρχή + (Διαφορά / 2)
        middle_date = start_date + (end_date - start_date) / 2

        # Επιστροφή μόνο της ημερομηνίας σε μορφή YYYY-MM-DD
        return middle_date.strftime('%Y-%m-%d')
    except:
        return date_str

# 3. Εφαρμογή της συνάρτησης μόνο στις επιλεγμένες γραμμές
df.loc[mask, 'eventDate'] = df.loc[mask, 'eventDate'].apply(get_middle_date)

# 4. Αποθήκευση του αποτελέσματος σε νέο αρχείο Excel
df.to_csv(r'C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS datasets\datasets_1_3_4_6_v3.csv', index=False)

#######################################################

import numpy as np

mask2 = (df['date_ok'] == 2) | (df['date_ok'] == 1)

df['number is numeric'] = -1.0

# 2. Προετοιμασία των δεδομένων μόνο για το mask2
# Αφαιρούμε κόμματα και κενά για να γίνει σωστά η μετατροπή
clean_values = (
    df.loc[mask2, 'organismQuantity']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)

# 3. Μετατροπή σε αριθμητικό τύπο (float)
# Όπου δεν είναι αριθμός (π.χ. '5x10^5'), θα επιστρέψει NaN
numeric_conversion = pd.to_numeric(clean_values, errors='coerce')

# 4. Ενημέρωση της στήλης μόνο για τις τιμές που μετατράπηκαν επιτυχώς
# Οι υπόλοιπες τιμές στο mask2 (και εκτός αυτού) θα παραμείνουν -1
df.loc[mask2, 'number is numeric'] = numeric_conversion.fillna(-1.0)

# 4. Αποθήκευση του αποτελέσματος σε νέο αρχείο Excel
df.to_csv(r'C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD\OBIS datasets\datasets_1_3_4_6_v4.csv', index=False)