# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:06:18 2026

@author: K. Pikounis
"""

import pandas as pd

INPUT_FILE = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\dataset_summary_with_splits_from_Server_with_predictions.xlsx"
MODEL_COLS = ["pred_res18_scl", "pred_res18_no_scl", "pred_convnext_scl", "pred_rdnet_no_scl"]
GT_COL = "indicative_class"

def analyze_voting_scenarios():
    df = pd.read_excel(INPUT_FILE) if INPUT_FILE.endswith('.xlsx') else pd.read_csv(INPUT_FILE)
    
    # Map Truth to Binary
    def map_gt(val): return 'no' if str(val).lower() == 'low' else 'yes'
    df['gt_binary'] = df[GT_COL].apply(map_gt)
    
    # Focus on VALIDATION set (since it represents unseen data best)
    val_df = df[df['split'] == 'validation'].copy()
    
    print(f"--- Scenario Analysis (Validation Set: n={len(val_df)}) ---")
    
    # 1. Calculate Votes
    # We convert yes/no to 1/0 for easier math
    for col in MODEL_COLS:
        val_df[f'{col}_int'] = val_df[col].apply(lambda x: 1 if x == 'yes' else 0)
    
    # Sum of votes for "Yes"
    val_df['yes_votes'] = val_df[[f'{c}_int' for c in MODEL_COLS]].sum(axis=1)
    
    # 2. Define Scenarios
    scenarios = {
        "Unanimous (4-0)": 0,
        "Majority (3-1) - RDNet with Majority": 0,
        "Majority (3-1) - RDNet Alone (The 'Lone Wolf' case)": 0,
        "Tie (2-2) - RDNet says Yes": 0,
        "Tie (2-2) - RDNet says No": 0
    }
    
    correct_counts = {k: 0 for k in scenarios}
    
    for idx, row in val_df.iterrows():
        votes = row['yes_votes']
        rdnet_val = row['pred_rdnet_no_scl_int']
        truth = 1 if row['gt_binary'] == 'yes' else 0
        
        # Scenario Logic
        case = ""
        prediction = -1
        
        if votes == 4 or votes == 0:
            case = "Unanimous (4-0)"
            prediction = rdnet_val # Doesn't matter, all same
            
        elif votes == 3:
            if rdnet_val == 1: 
                case = "Majority (3-1) - RDNet with Majority"
                prediction = 1
            else: 
                case = "Majority (3-1) - RDNet Alone (The 'Lone Wolf' case)"
                prediction = 0 # RDNet is 0, Majority is 1. If we follow Majority, pred is 1.
                
        elif votes == 1:
            if rdnet_val == 0: 
                case = "Majority (3-1) - RDNet with Majority"
                prediction = 0
            else: 
                case = "Majority (3-1) - RDNet Alone (The 'Lone Wolf' case)"
                prediction = 1 # RDNet is 1, Majority is 0.
                
        elif votes == 2:
            if rdnet_val == 1: 
                case = "Tie (2-2) - RDNet says Yes"
                prediction = 1
            else: 
                case = "Tie (2-2) - RDNet says No"
                prediction = 0
                
        # Update Stats
        if case:
            scenarios[case] += 1
            if prediction == truth:
                correct_counts[case] += 1

    # 3. Print Results
    print(f"{'Scenario':<55} | {'Count':<5} | {'Accuracy of Logic':<10}")
    print("-" * 85)
    for k, v in scenarios.items():
        acc = (correct_counts[k] / v * 100) if v > 0 else 0
        print(f"{k:<55} | {v:<5} | {acc:.1f}%")

if __name__ == "__main__":
    analyze_voting_scenarios()