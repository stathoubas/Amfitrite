# -*- coding: utf-8 -*-
"""
Standalone Plotting Engine for AMFITRITE 
Generates paired scatter plots and relative improvement histograms 
from an existing final_duel_results.csv file.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# PLOTTING ENGINE (MODULE 6)
# ==============================================================================

def generate_comparison_plots(csv_path, base_output_dir):
    """
    Generates scatter plots and relative difference histograms for all 3 model combinations.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        print("Please ensure the path is correct and the file exists.")
        return

    df = pd.read_csv(csv_path)
    
    # Base list of metrics to plot
    metrics = [
        ('f1', 'F1 Score (Macro)'),
        ('acc', 'Overall Accuracy'),
        ('bal_acc', 'Balanced Accuracy'),
        ('hab_acc', 'Global HAB Accuracy'),
        ('nonhab_acc', 'Global non-HAB Accuracy'),
        ('iw_hab_acc', 'IW HABs Accuracy'),
        ('iw_nonhab_acc', 'IW non-HABs Accuracy'),
        ('land_acc', 'Land Accuracy'),
        ('clouds_acc', 'Clouds Accuracy')
    ]
    
    # Define the 3 duels
    scenarios = [
        {
            'col_x': 'baseline_', 'col_y': 'zeroshot_', 
            'name_x': 'Baseline', 'name_y': 'Zero-Shot', 
            'folder': 'duel_1_baseline_vs_zeroshot'
        },
        {
            'col_x': 'baseline_', 'col_y': 'ftfound_', 
            'name_x': 'Baseline', 'name_y': 'Fine-Tuned', 
            'folder': 'duel_2_baseline_vs_ftfound'
        },
        {
            'col_x': 'zeroshot_', 'col_y': 'ftfound_', 
            'name_x': 'Zero-Shot', 'name_y': 'Fine-Tuned', 
            'folder': 'duel_3_zeroshot_vs_ftfound'
        }
    ]
    
    for sc in scenarios:
        duel_dir = os.path.join(base_output_dir, sc['folder'])
        os.makedirs(duel_dir, exist_ok=True)
        print(f"\n[*] Generating plots for: {sc['name_x']} vs {sc['name_y']}")
        
        for m_suffix, title in metrics:
            m_x = sc['col_x'] + m_suffix
            m_y = sc['col_y'] + m_suffix
            
            # Ensure both columns exist and drop NaNs
            if m_x not in df.columns or m_y not in df.columns:
                continue
                
            valid_df = df.dropna(subset=[m_x, m_y])
            if len(valid_df) == 0: 
                continue
                
            # ---------------------------------------------------------
            # 1. SCATTER PLOT
            # ---------------------------------------------------------
            plt.figure(figsize=(8, 8))
            sns.scatterplot(x=valid_df[m_x], y=valid_df[m_y], s=150, color='blue', edgecolor='black', alpha=0.8)
            
            min_val = min(valid_df[m_x].min(), valid_df[m_y].min()) - 0.02
            max_val = max(valid_df[m_x].max(), valid_df[m_y].max()) + 0.02
            
            plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label="Parity (y = x)")
            
            plt.fill_between([min_val, max_val], [min_val, max_val], max_val, color='green', alpha=0.05, label=f"{sc['name_y']} Wins")
            plt.fill_between([min_val, max_val], min_val, [min_val, max_val], color='red', alpha=0.05, label=f"{sc['name_x']} Wins")
            
            plt.title(f"Monte Carlo Duel: {title}", fontsize=16, fontweight='bold')
            plt.xlabel(f"{sc['name_x']} Score", fontsize=14)
            plt.ylabel(f"{sc['name_y']} Score", fontsize=14)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend(loc='upper left', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(duel_dir, f"scatter_{m_suffix}.png"), dpi=300)
            plt.close()

            # ---------------------------------------------------------
            # 2. BINNED HISTOGRAM (Percentage Difference)
            # ---------------------------------------------------------
            # Calculation: 100 * (Y - X) / X
            pct_diff = 100 * (valid_df[m_y] - valid_df[m_x]) / valid_df[m_x]
            
            plt.figure(figsize=(8, 6))
            ax = sns.histplot(pct_diff, bins=10, kde=True, color='gray', edgecolor='black')
            
            # Add a vertical line at 0% (No improvement)
            plt.axvline(0, color='black', linestyle='--', linewidth=2, label="0% Difference")
            
            plt.title(f"Relative Improvement: {sc['name_y']} vs {sc['name_x']}\nMetric: {title}", fontsize=14, fontweight='bold')
            plt.xlabel(f"% Improvement over {sc['name_x']}", fontsize=12)
            plt.ylabel("Count (Monte Carlo Iterations)", fontsize=12)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(duel_dir, f"hist_{m_suffix}.png"), dpi=300)
            plt.close()
            
        print(f" -> Saved to {duel_dir}/")

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Define your paths here
    BASE_OUTPUT_DIR = "/home/kostas/AMFITRITE/IW_and_OW_CNN/IW_after_OW_and_IW_comparisons"
    FINAL_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, "final_duel_results.csv")
    
    print("\n" + "="*60)
    print("STARTING PLOT GENERATION")
    print("="*60)
    
    generate_comparison_plots(FINAL_CSV_PATH, BASE_OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*60)