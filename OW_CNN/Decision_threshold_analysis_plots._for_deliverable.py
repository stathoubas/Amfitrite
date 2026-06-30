# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:09:33 2026

@author: K. Pikounis
"""

import pandas as pd
import matplotlib.pyplot as plt

def generate_presentation_tradeoff_plot(csv_path, set_name, output_filename, opt_thresh):
    """
    Generates a high-resolution, presentation-ready tradeoff plot.
    Uses a pre-calculated optimal threshold (derived from the Validation set).
    """
    # 2. Load the data
    df = pd.read_csv(csv_path)
    
    # 3. Figure Size: Generous layout for presentation slides
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 4. Plot the curves (Thicker lines for projector/screen visibility)
    ax.plot(df['Threshold'], df['F2_Score'], label='F2-Score', color='#2ca02c', linestyle='-', linewidth=3.5)
    ax.plot(df['Threshold'], df['Recall_TPR'], label='Recall', color='#1f77b4', linestyle='--', linewidth=3.5)
    
    # "FPR" is expanded to "False Positive Rate"
    ax.plot(df['Threshold'], df['FPR'], label='False Positive Rate', color='#d62728', linestyle=':', linewidth=3.5)
    
    # Vertical line using the globally optimal threshold calculated from Validation
    ax.axvline(x=opt_thresh, color='black', linestyle='-.', linewidth=2.0, alpha=0.8, 
               label=f'Optimal Threshold ({opt_thresh:.2f})')
    
    # 5. Set labels with bold, large fonts
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Decision Threshold Analysis ({set_name} Set)')
    
    # Tick formatting
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 1.05) # Adds headroom so the top lines aren't cut off
    ax.tick_params(axis='both', which='major')
    
    # Stronger grid for readability on screens
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.0)
    
    # 6. Legend - Moved to the right to avoid overlapping the central data lines
    ax.legend(fontsize=14, loc='center right', frameon=True, borderpad=0.8, labelspacing=0.6, shadow=True)
    
    plt.tight_layout()
    
    # Save the figure as PNG (Best format for presentation software)
    plt.savefig(output_filename, format=output_filename.split('.')[-1], 
                dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"[Success] Saved presentation plot: {output_filename}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # Define your file paths here
    val_csv = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD_CNN\IW_and_OW_CNN\bigearthnet_resnet18\Operational_Sensitivity_Analysis_ep_35\Validation_sensitivity_metrics.csv"
    test_csv = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD_CNN\IW_and_OW_CNN\bigearthnet_resnet18\Operational_Sensitivity_Analysis_ep_35\Test_sensitivity_metrics.csv"
    
    try:
        # STEP 1: Calculate the optimal threshold from the Validation set ONLY
        print("Calculating optimal threshold from Validation set...")
        val_df = pd.read_csv(val_csv)
        opt_idx = val_df['F2_Score'].idxmax()
        validation_optimal_threshold = val_df.loc[opt_idx, 'Threshold']
        print(f" -> Optimal Threshold found at: {validation_optimal_threshold:.2f}")
        
        # STEP 2: Generate BOTH plots using this single shared threshold (Saving as PNGs)
        generate_presentation_tradeoff_plot(
            csv_path=val_csv, 
            set_name="Validation", 
            output_filename=r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD_CNN\IW_and_OW_CNN\plots_for_deliverable\Validation_Operational_Tradeoff_Presentation.png", 
            opt_thresh=validation_optimal_threshold
        )
        
        generate_presentation_tradeoff_plot(
            csv_path=test_csv, 
            set_name="Test", 
            output_filename=r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD_CNN\IW_and_OW_CNN\plots_for_deliverable\Test_Operational_Tradeoff_Presentation.png", 
            opt_thresh=validation_optimal_threshold
        )
        
    except FileNotFoundError as e:
        print(f"Error: Could not find the CSV file. {e}")