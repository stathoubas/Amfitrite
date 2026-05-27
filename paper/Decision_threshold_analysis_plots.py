# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

def generate_ieee_tradeoff_plot(csv_path, set_name, output_filename, opt_thresh):
    """
    Generates a 300 DPI TIFF tradeoff plot compliant with IEEE 8pt Times New Roman standards.
    Uses a pre-calculated optimal threshold (derived from the Validation set).
    """
    # 1. Enforce IEEE Font Standards globally
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # 2. Load the data
    df = pd.read_csv(csv_path)
    
    # 3. Figure Size: Standard single-column width (3.2 inches), proportional height (2.4 inches)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    
    # 4. Plot the curves
    ax.plot(df['Threshold'], df['F2_Score'], label='F2-Score', color='#2ca02c', linestyle='-', linewidth=1.5)
    ax.plot(df['Threshold'], df['Recall_TPR'], label='Recall', color='#1f77b4', linestyle='--', linewidth=1.5)
    
    # Note: "FPR" is expanded to "False Positive Rate" to strictly comply with IEEE rules
    ax.plot(df['Threshold'], df['FPR'], label='False Positive Rate', color='#d62728', linestyle=':', linewidth=1.5)
    
    # Vertical line using the globally optimal threshold calculated from Validation
    ax.axvline(x=opt_thresh, color='black', linestyle='-.', linewidth=1.2, alpha=0.8, 
               label=f'Optimal Threshold ({opt_thresh:.2f})')
    
    # 5. Set labels strictly at 8pt
    ax.set_xlabel('Decision Threshold', fontsize=8, labelpad=2)
    ax.set_ylabel('Metric Value', fontsize=8, labelpad=2)
    ax.set_title(f'Decision Threshold Analysis ({set_name} set)', fontsize=8, pad=6)
    
    # Tick formatting
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 1.05) # Adds headroom so the top lines aren't cut off
    ax.tick_params(axis='both', which='major', labelsize=8, length=3, pad=2)
    
    # Faint grid for readability
    ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
    
    # 6. Legend - Moved to the exact middle of the plot
    ax.legend(fontsize=7, loc='center', frameon=True, borderpad=0.4, labelspacing=0.2)
    
    plt.tight_layout(pad=0.1)
    
    # Save the figure, cropping out all excess white margins
    plt.savefig(output_filename, format=output_filename.split('.')[-1], 
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    
    plt.close()
    print(f"[Success] Saved {output_filename} at 300 DPI.")

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
        
        # STEP 2: Generate BOTH plots using this single shared threshold
        generate_ieee_tradeoff_plot(
            csv_path=val_csv, 
            set_name="Validation", 
            output_filename=r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\paper\figures\Validation_Operational_Tradeoff_IEEE.tiff", 
            opt_thresh=validation_optimal_threshold
        )
        
        generate_ieee_tradeoff_plot(
            csv_path=test_csv, 
            set_name="Test", 
            output_filename=r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\paper\figures\Test_Operational_Tradeoff_IEEE.tiff", 
            opt_thresh=validation_optimal_threshold
        )
        
    except FileNotFoundError as e:
        print(f"Error: Could not find the CSV file. {e}")