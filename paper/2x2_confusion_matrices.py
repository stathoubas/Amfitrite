# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:41:48 2026

@author: K. Pikounis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_ieee_confusion_matrix(values, set_name, output_filename="confusion_matrix.tiff"):
    """
    Generates a 300 DPI 2x2 confusion matrix suitable for an IEEE paper.
    Uses a manual text loop to guarantee numbers are NEVER hidden.
    """
    # Enforce Times New Roman font globally
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # Configure Matplotlib to embed fonts for EPS/TIFF
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Reshape the 4 values into a 2x2 numpy array
    data = np.array(values).reshape(2, 2)
    labels = ["non-HAB", "HAB"]
    
    # Create figure: Width 1.6 inches, Height 1.4 inches.
    # Giving it a tiny bit more height prevents the axes from cutting off.
    fig, ax = plt.subplots(figsize=(1.6, 1.4)) 
    
    # Create the heatmap WITHOUT auto-annotations (annot=False)
    sns.heatmap(data, annot=False, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar=False, ax=ax)
    
    # ==========================================
    # THE FIX: Manually inject the numbers
    # ==========================================
    # Calculate threshold for text color (white on dark cells, black on light cells)
    threshold = data.max() / 2.0
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = "white" if val > threshold else "black"
            
            # Force the text into the exact center of each cell at exactly 8pt
            ax.text(j + 0.5, i + 0.5, str(val),
                    ha="center", va="center",
                    color=text_color, fontsize=8, family="Times New Roman")
    # ==========================================
            
    # Set titles and labels strictly to 8pt
    ax.set_title(f"Confusion Matrix ({set_name})", fontsize=8, pad=6)
    ax.set_ylabel("Actual", fontsize=8, labelpad=2)
    ax.set_xlabel("Predicted", fontsize=8, labelpad=2)
    
    # Force the tick labels to stay flat and strictly at 8pt
    plt.xticks(rotation=0)
    plt.yticks(rotation=90, va='center')
    ax.tick_params(axis='both', which='major', labelsize=8, length=2, pad=2)
    
    # Tight layout minimizes the internal spacing
    plt.tight_layout(pad=0.1)
    
    # Save the figure, cropping out all excess white margins
    plt.savefig(output_filename, 
                format=output_filename.split('.')[-1], 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0.01)
    
    plt.close()
    print(f"Successfully saved {output_filename} at 300 DPI.")


# ==========================================
# EXAMPLE USAGE:
# ==========================================
if __name__ == "__main__":
    # The exact values (4 elements for a 2x2 matrix: TN, FP, FN, TP)
    matrix_values = [377, 63, 
                     68, 642]
    
    # Generate TIFF
    generate_ieee_confusion_matrix(matrix_values, "Validation", r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\paper\figures\cm_2_classes_universal_model_validation_fig4_left.tiff")
    
    matrix_values = [391, 49, 
                     74, 636]
    generate_ieee_confusion_matrix(matrix_values, "Test", r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\paper\figures\cm_2_classes_universal_model_test_fig4_right.tiff")