# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:50:19 2026

@author: K. Pikounis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_ieee_confusion_matrix(values, set_name, output_filename="confusion_matrix.tiff"):
    """
    Generates a 300 DPI confusion matrix suitable for an IEEE paper.
    Uses a manual text loop to guarantee numbers are NEVER hidden.
    """
    # Enforce Times New Roman font globally
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # Configure Matplotlib to embed fonts for EPS/TIFF
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Reshape the 9 values into a 3x3 numpy array
    data = np.array(values).reshape(3, 3)
    labels = ["Low", "Mod.", "High"]
    
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
    plt.yticks(rotation=0)
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
    # The exact values from your uploaded image
    matrix_values = [165, 42, 3, 
                     35, 117, 49, 
                     4, 58, 232]
    
    # Generate TIFF
    generate_ieee_confusion_matrix(matrix_values, "Validation", r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\paper\figures\cm_3_classes_validation_fig1_left.tiff")
    
    matrix_values = [163, 41, 6, 
                     19, 129, 53, 
                     6, 64, 224]
    generate_ieee_confusion_matrix(matrix_values, "Test", r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\paper\figures\cm_3_classes_test_fig1_right.tiff")