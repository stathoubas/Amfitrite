# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:29:27 2026

@author: K. Pikounis
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

import pandas as pd
import matplotlib.pyplot as plt

def generate_compact_convergence_plots(csv_path="metrics.csv", out_folder = ""):
    """
    Generates two highly compact 300 DPI TIFF plots (Loss and F1-Score) 
    compliant with IEEE 8pt Times New Roman standards.
    Uses points instead of lines, 1.25 inch height, and narrow width.
    """
    # 1. Load and prepare the data
    df = pd.read_csv(csv_path)
    
    # Grouping by epoch safely merges logged steps into a single point per epoch
    epoch_data = df.groupby('epoch').mean()
    
    epochs = epoch_data.index
    train_loss = epoch_data['train_loss']
    val_loss = epoch_data['val_loss']
    train_f1 = epoch_data['train_f1_macro']
    val_f1 = epoch_data['val_f1_macro']
    
    # 2. Enforce IEEE Font Standards
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Compact figure parameters: Width 1.8 inches, Height 1.25 inches
    fig_size = (1.8, 1.25)
    
    # ==========================================
    # PLOT 1: LOSS CONVERGENCE (POINTS)
    # ==========================================
    fig, ax = plt.subplots(figsize=fig_size)
    
    # markersize=2.5 prevents the 50 epochs from turning into an unreadable blob
    ax.plot(epochs, train_loss, label='Training', color='#1f77b4', marker='.', markersize=2.5, linestyle='', alpha=0.8)
    ax.plot(epochs, val_loss, label='Validation', color='#ff7f0e', marker='x', markersize=2.5, linestyle='', alpha=0.8)
    
    # Strict 8pt labels
    ax.set_xlabel('Epoch', fontsize=8, labelpad=1)
    ax.set_ylabel('Loss', fontsize=8, labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=8, length=2, pad=1)
    
    # Faint grid to help guide the eye when zooming in
    ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
    
    # Legend is frame-less and highly compact so it doesn't block data
    ax.legend(fontsize=7, frameon=False, loc='upper right', borderpad=0.1, labelspacing=0.1, handletextpad=0.1)
    
    plt.tight_layout(pad=0.1)
    
    loss_filename = "plot_unvresal_RESNet18_loss_convergence_left.tiff"
    plt.savefig(os.path.join(out_folder, loss_filename), format='tiff', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"Successfully saved {loss_filename}")

    # ==========================================
    # PLOT 2: F1-SCORE CONVERGENCE (POINTS)
    # ==========================================
    fig, ax = plt.subplots(figsize=fig_size)
    
    ax.plot(epochs, train_f1, label='Training', color='#2ca02c', marker='.', markersize=2.5, linestyle='', alpha=0.8)
    ax.plot(epochs, val_f1, label='Validation', color='#d62728', marker='x', markersize=2.5, linestyle='', alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=8, labelpad=1)
    ax.set_ylabel('F1-Score', fontsize=8, labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=8, length=2, pad=1)
    
    ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
    
    # Placed in the lower right where F1 scores typically leave empty space
    ax.legend(fontsize=7, frameon=False, loc='lower right', borderpad=0.1, labelspacing=0.1, handletextpad=0.1)
    
    plt.tight_layout(pad=0.1)
    
    f1_filename = "plot_RESNet18_f1_convergence_right.tiff"
    plt.savefig(os.path.join(out_folder, f1_filename), format='tiff', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"Successfully saved {f1_filename}")


# Run the function
if __name__ == "__main__":
    generate_compact_convergence_plots(r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\OWD_CNN\IW_and_OW_CNN\bigearthnet_resnet18\metrics.csv", r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\paper\figures")