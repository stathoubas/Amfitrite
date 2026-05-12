# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:40:48 2026

@author: K. Pikounis
"""
import os
import argparse
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DYNAMIC DATASET & MODEL BUILDER
# ==============================================================================

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, split_name, num_bands):
        self.df = dataframe[dataframe['split'] == split_name].reset_index(drop=True)
        self.num_bands = num_bands
        
        self.band_names_12 = [
            "B01_raw.tif", "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", 
            "B05_raw.tif", "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", 
            "B8A_raw.tif", "B09_raw.tif", "B11_raw.tif", "B12_raw.tif"
        ]
        
        self.band_names_10 = [
            "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", "B05_raw.tif", 
            "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", "B8A_raw.tif", 
            "B11_raw.tif", "B12_raw.tif"
        ]
        
        self.active_bands = self.band_names_10 if self.num_bands == 10 else self.band_names_12

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_path = row['folder_path']
        label = row['binary_label']
        
        band_data = []
        for b_name in self.active_bands:
            with rasterio.open(os.path.join(folder_path, b_name)) as src:
                band_data.append(src.read(1).astype(np.float32))
                
        bands_stack = np.stack(band_data, axis=0)
        tensor = torch.from_numpy(bands_stack) / 10000.0
        return tensor, label

def load_resnet_model(pth_path, architecture, num_bands, device):
    """Dynamically builds ResNet18 or ResNet34 and loads the weights."""
    if architecture == 'resnet18':
        model = models.resnet18(weights=None)
    elif architecture == 'resnet34':
        model = models.resnet34(weights=None)
    else:
        raise ValueError("Architecture must be 'resnet18' or 'resnet34'")
        
    model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ==============================================================================
# 2. INFERENCE & OVERLAY PLOTTING ENGINE
# ==============================================================================

def get_probabilities(model, loader, device):
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            prob_hab = probs[:, 1].cpu().numpy()
            
            all_probs.extend(prob_hab)
            all_labels.extend(labels.numpy())
            
    return np.array(all_probs), np.array(all_labels)

def create_probability_distribution_plot(pth_path, csv_path, output_plot_path, architecture, num_bands):
    print("Initializing hardware...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {architecture.upper()} ({num_bands}-band) from {pth_path}...")
    model = load_resnet_model(pth_path, architecture, num_bands, device)
    
    print(f"Loading dataset split from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Plotting Configuration
    splits = ['training', 'validation', 'test']
    
    # Visual Styles: Training is a filled background, Val/Test are bold outlines
    styles = {
        'training':   {'c0': 'orange', 'c1': 'green', 'ls': '-',  'lw': 1.5, 'fill': True,  'alpha': 0.25},
        'validation': {'c0': 'maroon', 'c1': 'cyan',  'ls': '-',  'lw': 2.5, 'fill': False, 'alpha': 0.9},
        'test':       {'c0': 'red',    'c1': 'blue',  'ls': '--', 'lw': 2.5, 'fill': False, 'alpha': 0.9}
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f"HAB Prediction Confidence Distribution ({architecture.upper()} | {num_bands} Bands)", fontsize=16, fontweight='bold')
    
    bins = np.linspace(0, 1, 50)
    
    # 2. Data Processing & Plotting
    for split in splits:
        print(f" -> Processing {split} split...")
        ds = InferenceDataset(df, split_name=split, num_bands=num_bands)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
        
        probs, labels = get_probabilities(model, loader, device)
        
        p0 = probs[labels == 0]
        p1 = probs[labels == 1]
        
        s = styles[split]
        htype = 'stepfilled' if s['fill'] else 'step'
        
        # Plot Non-HAB (Class 0)
        ax.hist(p0, bins=bins, density=True, histtype=htype, 
                color=s['c0'], linestyle=s['ls'], linewidth=s['lw'], alpha=s['alpha'], 
                label=f"{split.capitalize()} (Non-HAB Actual)")
        
        # Plot HAB (Class 1)
        ax.hist(p1, bins=bins, density=True, histtype=htype, 
                color=s['c1'], linestyle=s['ls'], linewidth=s['lw'], alpha=s['alpha'], 
                label=f"{split.capitalize()} (HAB Actual)")

    # 3. Formatting the Output
    ax.set_yscale('log') # Logarithmic Y-Axis
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Predicted Probability of being a HAB", fontsize=12)
    ax.set_ylabel("Density (Log Scale)", fontsize=12)
    
    # The Decision Boundary
    ax.axvline(0.5, color='black', linestyle=':', linewidth=2, alpha=0.8, label="Decision Boundary (0.5)")
    
    # Clean up the legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Clean Binned Histogram saved to {output_plot_path}")

if __name__ == "__main__":
    
    pth_path_name = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/amfitrite_resnet18_bigearth_best.pth"
    csv_path_name = "/home/kostas/AMFITRITE/IW_and_OW_CNN/amfitrite_universal_split.csv"
    plot_path_name = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/class_separation.png"
    architecture = "resnet18"
    bands = 10
    
    create_probability_distribution_plot(
        pth_path=pth_path_name,
        csv_path=csv_path_name,
        output_plot_path=plot_path_name,
        architecture=architecture,
        num_bands=bands
    )