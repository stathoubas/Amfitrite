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
import seaborn as sns

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
        
    # Adjust Stem
    model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Adjust Head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ==============================================================================
# 2. INFERENCE & PLOTTING ENGINE
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
    
    splits = ['training', 'validation', 'test']
    titles = ['Training Set', 'Validation Set', 'Test Set']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f"HAB Prediction Confidence Distribution ({architecture.upper()} | {num_bands} Bands)", fontsize=16, fontweight='bold')
    
    for i, (split, title) in enumerate(zip(splits, titles)):
        print(f" -> Processing {split} split...")
        ds = InferenceDataset(df, split_name=split, num_bands=num_bands)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
        
        probs, labels = get_probabilities(model, loader, device)
        
        ax = axes[i]
        sns.histplot(probs[labels == 0], bins=50, color='red', alpha=0.6, label='Non-HAB', ax=ax, stat='density', edgecolor=None)
        sns.histplot(probs[labels == 1], bins=50, color='blue', alpha=0.6, label='HAB', ax=ax, stat='density', edgecolor=None)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted Probability of being a HAB", fontsize=12)
        if i == 0: ax.set_ylabel("Density", fontsize=12)
        else: ax.set_ylabel("")
            
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(loc='upper center')
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_plot_path, dpi=300)
    print(f"\n[Success] Plot saved to {output_plot_path}")


if __name__ == "__main__":
    
    pth_path_name = "/home/kostas/AMFITRITE/IW_and_OW_CNN/amfitrite_resnet18_bigearth_best.pth"
    csv_path_name = "/home/kostas/AMFITRITE/IW_and_OW_CNN/amfitrite_universal_split.csv"
    plot_path_name = "/home/kostas/AMFITRITE/IW_and_OW_CNN/class_separation.png"
    architecture = "resnet18"
    bands = 10
    
    create_probability_distribution_plot(
        pth_path=pth_path_name,
        csv_path=csv_path_name,
        output_plot_path=plot_path_name,
        architecture=architecture,
        num_bands=bands
    )