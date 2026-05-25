# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:56:10 2026


@author: K. Pikounis
"""
import os
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
# 2. INFERENCE ENGINE 
# ==============================================================================

def get_probabilities(model, loader, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            prob_hab = probs[:, 1].cpu().numpy()
            
            all_probs.extend(prob_hab)
            all_labels.extend(labels.numpy())
            
    return np.array(all_probs), np.array(all_labels)

def extract_eval_data(pth_path, csv_path, architecture, num_bands):
    """Extracts data only for Validation and Test to save compute time."""
    print("Initializing hardware...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {architecture.upper()} ({num_bands}-band) from {pth_path}...")
    model = load_resnet_model(pth_path, architecture, num_bands, device)
    
    print(f"Loading dataset split from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    data_dict = {}
    # Removed 'training' to save processing time since we only plot validation and test
    for split in ['validation', 'test']:
        print(f" -> Running inference on {split} split...")
        ds = InferenceDataset(df, split_name=split, num_bands=num_bands)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
        probs, labels = get_probabilities(model, loader, device)
        data_dict[split] = {'probs': probs, 'labels': labels}
        
    return data_dict

# ==============================================================================
# 3. PLOTTING FUNCTION (IEEE COMPLIANT)
# ==============================================================================

def plot_ieee_class_separation(data_dict, split_name, output_path):
    """Generates a highly compact, IEEE-compliant TIFF class separation plot."""
    
    # Enforce IEEE Font Standards
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Figure Size: Width 2.0 inches, Height 1.25 inches
    fig, ax = plt.subplots(figsize=(2.0, 1.25))
    
    p0 = data_dict[split_name]['probs'][data_dict[split_name]['labels'] == 0]
    p1 = data_dict[split_name]['probs'][data_dict[split_name]['labels'] == 1]
    
    bins = np.linspace(0, 1, 50)
    
    # Plot histograms
    ax.hist(p0, bins=bins, density=True, histtype='stepfilled', color='#1f77b4', alpha=0.7, label='Non-HAB')
    ax.hist(p1, bins=bins, density=True, histtype='stepfilled', color='#d62728', alpha=0.7, label='HAB')
    
    # Log scale is mandatory for these density plots to see the overlap
    ax.set_yscale('log')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.05, 20)
    
    # IEEE Labels using full words, strictly 8pt
    ax.set_xlabel("Prediction Score", fontsize=8, labelpad=1)
    ax.set_ylabel("Density", fontsize=8, labelpad=1)
    
    # Tick formatting
    ax.tick_params(axis='both', which='major', labelsize=8, length=2, pad=1)
    
    # Decision Boundary Line
    ax.axvline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.8)
    
    # Faint grid
    ax.grid(True, which='both', linestyle=':', alpha=0.4, linewidth=0.5)

    # Legend configured to fit the small space without blocking data
    ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, 
              frameon=False, columnspacing=0.5, handletextpad=0.2)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, format='tiff', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[Success] Saved {output_path}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    csv_path_name = "/home/kostas/AMFITRITE/IW_and_OW_CNN/amfitrite_universal_split.csv"

    #pth_path_name = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18_focal_loss_gamma_2/resnet18_bigearth_focal_loss.pth"
    #plot_path_name_prefix = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18_focal_loss_gamma_2/for_paper/class_separation"
    
    pth_path_name = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/amfitrite_resnet18_bigearth_best_ep_35.pth"
    plot_path_name_prefix = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/for_paper/bigearthnet_resnet18_universal"
    
    
    architecture = "resnet18"
    bands = 10
    
    print("\n--- Phase 1: Extracting Neural Network Probabilities ---")
    data_dict = extract_eval_data(pth_path_name, csv_path_name, architecture, bands)
    
    print("\n--- Phase 2: Generating IEEE Compliant TIFF Plots ---")
    
    # Generate Validation Plot
    val_out = plot_path_name_prefix + "_validation.tiff"
    plot_ieee_class_separation(data_dict, 'validation', val_out)
    
    # Generate Test Plot
    test_out = plot_path_name_prefix + "_test.tiff"
    plot_ieee_class_separation(data_dict, 'test', test_out)
    
    print("\nAll done!")