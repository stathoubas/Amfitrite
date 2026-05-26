# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:40:19 2026

@author: K. Pikounis

Comprehensive Domain-Specific Evaluation Script
Generates Global, Inland Water (IW), and Open Water (OW) metrics, 
including specific Sensitivity (HAB) and Specificity (non-HAB) rates.
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torchvision import models
import rasterio

# ==============================================================================
# DATASET & MODEL DEFINITIONS
# ==============================================================================

class UniversalWaterDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, mode='test', num_bands=10):
        self.df = dataframe[dataframe['split'] == mode].reset_index(drop=True)
        self.num_bands = num_bands
        self.active_bands = [
            "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", "B05_raw.tif", 
            "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", "B8A_raw.tif", 
            "B11_raw.tif", "B12_raw.tif"
        ]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_path = row['folder_path']
        label = row['binary_label']
        source = row['source']  # 'IW' or 'OW'
        
        band_data = []
        for b_name in self.active_bands:
            with rasterio.open(os.path.join(folder_path, b_name)) as src:
                band_data.append(src.read(1).astype(np.float32))
        
        tensor = torch.from_numpy(np.stack(band_data, axis=0)) / 10000.0
        return tensor, label, source

def build_water_resnet(architecture='resnet18', num_bands=10):
    if architecture == 'resnet18': model = models.resnet18(weights=None)
    elif architecture == 'resnet34': model = models.resnet34(weights=None)
    
    model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# ==============================================================================
# METRICS CALCULATION ENGINE
# ==============================================================================

def calculate_granular_metrics(y_true, y_pred, prefix=""):
    """Calculates F1, Acc, Bal Acc, Sensitivity (HAB), and Specificity (non-HAB)."""
    if len(y_true) == 0:
        return {f"{prefix}F1": np.nan, f"{prefix}Acc": np.nan, f"{prefix}Bal_Acc": np.nan, f"{prefix}HAB_Acc": np.nan, f"{prefix}nonHAB_Acc": np.nan}
        
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # Calculate Class-Specific Accuracies (Sensitivity & Specificity)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    hab_acc = tp / (tp + fn) if (tp + fn) > 0 else np.nan     # Sensitivity
    nonhab_acc = tn / (tn + fp) if (tn + fp) > 0 else np.nan  # Specificity
    
    return {
        f"{prefix}F1": round(f1, 4),
        f"{prefix}Acc": round(acc, 4),
        f"{prefix}Bal_Acc": round(bal_acc, 4),
        f"{prefix}HAB_Acc_(Sensitivity)": round(hab_acc, 4),
        f"{prefix}nonHAB_Acc_(Specificity)": round(nonhab_acc, 4)
    }

def evaluate_split(model, dataloader, device, split_name):
    """Runs inference and delegates metric calculations per domain."""
    model.eval()
    all_preds, all_labels, all_sources = [], [], []
    
    print(f"\n[*] Evaluating {split_name} Split...")
    with torch.no_grad():
        for images, labels, sources in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_sources.extend(sources)
            
    df_results = pd.DataFrame({'True': all_labels, 'Pred': all_preds, 'Source': all_sources})
    
    # 1. Global Metrics (IW + OW combined)
    metrics = {'Split': split_name}
    metrics.update(calculate_granular_metrics(df_results['True'], df_results['Pred'], prefix="Global_"))
    
    # 2. Inland Water (IW) Metrics
    iw_df = df_results[df_results['Source'] == 'IW']
    metrics.update(calculate_granular_metrics(iw_df['True'], iw_df['Pred'], prefix="IW_"))
    
    # 3. Open Water (OW) Metrics
    ow_df = df_results[df_results['Source'] == 'OW']
    metrics.update(calculate_granular_metrics(ow_df['True'], ow_df['Pred'], prefix="OW_"))
    
    return metrics

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # ------------------ DEFINE YOUR PATHS HERE ------------------
    SPLIT_CSV_PATH = "/home/kostas/AMFITRITE/IW_and_OW_CNN/amfitrite_universal_split.csv"
    
    # Path to your Universal Foundation Checkpoint (Epoch 35!)
    MODEL_PATH = "//home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/amfitrite_resnet18_bigearth_best_ep_35.pth"
    
    OUTPUT_DIR = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/Final_Standard_Evaluation"
    # ------------------------------------------------------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware initialized on: {DEVICE}")
    
    # 1. Load Data
    df_master = pd.read_csv(SPLIT_CSV_PATH)
    
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    train_loader = torch.utils.data.DataLoader(UniversalWaterDataset(df_master, 'training'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader   = torch.utils.data.DataLoader(UniversalWaterDataset(df_master, 'validation'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = torch.utils.data.DataLoader(UniversalWaterDataset(df_master, 'test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 2. Load Model Safely (ignoring Lightning buffers)
    model = build_water_resnet('resnet18', num_bands=10)
    print(f"\n[*] Loading weights from {MODEL_PATH}")
    
    if MODEL_PATH.endswith('.ckpt'):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint['state_dict']
        clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        model.load_state_dict(clean_state_dict, strict=True)
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        
    model.to(DEVICE)
    
    # 3. Evaluate All Splits
    final_results = []
    final_results.append(evaluate_split(model, train_loader, DEVICE, "Training"))
    final_results.append(evaluate_split(model, val_loader, DEVICE, "Validation"))
    final_results.append(evaluate_split(model, test_loader, DEVICE, "Test"))
    
    # 4. Save to CSV
    summary_df = pd.DataFrame(final_results)
    
    # Reorder columns for readability in Excel
    cols = ['Split', 
            'Global_F1', 'Global_Acc', 'Global_Bal_Acc', 'Global_HAB_Acc_(Sensitivity)', 'Global_nonHAB_Acc_(Specificity)',
            'IW_F1', 'IW_Acc', 'IW_Bal_Acc', 'IW_HAB_Acc_(Sensitivity)', 'IW_nonHAB_Acc_(Specificity)',
            'OW_F1', 'OW_Acc', 'OW_Bal_Acc', 'OW_HAB_Acc_(Sensitivity)', 'OW_nonHAB_Acc_(Specificity)']
    summary_df = summary_df[cols]
    
    csv_out = os.path.join(OUTPUT_DIR, "Comprehensive_Universal_Evaluation.csv")
    summary_df.to_csv(csv_out, index=False)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print(f"Metrics successfully saved to: {csv_out}")
    print("="*70)
    
    # Print transposed for easy reading in terminal
    print("\n--- Summary Preview ---")
    print(summary_df.set_index('Split').T.to_string())