# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:17:11 2026

Standard Classification Evaluation Script
Calculates F1, Accuracy, Balanced Accuracy, and generates Confusion Matrices 
for Training, Validation, and Test splits.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report

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
        self.mode = mode
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
        
        band_data = []
        for b_name in self.active_bands:
            with rasterio.open(os.path.join(folder_path, b_name)) as src:
                band_data.append(src.read(1).astype(np.float32))
        
        tensor = torch.from_numpy(np.stack(band_data, axis=0)) / 10000.0
        return tensor, label

def build_water_resnet(architecture='resnet18', num_bands=10):
    if architecture == 'resnet18': model = models.resnet18(weights=None)
    elif architecture == 'resnet34': model = models.resnet34(weights=None)
    
    model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# ==============================================================================
# EVALUATION ENGINE
# ==============================================================================

def evaluate_and_plot(model, dataloader, device, split_name, output_dir):
    """
    Runs inference, calculates standard metrics, and plots the Confusion Matrix.
    Uses the standard argmax (0.5 probability) decision boundary.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\n[*] Evaluating {split_name} split...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            
            # Standard argmax classification
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    print(f" -> Accuracy: {acc:.4f} | Bal Acc: {bal_acc:.4f} | Macro F1: {f1_macro:.4f}")
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["non-HAB", "HAB"], 
                yticklabels=["non-HAB", "HAB"],
                annot_kws={"size": 14})
    
    plt.title(f"Confusion Matrix: {split_name}", fontsize=16, fontweight='bold')
    plt.xlabel("CNN Predicted Label", fontsize=14)
    plt.ylabel("Actual Ground Truth", fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"Confusion_Matrix_{split_name}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Save a detailed classification report text file
    report = classification_report(all_labels, all_preds, target_names=["non-HAB", "HAB"], digits=4)
    with open(os.path.join(output_dir, f"Classification_Report_{split_name}.txt"), "w") as f:
        f.write(report)
        
    return {
        'Split': split_name,
        'Accuracy': round(acc, 4),
        'Balanced_Accuracy': round(bal_acc, 4),
        'Macro_F1': round(f1_macro, 4)
    }

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # ------------------ DEFINE YOUR PATHS HERE ------------------
    # The path to your fully processed dataset split CSV
    SPLIT_CSV_PATH = "/home/kostas/AMFITRITE/IW_and_OW_CNN/amfitrite_universal_split.csv"
    
    # The path to your trained model weights (.ckpt or .pth)
    MODEL_PATH = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/amfitrite_resnet18_bigearth_best_ep_35.pth"
    
    # The directory where you want the CMs and metrics CSV saved
    OUTPUT_DIR = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/epoch_32_confusion_matrices"
    
    ARCHITECTURE = 'resnet18'
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    # ------------------------------------------------------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware initialized on: {DEVICE}")
    
    # 1. Load Data
    print(f"[*] Loading dataset split from {SPLIT_CSV_PATH}")
    df_master = pd.read_csv(SPLIT_CSV_PATH)
    
    # 2. Create DataLoaders
    # Note: Shuffle=False for ALL loaders here so confusion matrices match deterministic ordering
    train_loader = torch.utils.data.DataLoader(UniversalWaterDataset(df_master, 'training'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader   = torch.utils.data.DataLoader(UniversalWaterDataset(df_master, 'validation'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = torch.utils.data.DataLoader(UniversalWaterDataset(df_master, 'test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 3. Load Model and Weights Safely
    model = build_water_resnet(ARCHITECTURE, num_bands=10)
    print(f"\n[*] Loading model weights from {MODEL_PATH}")
    
    if MODEL_PATH.endswith('.ckpt'):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint['state_dict']
        # Safely strip Lightning modules and focal loss weights
        clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        model.load_state_dict(clean_state_dict, strict=True)
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        
    model.to(DEVICE)
    
    # 4. Evaluate all splits
    results = []
    results.append(evaluate_and_plot(model, train_loader, DEVICE, "Training", OUTPUT_DIR))
    results.append(evaluate_and_plot(model, val_loader, DEVICE, "Validation", OUTPUT_DIR))
    results.append(evaluate_and_plot(model, test_loader, DEVICE, "Test", OUTPUT_DIR))
    
    # 5. Save Summary DataFrame
    summary_df = pd.DataFrame(results)
    summary_csv_path = os.path.join(OUTPUT_DIR, "Final_Metrics_Summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print(f"Saved all Confusion Matrices and Summary to: {OUTPUT_DIR}")
    print("="*50)
    print(summary_df.to_string(index=False))