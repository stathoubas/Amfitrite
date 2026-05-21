# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:01:32 2026


Operational Sensitivity Analysis and Threshold Calibration
Target: Generating Recall vs FPR metrics for real-world HAB anomaly detection.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

import torch
import torch.nn as nn
from torchvision import models
import rasterio
import torchvision.transforms.functional as TF

# ==============================================================================
# DATASET & MODEL DEFINITIONS (Required to load your saved files)
# ==============================================================================

def load_existing_split_and_get_weights(csv_path):
    """
    Loads an already partitioned dataset and calculates the class weights 
    strictly based on the 'training' split.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find the split dataset at {csv_path}")
        
    print(f"\n[*] Loading existing split from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract only the training data to compute weights (prevents data leakage)
    train_df = df[df['split'] == 'training']
    
    count_nonhab = len(train_df[train_df['binary_label'] == 0])
    count_hab = len(train_df[train_df['binary_label'] == 1])
    total = count_nonhab + count_hab
    
    # Inverse Frequency Weighting Formulation
    weight_nonhab = total / (2.0 * count_nonhab) if count_nonhab > 0 else 1.0
    weight_hab = total / (2.0 * count_hab) if count_hab > 0 else 1.0
    
    # Create tensor for PyTorch CrossEntropyLoss
    class_weights = torch.tensor([weight_nonhab, weight_hab], dtype=torch.float32)
    
    print("\n--- Training Set Imbalance Calculated ---")
    print(f"Non-HAB Samples (Class 0): {count_nonhab} -> Weight: {weight_nonhab:.3f}")
    print(f"HAB Samples (Class 1):     {count_hab} -> Weight: {weight_hab:.3f}")
    
    return df, class_weights

class UniversalWaterDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, mode='validation', num_bands=10):
        self.df = dataframe[dataframe['split'] == mode].reset_index(drop=True)
        self.num_bands = num_bands
        self.active_bands = ["B02_raw.tif", "B03_raw.tif", "B04_raw.tif", "B05_raw.tif", "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", "B8A_raw.tif", "B11_raw.tif", "B12_raw.tif"]

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
    """Builds the raw architecture to load weights into."""
    if architecture == 'resnet18': model = models.resnet18(weights=None)
    elif architecture == 'resnet34': model = models.resnet34(weights=None)
    
    model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# ==============================================================================
# INFERENCE & ANALYSIS ENGINE
# ==============================================================================

def get_probabilities(model, dataloader, device):
    """Runs inference and extracts the Softmax probability of the HAB class."""
    model.eval()
    all_probs = []
    all_labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            
            # Apply Softmax to convert logits to probabilities [0.0 to 1.0]
            probs = torch.softmax(logits, dim=1)
            
            # Extract probability of Class 1 (HAB)
            hab_probs = probs[:, 1].cpu().numpy()
            
            all_probs.extend(hab_probs)
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_probs)

def generate_sensitivity_report(true_labels, hab_probs, split_name, output_dir):
    """
    Sweeps through thresholds, calculates prevalence-invariant metrics, 
    saves to CSV, and generates operational plots.
    """
    print(f"\n[*] Generating Operational Analysis for: {split_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sweep Thresholds from 0.00 to 1.00 in steps of 0.01
    thresholds = np.arange(0.0, 1.01, 0.01)
    results = []
    
    for t in thresholds:
        preds = (hab_probs >= t).astype(int)
        
        # Ensure confusion matrix always returns 4 values even if a class is entirely missing
        tn, fp, fn, tp = confusion_matrix(true_labels, preds, labels=[0, 1]).ravel()
        
        # Prevalence-Invariant Metrics
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0      # False Alarm Rate
        
        # Standard Imbalance Metrics (For context)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # F2-Score (Weights Recall twice as high as Precision)
        f2 = (5 * precision * recall) / ((4 * precision) + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'Threshold': round(t, 2),
            'Recall_TPR': recall,
            'FPR': fpr,
            'F2_Score': f2,
            'Precision': precision,
            'True_Positives': tp,
            'False_Positives': fp,
            'True_Negatives': tn,
            'False_Negatives': fn
        })
        
    df_results = pd.DataFrame(results)
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{split_name}_sensitivity_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f" -> Saved Data Table: {csv_path}")

    # ========================================================
    # PLOT 1: Operational Trade-off (Threshold vs Recall & FPR)
    # ========================================================
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Threshold'], df_results['Recall_TPR'], label='Recall (Sensitivity)', color='green', linewidth=3)
    plt.plot(df_results['Threshold'], df_results['FPR'], label='False Positive Rate (FPR)', color='red', linewidth=3)
    
    plt.axvline(0.5, color='gray', linestyle='--', label='Standard 0.5 Threshold')
    
    plt.title(f"Operational Trade-off Curve ({split_name})", fontsize=16, fontweight='bold')
    plt.xlabel("CNN Decision Threshold", fontsize=14)
    plt.ylabel("Rate (0.0 to 1.0)", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12, loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{split_name}_Operational_Tradeoff.png"), dpi=300)
    plt.close()

    # ========================================================
    # PLOT 2: Standard ROC Curve
    # ========================================================
    fpr_roc, tpr_roc, _ = roc_curve(true_labels, hab_probs)
    roc_auc = auc(fpr_roc, tpr_roc)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f"Receiver Operating Characteristic - ROC ({split_name})", fontsize=16, fontweight='bold')
    plt.xlabel("False Positive Rate (FPR)", fontsize=14)
    plt.ylabel("True Positive Rate (Recall)", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc="lower right", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{split_name}_ROC_Curve.png"), dpi=300)
    plt.close()
    
    print(f" -> Saved Plots to: {output_dir}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # ------------------ DEFINE YOUR PATHS HERE ------------------
    # The path to the CSV from the specific Iteration you want to analyze
    SPLIT_CSV_PATH = "/home/kostas/AMFITRITE/IW_and_OW_CNN/amfitrite_universal_split.csv"
    
    # The path to your best model (.ckpt or .pth)
    #MODEL_PATH = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18_focal_loss_gamma_2/resnet18_bigearth_focal_loss.pth"
    # Where to save the plots and CSVs
    #OUTPUT_DIR = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18_focal_loss_gamma_2/Operational_Sensitivity_Analysis"
    
    # the standard network
    MODEL_PATH = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/amfitrite_resnet18_bigearth_best.pth"
    OUTPUT_DIR = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/Operational_Sensitivity_Analysis"
    
    ARCHITECTURE = 'resnet18'  # or 'resnet34'
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    # ------------------------------------------------------------
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {DEVICE}")
    
    # 1. Load Data Split
    df_master, _ = load_existing_split_and_get_weights(SPLIT_CSV_PATH)
    
    # 2. Create DataLoaders for Val and Test
    val_dataset = UniversalWaterDataset(df_master, mode='validation')
    test_dataset = UniversalWaterDataset(df_master, mode='test')
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 3. Load Model Weights
    model = build_water_resnet(ARCHITECTURE, num_bands=10)
    
    print(f"\n[*] Loading model weights from {MODEL_PATH}")
    if MODEL_PATH.endswith('.ckpt'):
        # If it's a PyTorch Lightning Checkpoint, extract the state dict
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint['state_dict']
        
        # FIX: Only extract keys belonging to the actual neural network 
        # (Ignore Lightning buffers like class_weights and criterion)
        clean_state_dict = {
            k.replace('model.', ''): v 
            for k, v in state_dict.items() 
            if k.startswith('model.')
        }
        model.load_state_dict(clean_state_dict, strict=True)
    else:
        # If it's a standard PyTorch .pth file
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        
    model.to(DEVICE)
    
    # 4. Generate Validation Report (USE THIS ONE TO PICK YOUR THRESHOLD)
    val_labels, val_probs = get_probabilities(model, val_loader, DEVICE)
    generate_sensitivity_report(val_labels, val_probs, "Validation", OUTPUT_DIR)
    
    # 5. Generate Test Report (ONLY REPORT THE ROW CORRESPONDING TO YOUR CHOSEN THRESHOLD)
    test_labels, test_probs = get_probabilities(model, test_loader, DEVICE)
    generate_sensitivity_report(test_labels, test_probs, "Test", OUTPUT_DIR)
    
    print("\n[✔] Operational Sensitivity Analysis Complete!")