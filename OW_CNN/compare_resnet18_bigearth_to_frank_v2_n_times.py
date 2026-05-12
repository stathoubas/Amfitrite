# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:03:22 2026

@author: K. Pikounis

train CNNs using both OW and IW datasest

run n times:
    split
    train resnet18 with bigerath weight -> get best model -> get statics
    train rsnet 34 with imagenet weights -> get best model
    intialize frankenstein model from teh weights of the best two above -> train model -> get best model -> get statics
    put metrcis per run in a csv
create comparison plots of the metrics using the csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import random
import rasterio
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, balanced_accuracy_score

import re
import gc

# ==============================================================================
# MODULE 1: CONFIGURATION & DATA HARMONIZER
# ==============================================================================

class DatasetHarmonizer:
    def __init__(self, iw_dir, ow_dir, iw_excel, ow_csv):
        self.iw_dir = iw_dir
        self.ow_dir = ow_dir
        self.iw_excel = iw_excel
        self.ow_csv = ow_csv

    def process_iw_data(self):
        """Loads and standardizes the Inland Water dataset."""
        print("Loading Inland Water (IW) Dataset...")
        df_iw = pd.read_excel(self.iw_excel)
        
        standardized_rows = []
        for _, row in df_iw.iterrows():
            uid = str(row['uid'])
            folder_path = os.path.join(self.iw_dir, uid)
            
            # Map labels
            indicative_class = str(row['new_indicative_class']).strip().lower()
            if indicative_class in ['high', 'moderate']:
                binary_label = 1
                strat_group = 'iw_hab'
            else:
                binary_label = 0
                strat_group = 'iw_nonhab'
                
            # Verify physical folder exists
            if os.path.isdir(folder_path):
                standardized_rows.append({
                    'uid': uid,
                    'folder_path': folder_path,
                    'binary_label': binary_label,
                    'strat_group': strat_group,
                    'source': 'IW'
                })
                
        print(f" -> Found {len(standardized_rows)} valid IW folders.")
        return pd.DataFrame(standardized_rows)

    def process_ow_data(self):
        """Loads and standardizes the Open Water dataset."""
        print("Loading Open Water (OW) Dataset...")
        df_ow = pd.read_csv(self.ow_csv)
        
        standardized_rows = []
        for _, row in df_ow.iterrows():
            uid = str(row['ID'])
            
            # Determine logic for Open Water suffix and strat group
            if row['tile_is_hab'] == True:
                suffix = "HAB"
                binary_label = 1
                strat_group = "ow_hab"
            else:
                binary_label = 0
                if row['tile_status'] == 'land':
                    suffix = "land"
                    strat_group = "land"
                elif row['tile_status'] == 'clouds':
                    suffix = "clouds"
                    strat_group = "clouds"
                else:
                    suffix = "nonHAB"
                    strat_group = "ow_nonhab"
                    
            folder_path = os.path.join(self.ow_dir, f"{uid}_{suffix}")
            
            # Verify physical folder exists
            if os.path.isdir(folder_path):
                standardized_rows.append({
                    'uid': uid,
                    'folder_path': folder_path,
                    'binary_label': binary_label,
                    'strat_group': strat_group,
                    'source': 'OW'
                })
                
        print(f" -> Found {len(standardized_rows)} valid OW folders.")
        return pd.DataFrame(standardized_rows)

    def create_master_registry(self):
        """Merges both datasets into a single, clean Pandas DataFrame."""
        df_iw = self.process_iw_data()
        df_ow = self.process_ow_data()
        
        master_df = pd.concat([df_iw, df_ow], ignore_index=True)
        print(f"\n[Success] Master Registry created with {len(master_df)} total images.")
        print("Distribution by Stratification Group:")
        print(master_df['strat_group'].value_counts())
        
        return master_df


# ==============================================================================
# MODULE 2: SPLITTER & IMBALANCE CALCULATOR
# ==============================================================================

def create_stratified_split_and_weights(master_df, output_registry_path, random_seed=42):
    """
    Splits the master dataframe 70/15/15 based on strat_group.
    Calculates PyTorch class weights purely from the Training set.
    """
    if os.path.exists(output_registry_path):
        print(f"\nLoading existing split from {output_registry_path}...")
        df = pd.read_csv(output_registry_path)
    else:
        print("\nPerforming 70/15/15 Stratified Split...")
        df = master_df.copy()
        df['split'] = 'junk'
        
        # 1. First split: 70% Train, 30% Temp
        train_idx, temp_idx = train_test_split(
            df.index, test_size=0.30, stratify=df['strat_group'], random_state=random_seed
        )
        # 2. Second split: 15% Val, 15% Test (from the 30% Temp)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, stratify=df.loc[temp_idx, 'strat_group'], random_state=random_seed
        )
        
        df.loc[train_idx, 'split'] = 'training'
        df.loc[val_idx, 'split'] = 'validation'
        df.loc[test_idx, 'split'] = 'test'
        
        df.to_csv(output_registry_path, index=False)
        print(f"Split saved to {output_registry_path}")

    # --- CALCULATE CLASS WEIGHTS ---
    # We strictly use the Training set to calculate weights to prevent data leakage!
    train_df = df[df['split'] == 'training']
    
    count_nonhab = len(train_df[train_df['binary_label'] == 0])
    count_hab = len(train_df[train_df['binary_label'] == 1])
    total = count_nonhab + count_hab
    
    # Inverse Frequency Weighting Formulation
    weight_nonhab = total / (2.0 * count_nonhab)
    weight_hab = total / (2.0 * count_hab)
    
    # Create tensor for PyTorch CrossEntropyLoss
    class_weights = torch.tensor([weight_nonhab, weight_hab], dtype=torch.float32)
    
    print("\n--- Training Set Imbalance Calculated ---")
    print(f"Non-HAB Samples (Class 0): {count_nonhab} -> Weight: {weight_nonhab:.3f}")
    print(f"HAB Samples (Class 1):     {count_hab} -> Weight: {weight_hab:.3f}")
    
    return df, class_weights

# ==============================================================================
# MODULE 3: UNIVERSAL PYTORCH DATASET
# ==============================================================================

class UniversalWaterDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, mode='training', num_bands=10):
        """
        dataframe: The master dataframe from Module 2 (must contain 'folder_path', 'binary_label', 'strat_group')
        mode: 'training', 'validation', or 'test'
        num_bands: 10 (for BigEarthNet) or 12 (for Generic/S2)
        """
        self.df = dataframe[dataframe['split'] == mode].reset_index(drop=True)
        self.mode = mode
        self.num_bands = num_bands
        
        # Define the 12-band stack
        self.band_names_12 = [
            "B01_raw.tif", "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", 
            "B05_raw.tif", "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", 
            "B8A_raw.tif", "B09_raw.tif", "B11_raw.tif", "B12_raw.tif"
        ]
        
        # Define the 10-band stack (Skipping B01 and B09)
        self.band_names_10 = [
                           "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", 
            "B05_raw.tif", "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", 
            "B8A_raw.tif",                "B11_raw.tif", "B12_raw.tif"
        ]
        
        self.active_bands = self.band_names_10 if self.num_bands == 10 else self.band_names_12

    def __len__(self):
        return len(self.df)

    def apply_augmentations(self, tensor):
        if random.random() > 0.5: tensor = TF.hflip(tensor)
        if random.random() > 0.5: tensor = TF.vflip(tensor)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0: tensor = TF.rotate(tensor, angle)
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_path = row['folder_path']
        label = row['binary_label']
        strat_group = row['strat_group'] 
        
        band_data = []
        for b_name in self.active_bands:
            with rasterio.open(os.path.join(folder_path, b_name)) as src:
                band_data.append(src.read(1).astype(np.float32))
        
        # Stack and normalize to roughly 0-1
        bands_stack = np.stack(band_data, axis=0)
        tensor = torch.from_numpy(bands_stack) / 10000.0
        
        # Note: Interpolation is removed since IW and OW are both natively 256x256 now!
        
        if self.mode == 'training':
            tensor = self.apply_augmentations(tensor)
            
        return tensor, label, strat_group


# ==============================================================================
# MODULE 4: MULTI-MODEL BUILDER
# ==============================================================================

def build_water_cnn(architecture='resnet18', num_bands=10, mode='generic', weights_path=None, 
                    donor_stem_path=None, donor_body_path=None):
    """
    Constructs the requested ResNet architecture, alters the input channels,
    loads the appropriate pre-trained weights safely, and alters the output head.
    """
    
    # 1. Base Architecture Selection
    if architecture == 'resnet18':
        model = models.resnet18(weights=None)
        default_weights = models.ResNet18_Weights.DEFAULT
    elif architecture == 'resnet34':
        model = models.resnet34(weights=None)
        default_weights = models.ResNet34_Weights.DEFAULT
    else:
        raise ValueError(f"Architecture {architecture} not supported.")

    # 2. Stem Surgery (Adjust input channels)
    model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 3. Load Pre-trained Weights based on Mode
    if mode == 'generic':
        print(f"[{architecture.upper()}] Mode: Generic - Inflating ImageNet weights to {num_bands} bands...")
        temp_model = models.resnet18(weights=default_weights) if architecture == 'resnet18' else models.resnet34(weights=default_weights)
        
        with torch.no_grad():
            w_avg = temp_model.conv1.weight.mean(dim=1, keepdim=True)
            model.conv1.weight.copy_(w_avg.repeat(1, num_bands, 1, 1))
            
        state_dict = temp_model.state_dict()
        del state_dict['conv1.weight']
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False)

    elif mode == 'bigearthnet':
        if num_bands != 10: raise ValueError("BigEarthNet strictly requires 10 bands.")
        print(f"[{architecture.upper()}] Mode: BigEarthNet - Loading 10-band weights...")
        if weights_path.endswith('.safetensors'): state_dict = load_file(weights_path)
        else: state_dict = torch.load(weights_path, map_location='cpu')
            
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('backbone.', '').replace('model.vision_encoder.', '')
            if 'fc.' in name: continue
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)

    elif mode == 'frank_custom_r34':
        if architecture != 'resnet34' or num_bands != 10:
            raise ValueError("Frankenstein requires 'resnet34' and 10 bands.")
        if not donor_stem_path or not donor_body_path:
            raise ValueError("You must provide both donor_stem_path (R18) and donor_body_path (R34).")
            
        print(f"[{architecture.upper()}] Mode: Frankenstein - Grafting Trained R18 onto Trained R34...")
        
        # A. Configure Head for 2 classes immediately
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        # B. Load Trained R34 (The Deep Body)
        body_ckpt = torch.load(donor_body_path, map_location='cpu')
        body_dict = body_ckpt['state_dict'] if 'state_dict' in body_ckpt else body_ckpt
        
        clean_body = {}
        for k, v in body_dict.items():
            if k.startswith('model.'):
                clean_body[k.replace('model.', '')] = v
            elif not 'class_weights' in k and not 'criterion' in k: # fallback if it's already a raw .pth
                clean_body[k] = v
                
        model.load_state_dict(clean_body, strict=False)
        print("-> Step 1: Loaded Custom R34 Body Weights.")
        
        # C. Load Trained R18 (The Early Stem)
        stem_ckpt = torch.load(donor_stem_path, map_location='cpu')
        stem_dict = stem_ckpt['state_dict'] if 'state_dict' in stem_ckpt else stem_ckpt
        
        clean_stem = {}
        for k, v in stem_dict.items():
            name = k.replace('model.', '').replace('module.', '').replace('backbone.', '')
            if 'fc.' in name: continue
            clean_stem[name] = v

        # D. SURGERY: Overwrite matching layers
        r34_dict = model.state_dict()
        overwritten_count = 0
        
        for name, r18_weight in clean_stem.items():
            if name in r34_dict and r34_dict[name].shape == r18_weight.shape:
                r34_dict[name] = r18_weight
                overwritten_count += 1
                
        model.load_state_dict(r34_dict)
        print(f"-> Step 2: Overwrote {overwritten_count} early layers with Trained R18 weights.")
        
        return model

    # 4. Standard Head Surgery (for generic and bigearthnet models)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    return model

# ==============================================================================
# MODULE 5: ADVANCED PYTORCH LIGHTNING SYSTEM
# ==============================================================================

class HABLightningSystem(L.LightningModule):
    def __init__(self, architecture, num_bands, mode, weights_path, lr, class_weights, 
                 donor_stem_path=None, donor_body_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        
        # 1. Build Model using Module 4
        self.model = build_water_cnn(
            architecture=architecture, 
            num_bands=num_bands, 
            mode=mode, 
            weights_path=weights_path,
            donor_stem_path=donor_stem_path,
            donor_body_path=donor_body_path
        )
        
        # 2. Loss Function (Weighted)
        self.register_buffer("class_weights", class_weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        # 3. Overall Standard Metrics
        self.train_f1 = MulticlassF1Score(num_classes=2, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=2, average='macro')
        
        self.train_acc = MulticlassAccuracy(num_classes=2, average='micro')
        self.val_acc = MulticlassAccuracy(num_classes=2, average='micro')
        
        # 4: Balanced Accuracy (Macro Average Accuracy)
        self.train_bal_acc = MulticlassAccuracy(num_classes=2, average='macro')
        self.val_bal_acc = MulticlassAccuracy(num_classes=2, average='macro')
        
        # 5. Storage for granular sub-domain tracking
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch  
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_f1(preds, y)
        self.train_acc(preds, y)
        self.train_bal_acc(preds, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1_macro", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_bal_acc", self.train_bal_acc, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, strat_groups = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_f1(preds, y)
        self.val_acc(preds, y)
        self.val_bal_acc(preds, y) 
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.append({
            'preds': preds.cpu(),
            'targets': y.cpu(),
            'groups': strat_groups
        })
        
        return loss

    def on_validation_epoch_end(self):
        # 1. Log overall metrics
        self.log("val_f1_macro", self.val_f1.compute(), prog_bar=True)
        self.log("val_acc_overall", self.val_acc.compute())
        self.log("val_bal_acc", self.val_bal_acc.compute()) 
        
        # 2. Extract all batches
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        all_groups = [g for x in self.validation_step_outputs for g in x['groups']]
        
        # 3. Calculate accuracy for every specific sub-domain
        unique_groups = ['iw_hab', 'ow_hab', 'iw_nonhab', 'ow_nonhab', 'land', 'clouds']
        
        for group in unique_groups:
            indices = [i for i, g in enumerate(all_groups) if g == group]
            if len(indices) > 0:
                group_preds = all_preds[indices]
                group_targets = all_targets[indices]
                acc = (group_preds == group_targets).float().mean()
                self.log(f"val_{group}_acc", acc)
            else:
                self.log(f"val_{group}_acc", 0.0)

        # 4. Reset for the next epoch
        self.val_f1.reset()
        self.val_acc.reset()
        self.val_bal_acc.reset()
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1_macro"}
        }



# ==============================================================================
# MODULE 6: PLOTTING
# ==============================================================================


def plot_training_history(csv_path, output_dir):
    """Reads metrics.csv and plots Loss, F1, Accuracy, Balanced Accuracy, and Sub-domain accuracies."""
    if not os.path.exists(csv_path):
        print(f"Metrics not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)

    def save_plot(metrics_list, labels_list, title, filename):
        plt.figure(figsize=(10, 6))
        for metric, label in zip(metrics_list, labels_list):
            if metric in df.columns:
                clean_df = df[['epoch', metric]].dropna()
                plt.plot(clean_df['epoch'], clean_df[metric], marker='o', label=label)
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # 1. Standard Curves
    save_plot(['train_loss', 'val_loss'], ['Train Loss', 'Val Loss'], 'Loss Curve', '1_loss.png')
    save_plot(['train_f1_macro', 'val_f1_macro'], ['Train F1', 'Val F1'], 'Macro F1 Score', '2_f1_macro.png')
    save_plot(['train_acc', 'val_acc_overall'], ['Train Accuracy', 'Val Accuracy'], 'Overall Accuracy (Micro)', '3_accuracy.png')
    save_plot(['train_bal_acc', 'val_bal_acc'], ['Train Balanced Acc', 'Val Balanced Acc'], 'Balanced Accuracy (Macro)', '4_balanced_accuracy.png')

    # 2. Granular Sub-Domain Validation Curves
    granular_metrics = [
        'val_iw_hab_acc', 'val_ow_hab_acc', 
        'val_iw_nonhab_acc', 'val_ow_nonhab_acc', 
        'val_land_acc', 'val_clouds_acc'
    ]
    labels = ['IW HAB', 'OW HAB', 'IW nonHAB', 'OW nonHAB', 'Land', 'Clouds']
    save_plot(granular_metrics, labels, 'Validation Accuracy by Sub-Domain', '5_granular_accuracies.png')
    
    
def plot_paired_scatter(csv_path, output_dir):
    """Generates individual, high-res academic scatter plots for each metric."""
    df = pd.read_csv(csv_path)
    
    # List of (R18_column, Mosaic_column, Plot Title, filename_suffix)
    metrics = [
        ('r18_f1', 'graft_f1', 'F1 Score (Macro)', 'f1_score'),
        ('r18_acc', 'graft_acc', 'Overall Accuracy', 'accuracy_overall'),
        ('r18_bal_acc', 'graft_bal_acc', 'Balanced Accuracy', 'balanced_accuracy'),
        ('r18_iw_hab_acc', 'graft_iw_hab_acc', 'IW HABs Accuracy', 'iw_hab_acc'),
        ('r18_ow_hab_acc', 'graft_ow_hab_acc', 'OW HABs Accuracy', 'ow_hab_acc'),
        ('r18_iw_nonhab_acc', 'graft_iw_nonhab_acc', 'IW non-HABs Accuracy', 'iw_nonhab_acc'),
        ('r18_ow_nonhab_acc', 'graft_ow_nonhab_acc', 'OW non-HABs Accuracy', 'ow_nonhab_acc'),
        ('r18_land_acc', 'graft_land_acc', 'Land Accuracy', 'land_acc'),
        ('r18_clouds_acc', 'graft_clouds_acc', 'Clouds Accuracy', 'clouds_acc')
    ]
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[*] Generating Individual Scatter Plots...")
    for m_r18, m_graft, title, filename in metrics:
        
        # Safeguard: Drop iterations where the test split had 0 images for this specific class
        valid_df = df.dropna(subset=[m_r18, m_graft])
        
        if len(valid_df) == 0:
            print(f" -> Skipping {title} (No data available in test splits)")
            continue
            
        plt.figure(figsize=(8, 8))
        
        # Plot the valid iterations
        sns.scatterplot(x=valid_df[m_r18], y=valid_df[m_graft], s=150, color='blue', edgecolor='black', alpha=0.8)
        
        # Dynamically calculate square limits with a nice buffer
        min_val = min(valid_df[m_r18].min(), valid_df[m_graft].min()) - 0.02
        max_val = max(valid_df[m_r18].max(), valid_df[m_graft].max()) + 0.02
        
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        
        # Draw Parity Line
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label="Parity (y = x)")
        
        # Fill Win Zones
        plt.fill_between([min_val, max_val], [min_val, max_val], max_val, color='green', alpha=0.05, label="Mosaic Wins")
        plt.fill_between([min_val, max_val], min_val, [min_val, max_val], color='red', alpha=0.05, label="R18 Wins")
        
        plt.title(f"Monte Carlo Duel: {title}", fontsize=16, fontweight='bold')
        plt.xlabel("Baseline ResNet-18 Score", fontsize=14)
        plt.ylabel("Mosaic ResNet-34 Score", fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='upper left', fontsize=12)

        plt.tight_layout()
        out_file = os.path.join(output_dir, f"scatter_{filename}.png")
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f" -> Saved: {out_file}")


def evaluate_split(model, loader, device, split_name, output_dir):
    """Runs final inference, saves Confusion Matrix, and returns detailed metrics."""
    model.eval()
    all_preds, all_labels, all_groups = [], [], []

    with torch.no_grad():
        for images, labels, groups in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_groups.extend(groups)

    # 1. Print Standard Report
    print(f"\n--- Results for {split_name} ---")
    print(classification_report(all_labels, all_preds, target_names=["nonHAB", "HAB"], zero_division=0))

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["nonHAB", "HAB"], yticklabels=["nonHAB", "HAB"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({split_name})")
    plt.savefig(os.path.join(output_dir, f"cm_{split_name.lower()}.png"))
    plt.close()

    # 3. Calculate all requested metrics
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    results_df = pd.DataFrame({'Actual': all_labels, 'Predicted': all_preds, 'Group': all_groups})
    results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

    metrics_dict = {
        'f1': f1,
        'acc': acc,
        'bal_acc': bal_acc
    }
    
    unique_groups = ['iw_hab', 'iw_nonhab', 'ow_hab', 'ow_nonhab', 'clouds', 'land']
    for group in unique_groups:
        group_data = results_df[results_df['Group'] == group]
        if not group_data.empty:
            metrics_dict[f'{group}_acc'] = group_data['Correct'].mean()
        else:
            metrics_dict[f'{group}_acc'] = np.nan

    return metrics_dict


if __name__ == "__main__":
    # 1. Define Master Paths
    IW_DIR = "/home/kostas/AMFITRITE/data256"
    OW_DIR = "/home/kostas/AMFITRITE/OWdata"
    IW_EXCEL = "/home/kostas/AMFITRITE/dataset_summary_256x256pixels.xlsx"
    OW_CSV = "/home/kostas/AMFITRITE/OWdata/amfitrite_open_waters_master.csv"
    
    BASE_OUTPUT_DIR = "/home/kostas/AMFITRITE/IW_and_OW_CNN/res18_bigearth_vs_farnk_v2"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print(f"Hardware initialized on: {DEVICE}")
    print("="*60)
    
    # 2. Global Configuration
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    TOTAL_ITERATIONS = 10
    
    # BigEarthNet Master Weights Path
    BIGEARTH_WEIGHTS = '/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0_resnet18-s2-v0.2.0/model.safetensors'
    
    # Run Module 1 (Only harmonize once!)
    harmonizer = DatasetHarmonizer(IW_DIR, OW_DIR, IW_EXCEL, OW_CSV)
    master_dataframe = harmonizer.create_master_registry()
    
    master_results = []

    # ==========================================================================
    # THE 10-ITERATION MONTE CARLO LOOP
    # ==========================================================================
    for iteration in range(1, TOTAL_ITERATIONS + 1):
        print("\n" + "X"*60)
        print(f"XXX STARTING MONTE CARLO ITERATION {iteration}/{TOTAL_ITERATIONS} XXX")
        print("X"*60)
        
        L.seed_everything(1234 + iteration, workers=True)
        
        iter_dir = os.path.join(BASE_OUTPUT_DIR, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # 1. NEW DATA SPLIT (Guaranteed to be randomly unique every iteration)
        split_csv = os.path.join(iter_dir, f"split_iter_{iteration}.csv")
        split_df, class_weights = create_stratified_split_and_weights(master_dataframe, split_csv, random_seed=42+iteration)

        # 2. DATA LOADERS (All strictly 10 bands for a mathematically fair duel)
        train_ds = UniversalWaterDataset(split_df, mode='training', num_bands=10)
        val_ds   = UniversalWaterDataset(split_df, mode='validation', num_bands=10)
        test_ds  = UniversalWaterDataset(split_df, mode='test', num_bands=10)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        # Dictionary to hold the 3-model Test metrics for this loop
        iter_results = {'iteration': iteration}

        # ----------------------------------------------------------------------
        # PHASE A: Train BigEarthNet ResNet-18 (The Baseline & Stem Donor)
        # ----------------------------------------------------------------------
        print("\n[PHASE A] Training Baseline ResNet-18 (Stem Donor)...")
        model_r18 = HABLightningSystem(
            architecture='resnet18', num_bands=10, mode='bigearthnet',
            weights_path=BIGEARTH_WEIGHTS, lr=1e-4, class_weights=class_weights
        )
        
        ckpt_cb_r18 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1, filename="best-r18-{epoch:02d}")
        trainer_r18 = L.Trainer(
            max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="logs_r18"), 
            callbacks=[ckpt_cb_r18, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False
        )
        
        trainer_r18.fit(model_r18, train_loader, val_loader)
        path_r18 = ckpt_cb_r18.best_model_path
        
        # Evaluate & Log
        best_r18 = HABLightningSystem.load_from_checkpoint(path_r18, class_weights=class_weights).to(DEVICE)
        metrics_r18 = evaluate_split(best_r18, test_loader, DEVICE, "Test_R18", iter_dir)
        for k, v in metrics_r18.items(): 
            iter_results[f"r18_{k}"] = v
        
        del model_r18, trainer_r18, best_r18; gc.collect(); torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # PHASE B: Train Generic ResNet-34 (The Body Donor)
        # ----------------------------------------------------------------------
        print("\n[PHASE B] Training Generic ResNet-34 (Body Donor)...")
        model_r34 = HABLightningSystem(
            architecture='resnet34', num_bands=10, mode='generic',
            weights_path=None, lr=1e-4, class_weights=class_weights
        )
        
        ckpt_cb_r34 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1, filename="best-r34-{epoch:02d}")
        trainer_r34 = L.Trainer(
            max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="logs_r34"), 
            callbacks=[ckpt_cb_r34, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False
        )
        
        trainer_r34.fit(model_r34, train_loader, val_loader)
        path_r34 = ckpt_cb_r34.best_model_path
        
        # Evaluate & Log
        best_r34 = HABLightningSystem.load_from_checkpoint(path_r34, class_weights=class_weights).to(DEVICE)
        metrics_r34 = evaluate_split(best_r34, test_loader, DEVICE, "Test_R34", iter_dir)
        for k, v in metrics_r34.items(): 
            iter_results[f"r34_{k}"] = v
        
        del model_r34, trainer_r34, best_r34; gc.collect(); torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # PHASE C: Graft & Train Mosaic ResNet-34
        # ----------------------------------------------------------------------
        print("\n[PHASE C] Grafting & Training Mosaic ResNet-34...")
        model_frank = HABLightningSystem(
            architecture='resnet34', num_bands=10, mode='frank_custom_r34',
            weights_path=None, lr=1e-4, class_weights=class_weights,
            donor_stem_path=path_r18, donor_body_path=path_r34
        )
        
        ckpt_cb_frank = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1, filename="best-mosaic-{epoch:02d}")
        trainer_frank = L.Trainer(
            max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="logs_mosaic"), 
            callbacks=[ckpt_cb_frank, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False
        )
        
        trainer_frank.fit(model_frank, train_loader, val_loader)
        path_frank = ckpt_cb_frank.best_model_path
        
        # Evaluate & Log
        best_frank = HABLightningSystem.load_from_checkpoint(path_frank, class_weights=class_weights).to(DEVICE)
        metrics_frank = evaluate_split(best_frank, test_loader, DEVICE, "Test_Mosaic", iter_dir)
        for k, v in metrics_frank.items(): iter_results[f"graft_{k}"] = v
        
        del model_frank, trainer_frank, best_frank; gc.collect(); torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # END OF LOOP: Backup & Save
        # ----------------------------------------------------------------------
        master_results.append(iter_results)
        
        # Save a live backup so you don't lose days of work if it crashes!
        backup_csv = os.path.join(BASE_OUTPUT_DIR, "running_backup_results.csv")
        pd.DataFrame(master_results).to_csv(backup_csv, index=False)
        print(f"\n[Iteration {iteration} Complete] Results backed up to {backup_csv}")
        
        # Delete dataloaders just to be safe
        del train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
        gc.collect()

    # ==========================================================================
    # FINAL EXPORT & PLOTTING
    # ==========================================================================
    print("\n" + "="*60)
    print("ALL 10 ITERATIONS COMPLETED!")
    print("="*60)
    
    final_csv_path = os.path.join(BASE_OUTPUT_DIR, "final_duel_results.csv")
    pd.DataFrame(master_results).to_csv(final_csv_path, index=False)
    
    # Generate the masterpiece plot
    plot_output_dir = os.path.join(BASE_OUTPUT_DIR, "scatter_plots")
    plot_paired_scatter(final_csv_path, plot_output_dir)

