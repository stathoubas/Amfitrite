# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:45:29 2026

@author: K. Pikounis
"""

# -*- coding: utf-8 -*-
"""
Domain-Specific Fine-Tuning: Inland Water (Augmented with Land/Clouds)

Runs 10 Monte Carlo Iterations evaluating 3 ResNet-18 variants:
1. Baseline (BigEarthNet Weights -> Trained on IW)
2. Zero-Shot (Universal Foundation Model -> Evaluated on IW)
3. Fine-Tuned (Universal Foundation Model -> Trained on IW)
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

import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, balanced_accuracy_score
from safetensors.torch import load_file

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
        print("Loading Inland Water (IW) Dataset...")
        df_iw = pd.read_excel(self.iw_excel)
        
        standardized_rows = []
        for _, row in df_iw.iterrows():
            uid = str(row['uid'])
            folder_path = os.path.join(self.iw_dir, uid)
            
            indicative_class = str(row['new_indicative_class']).strip().lower()
            if indicative_class in ['high', 'moderate']:
                binary_label = 1
                strat_group = 'iw_hab'
            else:
                binary_label = 0
                strat_group = 'iw_nonhab'
                
            if os.path.isdir(folder_path):
                standardized_rows.append({
                    'uid': uid, 'folder_path': folder_path,
                    'binary_label': binary_label, 'strat_group': strat_group, 'source': 'IW'
                })
        return pd.DataFrame(standardized_rows)

    def process_ow_data(self):
        print("Loading Open Water (OW) Dataset...")
        df_ow = pd.read_csv(self.ow_csv)
        
        standardized_rows = []
        for _, row in df_ow.iterrows():
            uid = str(row['ID'])
            
            if row['tile_is_hab'] == True:
                suffix, binary_label, strat_group = "HAB", 1, "ow_hab"
            else:
                binary_label = 0
                if row['tile_status'] == 'land':
                    suffix, strat_group = "land", "land"
                elif row['tile_status'] == 'clouds':
                    suffix, strat_group = "clouds", "clouds"
                else:
                    suffix, strat_group = "nonHAB", "ow_nonhab"
                    
            folder_path = os.path.join(self.ow_dir, f"{uid}_{suffix}")
            if os.path.isdir(folder_path):
                standardized_rows.append({
                    'uid': uid, 'folder_path': folder_path,
                    'binary_label': binary_label, 'strat_group': strat_group, 'source': 'OW'
                })
        return pd.DataFrame(standardized_rows)

    def create_master_registry(self):
        df_iw = self.process_iw_data()
        df_ow = self.process_ow_data()
        return pd.concat([df_iw, df_ow], ignore_index=True)


# ==============================================================================
# MODULE 2: SPLITTER & IMBALANCE CALCULATOR
# ==============================================================================

def create_stratified_split_and_weights(master_df, output_registry_path, random_seed=42):
    print("\nPerforming 70/15/15 Stratified Split on Augmented Dataset...")
    df = master_df.copy()
    df['split'] = 'junk'
    
    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.30, stratify=df['strat_group'], random_state=random_seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=df.loc[temp_idx, 'strat_group'], random_state=random_seed
    )
    
    df.loc[train_idx, 'split'] = 'training'
    df.loc[val_idx, 'split'] = 'validation'
    df.loc[test_idx, 'split'] = 'test'
    df.to_csv(output_registry_path, index=False)

    # Class Weights based ONLY on Training set
    train_df = df[df['split'] == 'training']
    count_nonhab = len(train_df[train_df['binary_label'] == 0])
    count_hab = len(train_df[train_df['binary_label'] == 1])
    total = count_nonhab + count_hab
    
    weight_nonhab = total / (2.0 * count_nonhab)
    weight_hab = total / (2.0 * count_hab)
    class_weights = torch.tensor([weight_nonhab, weight_hab], dtype=torch.float32)
    
    return df, class_weights

# ==============================================================================
# MODULE 3: UNIVERSAL PYTORCH DATASET
# ==============================================================================

class UniversalWaterDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, mode='training', num_bands=10):
        self.df = dataframe[dataframe['split'] == mode].reset_index(drop=True)
        self.mode = mode
        self.num_bands = num_bands
        
        self.active_bands = [
                           "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", 
            "B05_raw.tif", "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", 
            "B8A_raw.tif",                "B11_raw.tif", "B12_raw.tif"
        ]

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
        
        tensor = torch.from_numpy(np.stack(band_data, axis=0)) / 10000.0
        
        if self.mode == 'training':
            tensor = self.apply_augmentations(tensor)
            
        return tensor, label, strat_group


# ==============================================================================
# MODULE 4: MULTI-MODEL BUILDER
# ==============================================================================

def build_water_cnn(architecture='resnet18', num_bands=10, mode='generic', weights_path=None):
    if architecture != 'resnet18':
        raise ValueError("This script is strictly constrained to ResNet-18 duels.")

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if mode == 'bigearthnet':
        print("[RESNET18] Mode: BigEarthNet Baseline - Loading safe tensors...")
        if weights_path.endswith('.safetensors'): state_dict = load_file(weights_path)
        else: state_dict = torch.load(weights_path, map_location='cpu')
            
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        
        new_state_dict = {
            k.replace('module.', '').replace('backbone.', '').replace('model.vision_encoder.', ''): v 
            for k, v in state_dict.items() if not 'fc.' in k
        }
        model.load_state_dict(new_state_dict, strict=False)
        
        # Modify Head for Binary Classification
        model.fc = nn.Linear(model.fc.in_features, 2)

    elif mode == 'custom_r18':
        print("[RESNET18] Mode: Universal Foundation - Loading custom weights...")
        
        # Alter the head FIRST so the shape matches the incoming checkpoint (2 classes)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        ckpt = torch.load(weights_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Robust dictionary mapping: Handles both raw .pth and lightning-prefixed dictionaries
        clean_dict = {}
        for k, v in state_dict.items():
            # If the prefix 'model.' exists, strip it. Otherwise, keep the key as is.
            clean_key = k.replace('model.', '') if k.startswith('model.') else k
            clean_dict[clean_key] = v
            
        model.load_state_dict(clean_dict, strict=True)
        print("-> Universal Foundation Weights mapped successfully.")

    return model

# ==============================================================================
# MODULE 5: ADVANCED PYTORCH LIGHTNING SYSTEM
# ==============================================================================

class HABLightningSystem(L.LightningModule):
    def __init__(self, architecture, num_bands, mode, weights_path, lr, class_weights):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        
        self.model = build_water_cnn(architecture, num_bands, mode, weights_path)
        
        self.register_buffer("class_weights", class_weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        self.train_f1 = MulticlassF1Score(num_classes=2, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=2, average='macro')
        self.train_acc = MulticlassAccuracy(num_classes=2, average='micro')
        self.val_acc = MulticlassAccuracy(num_classes=2, average='micro')
        self.train_bal_acc = MulticlassAccuracy(num_classes=2, average='macro')
        self.val_bal_acc = MulticlassAccuracy(num_classes=2, average='macro')

    def forward(self, x): return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch  
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.train_f1(preds, y); self.train_acc(preds, y); self.train_bal_acc(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_f1_macro", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.val_f1(preds, y); self.val_acc(preds, y); self.val_bal_acc(preds, y) 
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_f1_macro", self.val_f1, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1_macro"}}

# ==============================================================================
# MODULE 6: PLOTTING & EVALUATION
# ==============================================================================

def plot_paired_scatter(csv_path, output_dir):
    """Generates individual, high-res academic scatter plots comparing Baseline vs Fine-Tuned."""
    df = pd.read_csv(csv_path)
    
    # We compare Baseline BigEarth (baseline_) vs Fine-Tuned Foundation (ftfound_)
    metrics = [
        ('baseline_f1', 'ftfound_f1', 'F1 Score (Macro)', 'f1_score'),
        ('baseline_acc', 'ftfound_acc', 'Overall Accuracy', 'accuracy_overall'),
        ('baseline_bal_acc', 'ftfound_bal_acc', 'Balanced Accuracy', 'balanced_accuracy'),
        ('baseline_iw_hab_acc', 'ftfound_iw_hab_acc', 'IW HABs Accuracy', 'iw_hab_acc'),
        ('baseline_iw_nonhab_acc', 'ftfound_iw_nonhab_acc', 'IW non-HABs Accuracy', 'iw_nonhab_acc'),
        ('baseline_land_acc', 'ftfound_land_acc', 'Land Accuracy', 'land_acc'),
        ('baseline_clouds_acc', 'ftfound_clouds_acc', 'Clouds Accuracy', 'clouds_acc')
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for m_base, m_ft, title, filename in metrics:
        valid_df = df.dropna(subset=[m_base, m_ft])
        if len(valid_df) == 0: continue
            
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=valid_df[m_base], y=valid_df[m_ft], s=150, color='blue', edgecolor='black', alpha=0.8)
        
        min_val = min(valid_df[m_base].min(), valid_df[m_ft].min()) - 0.02
        max_val = max(valid_df[m_base].max(), valid_df[m_ft].max()) + 0.02
        
        plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label="Parity (y = x)")
        
        plt.fill_between([min_val, max_val], [min_val, max_val], max_val, color='green', alpha=0.05, label="Foundation Wins")
        plt.fill_between([min_val, max_val], min_val, [min_val, max_val], color='red', alpha=0.05, label="Baseline Wins")
        
        plt.title(f"Monte Carlo Duel: {title}", fontsize=16, fontweight='bold')
        plt.xlabel("Baseline (BigEarthNet) Score", fontsize=14)
        plt.ylabel("Fine-Tuned (Foundation) Score", fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_{filename}.png"), dpi=300)
        plt.close()

def evaluate_split(model, loader, device, split_name, output_dir):
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

    print(f"\n--- Results for {split_name} ---")
    print(classification_report(all_labels, all_preds, target_names=["nonHAB", "HAB"], zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["nonHAB", "HAB"], yticklabels=["nonHAB", "HAB"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix ({split_name})")
    plt.savefig(os.path.join(output_dir, f"cm_{split_name.lower()}.png"))
    plt.close()

    results_df = pd.DataFrame({'Actual': all_labels, 'Predicted': all_preds, 'Group': all_groups})
    results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

    metrics_dict = {
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'acc': accuracy_score(all_labels, all_preds),
        'bal_acc': balanced_accuracy_score(all_labels, all_preds)
    }
    
    for group in ['iw_hab', 'iw_nonhab', 'ow_hab', 'ow_nonhab', 'clouds', 'land']:
        group_data = results_df[results_df['Group'] == group]
        metrics_dict[f'{group}_acc'] = group_data['Correct'].mean() if not group_data.empty else np.nan

    return metrics_dict


if __name__ == "__main__":
    IW_DIR = "/home/kostas/AMFITRITE/data256"
    OW_DIR = "/home/kostas/AMFITRITE/OWdata"
    IW_EXCEL = "/home/kostas/AMFITRITE/dataset_summary_256x256pixels.xlsx"
    OW_CSV = "/home/kostas/AMFITRITE/OWdata/amfitrite_open_waters_master.csv"
    
    BASE_OUTPUT_DIR = "/home/kostas/AMFITRITE/IW_and_OW_CNN/iw_augmented_specialization"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware initialized on: {DEVICE}")
    
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    TOTAL_ITERATIONS = 10
    
    # 1. Provide exact paths to your pre-trained files here:
    BIGEARTH_WEIGHTS = '/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0_resnet18-s2-v0.2.0/model.safetensors'
    UNIVERSAL_FOUNDATION_WEIGHTS = '/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/amfitrite_resnet18_bigearth_best.pth' 
    
    # 2. Harmonize & Filter for Augmented IW Dataset
    harmonizer = DatasetHarmonizer(IW_DIR, OW_DIR, IW_EXCEL, OW_CSV)
    master_dataframe = harmonizer.create_master_registry()
    
    print("\n[Data Engineering] Filtering strictly for IW images + OW Clouds + OW Land...")
    iw_augmented_df = master_dataframe[
        (master_dataframe['source'] == 'IW') | 
        (master_dataframe['strat_group'].isin(['clouds', 'land']))
    ].reset_index(drop=True)
    
    print(f"Augmented IW Dataset Size: {len(iw_augmented_df)} total images.")
    
    master_results = []

    # ==========================================================================
    # THE 10-ITERATION MONTE CARLO LOOP
    # ==========================================================================
    for iteration in range(1, TOTAL_ITERATIONS + 1):
        print("\n" + "X"*60)
        print(f"XXX MONTE CARLO ITERATION {iteration}/{TOTAL_ITERATIONS} XXX")
        print("X"*60)
        
        L.seed_everything(1234 + iteration, workers=True)
        iter_dir = os.path.join(BASE_OUTPUT_DIR, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        split_csv = os.path.join(iter_dir, f"split_iter_{iteration}.csv")
        split_df, class_weights = create_stratified_split_and_weights(iw_augmented_df, split_csv, random_seed=42+iteration)

        train_ds = UniversalWaterDataset(split_df, mode='training', num_bands=10)
        val_ds   = UniversalWaterDataset(split_df, mode='validation', num_bands=10)
        test_ds  = UniversalWaterDataset(split_df, mode='test', num_bands=10)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        iter_results = {'iteration': iteration}

        # ----------------------------------------------------------------------
        # PHASE A: Train Baseline (BigEarthNet -> Fine-Tuned)
        # ----------------------------------------------------------------------
        print("\n[PHASE A] Training Baseline BigEarthNet ResNet-18...")
        model_base = HABLightningSystem(
            architecture='resnet18', num_bands=10, mode='bigearthnet',
            weights_path=BIGEARTH_WEIGHTS, lr=1e-4, class_weights=class_weights
        )
        
        ckpt_cb_base = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1, filename="best-baseline-{epoch:02d}")
        trainer_base = L.Trainer(
            max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="logs_baseline"), 
            callbacks=[ckpt_cb_base, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False
        )
        trainer_base.fit(model_base, train_loader, val_loader)
        
        best_base = HABLightningSystem.load_from_checkpoint(ckpt_cb_base.best_model_path, class_weights=class_weights).to(DEVICE)
        metrics_base = evaluate_split(best_base, test_loader, DEVICE, "Test_Baseline", iter_dir)
        for k, v in metrics_base.items(): iter_results[f"baseline_{k}"] = v
        
        del model_base, trainer_base, best_base; gc.collect(); torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # PHASE B: Evaluate Zero-Shot (Foundation Model without training)
        # ----------------------------------------------------------------------
        print("\n[PHASE B] Evaluating Zero-Shot Universal Foundation Model...")
        model_zero = HABLightningSystem(
            architecture='resnet18', num_bands=10, mode='custom_r18',
            weights_path=UNIVERSAL_FOUNDATION_WEIGHTS, lr=1e-4, class_weights=class_weights
        ).to(DEVICE)
        
        metrics_zero = evaluate_split(model_zero, test_loader, DEVICE, "Test_ZeroShot", iter_dir)
        for k, v in metrics_zero.items(): iter_results[f"zeroshot_{k}"] = v
        
        del model_zero; gc.collect(); torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # PHASE C: Train Specialized Model (Foundation -> Fine-Tuned)
        # ----------------------------------------------------------------------
        print("\n[PHASE C] Training Specialized Model from Foundation Weights...")
        model_ft = HABLightningSystem(
            architecture='resnet18', num_bands=10, mode='custom_r18',
            weights_path=UNIVERSAL_FOUNDATION_WEIGHTS, lr=1e-5, class_weights=class_weights # NOTE: Ultra-low learning rate!
        )
        
        ckpt_cb_ft = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1, filename="best-ftfound-{epoch:02d}")
        trainer_ft = L.Trainer(
            max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="logs_ftfound"), 
            callbacks=[ckpt_cb_ft, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False
        )
        trainer_ft.fit(model_ft, train_loader, val_loader)
        
        best_ft = HABLightningSystem.load_from_checkpoint(ckpt_cb_ft.best_model_path, class_weights=class_weights).to(DEVICE)
        metrics_ft = evaluate_split(best_ft, test_loader, DEVICE, "Test_FT_Foundation", iter_dir)
        for k, v in metrics_ft.items(): iter_results[f"ftfound_{k}"] = v
        
        del model_ft, trainer_ft, best_ft; gc.collect(); torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # Backup and clear
        # ----------------------------------------------------------------------
        master_results.append(iter_results)
        backup_csv = os.path.join(BASE_OUTPUT_DIR, "running_backup_results.csv")
        pd.DataFrame(master_results).to_csv(backup_csv, index=False)
        print(f"\n[Iteration {iteration} Complete] Results backed up.")
        
        del train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
        gc.collect()

    print("\n" + "="*60)
    print("ALL 10 ITERATIONS COMPLETED!")
    print("="*60)
    
    final_csv_path = os.path.join(BASE_OUTPUT_DIR, "final_duel_results.csv")
    pd.DataFrame(master_results).to_csv(final_csv_path, index=False)
    
    plot_output_dir = os.path.join(BASE_OUTPUT_DIR, "scatter_plots")
    plot_paired_scatter(final_csv_path, plot_output_dir)