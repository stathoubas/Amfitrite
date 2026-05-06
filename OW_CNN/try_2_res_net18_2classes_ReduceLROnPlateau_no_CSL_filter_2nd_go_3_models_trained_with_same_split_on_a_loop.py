# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:06:18 2026

@author: K. Pikounis

Stability & Meta-Analysis Pipeline for Amfitrite Open Waters
Runs N iterations with different random splits and statistically evaluates generic, s2, and iw models.
"""

import os
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import rasterio
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.wrappers import ClasswiseWrapper
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns

# --- DATASET & MODEL DEFINITIONS ---

class HABDataset(Dataset):
    def __init__(self, dataframe, root_dir, mode='training'):
        self.df = dataframe[dataframe['split'] == mode].reset_index(drop=True)
        self.root_dir = root_dir
        self.mode = mode
        
        self.band_names = [
            "B01_raw.tif", "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", 
            "B05_raw.tif", "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", 
            "B8A_raw.tif", "B09_raw.tif", "B11_raw.tif", "B12_raw.tif"
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
        uid = str(row['ID'])
        
        folder_suffix = "HAB" if row['tile_is_hab'] else "nonHAB"
        if row['tile_status'] == "land": folder_suffix = "land"
        if row['tile_status'] == "clouds": folder_suffix = "clouds"
        
        folder_path = os.path.join(self.root_dir, f"{uid}_{folder_suffix}")
        label = 1 if row['tile_is_hab'] else 0
        strat_cat = row['strat_category'] 
        
        band_data = []
        for b_name in self.band_names:
            with rasterio.open(os.path.join(folder_path, b_name)) as src:
                band_data.append(src.read(1).astype(np.float32))
        
        bands_stack = np.stack(band_data, axis=0)
        tensor = torch.from_numpy(bands_stack) / 10000.0
        
        if self.mode == 'training':
            tensor = self.apply_augmentations(tensor)
            
        return tensor, label, strat_cat

class HABLightningModel(L.LightningModule):
    def __init__(self, mode='generic', weights_path=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._build_model(self.hparams.mode, self.hparams.weights_path)
        
        self.register_buffer("class_weights", torch.tensor([1.0, 1.0]))  
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        def get_metrics(prefix):
            return torchmetrics.MetricCollection({
                'acc': MulticlassAccuracy(num_classes=2, average='micro'),
                'bal_acc': MulticlassAccuracy(num_classes=2, average='macro'),
                'f1': MulticlassF1Score(num_classes=2, average='macro'),
                'per_class': ClasswiseWrapper(
                    MulticlassAccuracy(num_classes=2, average=None),
                    labels=["NonHAB", "HAB"]
                )
            }, prefix=prefix)

        self.train_metrics = get_metrics('train_')
        self.val_metrics = get_metrics('val_')

    def _build_model(self, mode, weights_path):
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        if mode == 'generic':
            temp_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            with torch.no_grad():
                w_avg = temp_resnet.conv1.weight.mean(dim=1, keepdim=True)
                model.conv1.weight.copy_(w_avg.repeat(1, 12, 1, 1))
            state_dict = temp_resnet.state_dict()
            del state_dict['conv1.weight']
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            model.load_state_dict(state_dict, strict=False)
                
        elif mode == 's2':
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '').replace('backbone.', '')
                if 'fc.' in name: continue 
                if name == 'conv1.weight' and v.shape[1] == 13:
                    keep_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
                    v = v[:, keep_indices, :, :]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            
        elif mode == 'iw':
            ckpt = torch.load(weights_path, map_location='cpu')
            state_dict = ckpt['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('model.', '')
                if name == 'conv1.weight' and v.shape[1] == 10:
                    new_conv1 = torch.zeros([64, 12, 7, 7], dtype=v.dtype)
                    new_conv1[:, :10, :, :] = v
                    new_conv1[:, 10:, :, :] = v.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
                    v = new_conv1
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)

        return model

    def forward(self, x): return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_metrics.update(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_metrics.update(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"}}

# --- PIPELINE FUNCTIONS ---

def get_or_create_split(excel_path, data_root, registry_path, random_seed):
    """Creates a unique split based on the random seed."""
    if os.path.exists(registry_path):
        return pd.read_csv(registry_path)

    df = pd.read_csv(excel_path)
    df['split'] = 'junk'
    
    def get_strat_cat(row):
        if row['tile_is_hab'] == True: return 'hab'
        elif row['tile_status'] == 'land': return 'land'
        elif row['tile_status'] == 'clouds': return 'clouds'
        else: return 'nonhab'
            
    df['strat_category'] = df.apply(get_strat_cat, axis=1)
    
    valid_indices = []
    for idx, row in df.iterrows():
        uid = str(row['ID'])
        folder_suffix = "HAB" if row['tile_is_hab'] else "nonHAB"
        if row['tile_status'] == "land": folder_suffix = "land"
        if row['tile_status'] == "clouds": folder_suffix = "clouds"
        
        if os.path.isdir(os.path.join(data_root, f"{uid}_{folder_suffix}")):
            valid_indices.append(idx)

    valid_df = df.loc[valid_indices].copy()
    
    # Use the dynamic seed to ensure different splits per run!
    train_idx, temp_idx = train_test_split(valid_df.index, test_size=0.30, stratify=valid_df['strat_category'], random_state=random_seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=valid_df.loc[temp_idx, 'strat_category'], random_state=random_seed)
    
    df.loc[train_idx, 'split'] = 'training'
    df.loc[val_idx, 'split'] = 'validation'
    df.loc[test_idx, 'split'] = 'test'
    
    df.to_csv(registry_path, index=False)
    return df

def evaluate_split(model, loader, device, split_name="Test", output_dir="plots"):
    """Evaluates the model and returns a dictionary of extracted metrics."""
    model.eval()
    all_preds, all_labels, all_cats = [], [], []
    
    with torch.no_grad():
        for images, labels, cats in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_cats.extend(cats)
            
    # Calculate Standard Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    results = pd.DataFrame({'Actual': all_labels, 'Predicted': all_preds, 'Category': all_cats})
    results['Correct'] = results['Actual'] == results['Predicted']
    
    metrics_dict = {
        'split': split_name,
        'accuracy': acc,
        'f1_macro': f1
    }
    
    # Calculate Category-Specific Metrics
    for cat in ["hab", "nonhab", "land", "clouds"]:
        cat_data = results[results['Category'] == cat]
        if not cat_data.empty:
            metrics_dict[f'{cat}_acc'] = cat_data['Correct'].mean()
        else:
            metrics_dict[f'{cat}_acc'] = np.nan # Handle empty classes safely

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["NonHAB", "HAB"], yticklabels=["NonHAB", "HAB"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({split_name})")
    plt.savefig(f"{output_dir}/cm_{split_name.lower()}.png")
    plt.close()
    
    return metrics_dict

def run_experiment(mode, run_id, weights_path, base_output_dir, train_loader, val_loader, test_loader, train_eval_loader, device):
    """Runs one specific configuration and returns its metrics."""
    output_dir = f"{base_output_dir}_{run_id}_{mode}"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    print(f"\n[{run_id}/10] Training Mode: {mode.upper()} -> Output: {output_dir}")

    model = HABLightningModel(mode=mode, lr=1e-4, weights_path=weights_path)
    logger = CSVLogger(output_dir, name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1", mode="max", save_top_k=1, save_last=True, 
        filename="best-hab-{epoch:02d}-{val_f1:.3f}"
    )
    early_stop_callback = EarlyStopping(monitor="val_f1", patience=20, mode="max", verbose=False)    

    # Hide progress bar to keep logs clean during 30 runs, but log normally
    trainer = L.Trainer(
        max_epochs=60, accelerator="gpu", devices=1, precision="32-true",
        logger=logger, callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=False 
    )
    trainer.fit(model, train_loader, val_loader)

    # Load best model for evaluation
    best_model = HABLightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.to(device)
    
    # Extract Metrics
    train_mets = evaluate_split(best_model, train_eval_loader, device, split_name="Train", output_dir=output_dir)
    val_mets = evaluate_split(best_model, val_loader, device, split_name="Validation", output_dir=output_dir)
    test_mets = evaluate_split(best_model, test_loader, device, split_name="Test", output_dir=output_dir)
    
    del model, best_model, trainer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    return train_mets, val_mets, test_mets

# --- MAIN AUTOMATION & META-ANALYSIS ---
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware initialized on: {DEVICE}")
    
    CSV_PATH = "/home/kostas/AMFITRITE/OWdata/amfitrite_open_waters_master.csv"
    DATA_ROOT = "/home/kostas/AMFITRITE/OWdata"
    BASE_OUTPUT_DIR = "/home/kostas/AMFITRITE/OW/res18/multiple_runs/results"          
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    TOTAL_RUNS = 10
    
    WEIGHTS = {
        'generic': None,
        's2': '/home/kostas/AMFITRITE/pretrained_model_weights/MoCo_ResNet18_S2-L1C_13_bands/B13_rn18_moco_0099_ckpt.pth',
        'iw': '/home/kostas/AMFITRITE/IW/res18_2classes/results12/hab_experiment/version_0/checkpoints/best-hab-epoch=26-val_f1=0.881.ckpt'
    }

    all_results = []
    modes_to_run = ['generic', 's2', 'iw']

    # --- 1. THE ITERATION LOOP ---
    for run_id in range(1, TOTAL_RUNS + 1):
        print("\n" + "="*60)
        print(f"STARTING STABILITY RUN {run_id}/{TOTAL_RUNS}")
        print("="*60)
        
        # New seed per run guarantees a new random split
        random_seed = 42 + run_id
        registry_path = f"./ow_summary_with_splits_run{run_id}.csv"
        
        df = get_or_create_split(CSV_PATH, DATA_ROOT, registry_path, random_seed)

        train_ds = HABDataset(df, DATA_ROOT, mode='training')
        val_ds = HABDataset(df, DATA_ROOT, mode='validation')
        test_ds = HABDataset(df, DATA_ROOT, mode='test')

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)
        train_eval_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)
        
        for current_mode in modes_to_run:
            train_mets, val_mets, test_mets = run_experiment(
                mode=current_mode, run_id=run_id, weights_path=WEIGHTS[current_mode],
                base_output_dir=BASE_OUTPUT_DIR, train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, train_eval_loader=train_eval_loader, device=DEVICE
            )
            
            # Package results for the DataFrame
            for mets in [train_mets, val_mets, test_mets]:
                mets['run_id'] = run_id
                mets['mode'] = current_mode
                all_results.append(mets)

    # --- 2. STATISTICAL META-ANALYSIS ---
    print("\n" + "="*60)
    print("ALL RUNS COMPLETE. CALCULATING STATISTICS...")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns for readability
    cols = ['mode', 'split', 'run_id', 'f1_macro', 'accuracy', 'hab_acc', 'nonhab_acc', 'land_acc', 'clouds_acc']
    results_df = results_df[cols]
    
    # 1. Mean and Std Dev for all metrics (Grouped by Mode and Split)
    numeric_cols = ['f1_macro', 'accuracy', 'hab_acc', 'nonhab_acc', 'land_acc', 'clouds_acc']
    summary_stats = results_df.groupby(['mode', 'split'])[numeric_cols].agg(['mean', 'std'])
    
    print("\n--- MEAN & STD DEVIATION ACROSS 10 SPLITS ---")
    print(summary_stats.to_string())
    
    # Save raw data and summary
    results_df.to_csv("./stability_raw_results.csv", index=False)
    summary_stats.to_csv("./stability_summary_stats.csv")
    
    # 2. Win Rate Analysis (Which model performs best on the TEST set per run?)
    test_df = results_df[results_df['split'] == 'Test']
    
    # Pivot to easily compare F1 scores across modes for the same run
    pivot_f1 = test_df.pivot(index='run_id', columns='mode', values='f1_macro')
    
    # Find which mode had the highest F1 in each row (run)
    winners = pivot_f1.idxmax(axis=1)
    win_counts = winners.value_counts()
    
    print("\n--- WIN RATE ANALYSIS (Highest Test F1 Score) ---")
    print(f"Total Runs Analyzed: {TOTAL_RUNS}")
    for mode in modes_to_run:
        wins = win_counts.get(mode, 0)
        win_rate = (wins / TOTAL_RUNS) * 100
        print(f"Mode '{mode.upper()}': Won {wins} times ({win_rate:.1f}% Win Rate)")
        
    print("\nAnalysis Saved: 'stability_raw_results.csv' and 'stability_summary_stats.csv'")