# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:41:05 2026

@author: K. Pikounis

Created for Amfitrite Open Waters
Training Script (Generic, S2, or Inland Water Pretrained)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torchinfo import summary
from torchview import draw_graph

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
        if random.random() > 0.5:
            tensor = TF.hflip(tensor)
        if random.random() > 0.5:
            tensor = TF.vflip(tensor)
            
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            tensor = TF.rotate(tensor, angle)
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row['ID']) # Assuming ID is the base of the folder name
        
        # Determine Folder Name suffix based on strat_category
        folder_suffix = "HAB" if row['tile_is_hab'] else "nonHAB"
        if row['tile_status'] == "land": folder_suffix = "land"
        if row['tile_status'] == "clouds": folder_suffix = "clouds"
        
        folder_path = os.path.join(self.root_dir, f"{uid}_{folder_suffix}")
        
        # Binary Label for CNN (1 = HAB, 0 = Non-HAB / Land / Cloud)
        label = 1 if row['tile_is_hab'] else 0
        
        # Store category for tracking metrics later
        strat_cat = row['strat_category'] 
        
        band_data = []
        for b_name in self.band_names:
            with rasterio.open(os.path.join(folder_path, b_name)) as src:
                band_data.append(src.read(1).astype(np.float32))
        
        bands_stack = np.stack(band_data, axis=0)
        tensor = torch.from_numpy(bands_stack)
        
        # Normalization
        tensor = tensor / 10000.0
        
        if self.mode == 'training':
            tensor = self.apply_augmentations(tensor)
            
        # Return tensor, label, and the text category for analysis
        return tensor, label, strat_cat

class HABLightningModel(L.LightningModule):
    def __init__(self, mode='generic', weights_path=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = self._build_model(self.hparams.mode, self.hparams.weights_path)
        
        # Adjust weights if HAB vs NonHAB balance is different in OW dataset
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
        
        # Stem Surgery
        model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if mode == 'generic':
            print("Mode: Generic - Inflating ImageNet weights...")
            temp_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            with torch.no_grad():
                w_avg = temp_resnet.conv1.weight.mean(dim=1, keepdim=True)
                model.conv1.weight.copy_(w_avg.repeat(1, 12, 1, 1))
            state_dict = temp_resnet.state_dict()
            del state_dict['conv1.weight']
            model.load_state_dict(state_dict, strict=False)
                
        elif mode == 's2':
            print(f"Mode: S2 - Loading Sentinel-2 SSL4EO weights...")
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '').replace('backbone.', '')
                if name == 'conv1.weight' and v.shape[1] == 13:
                    keep_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
                    v = v[:, keep_indices, :, :]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            
        elif mode == 'iw':
            print(f"Mode: Inland Waters (IW) - Loading pretrained Lightning weights...")
            # Lightning saves the entire module state_dict, we need to extract 'model.'
            ckpt = torch.load(weights_path, map_location='cpu')
            state_dict = ckpt['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove the "model." prefix Lightning adds
                if k.startswith('model.'):
                    new_state_dict[k.replace('model.', '')] = v
            model.load_state_dict(new_state_dict, strict=False)

        # Head Surgery
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        # If IW mode, load the head weights as well
        if mode == 'iw':
            model.fc.weight.data.copy_(new_state_dict['fc.weight'])
            model.fc.bias.data.copy_(new_state_dict['fc.bias'])
            
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch # Ignore text category during pure training
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_metrics.update(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y, strat_cats = batch
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"}}


def prepare_dataset_registry_and_split(excel_path, data_root, output_registry_path):
    print("Step 1: Loading master Excel and creating stratification categories...")
    df = pd.read_csv(excel_path) # Changed to read_csv based on master.csv standard
    df['split'] = 'junk'
    
    # Stratification Logic
    def get_strat_cat(row):
        if row['tile_is_hab'] == True:
            return 'hab'
        elif row['tile_status'] == 'land':
            return 'land'
        elif row['tile_status'] == 'clouds':
            return 'clouds'
        else:
            return 'nonhab'
            
    df['strat_category'] = df.apply(get_strat_cat, axis=1)
    
    # Filter physical existence
    valid_indices = []
    for idx, row in df.iterrows():
        uid = str(row['ID'])
        folder_suffix = "HAB" if row['tile_is_hab'] else "nonHAB"
        if row['tile_status'] == "land": folder_suffix = "land"
        if row['tile_status'] == "clouds": folder_suffix = "clouds"
        
        folder_path = os.path.join(data_root, f"{uid}_{folder_suffix}")
        if os.path.isdir(folder_path):
            valid_indices.append(idx)

    print(f"  -> Found {len(valid_indices)} valid folders.")
    valid_df = df.loc[valid_indices].copy()
    
    # 70/15/15 Stratified Split based on 4 categories
    train_idx, temp_idx = train_test_split(
        valid_df.index, test_size=0.30, stratify=valid_df['strat_category'], random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=valid_df.loc[temp_idx, 'strat_category'], random_state=42
    )
    
    df.loc[train_idx, 'split'] = 'training'
    df.loc[val_idx, 'split'] = 'validation'
    df.loc[test_idx, 'split'] = 'test'
    
    df.to_csv(output_registry_path, index=False)
    print("Split counts per category (Training):")
    print(df[df['split'] == 'training']['strat_category'].value_counts())
    return df


def evaluate_split(model, loader, device, split_name="Test", output_dir="plots"):
    model.eval()
    all_preds = []
    all_labels = []
    all_cats = []
    
    print(f"\n--- Evaluating on {split_name} Set ---")
    with torch.no_grad():
        for images, labels, cats in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_cats.extend(cats)
            
    # Standard Classification Report
    print(classification_report(all_labels, all_preds, target_names=["NonHAB", "HAB"]))
    
    # Custom 4-Category Breakdown
    print("\n--- Breakdown by Stratification Category ---")
    results = pd.DataFrame({'Actual': all_labels, 'Predicted': all_preds, 'Category': all_cats})
    results['Correct'] = results['Actual'] == results['Predicted']
    
    for cat in ["hab", "nonhab", "land", "clouds"]:
        cat_data = results[results['Category'] == cat]
        if not cat_data.empty:
            acc = cat_data['Correct'].mean() * 100
            print(f"{cat.upper():>8} Accuracy: {acc:.2f}% ({cat_data['Correct'].sum()}/{len(cat_data)})")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["NonHAB", "HAB"], yticklabels=["NonHAB", "HAB"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({split_name})")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/cm_{split_name.lower()}.png")
    plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. SETUP DATA ---
    CSV_PATH = "/home/kostas/AMFITRITE/OWdata/amfitrite_open_waters_master.csv"
    DATA_ROOT = "/home/kostas/AMFITRITE/OWdata"
    registry_path = "/home/kostas/AMFITRITE/OWdata/ow_summary_with_splits.csv"
    output_path = "/home/kostas/AMFITRITE/OW/res18/results1"
    batch_size = 64
    num_workers = 8

    df = prepare_dataset_registry_and_split(CSV_PATH, DATA_ROOT, registry_path)

    train_ds = HABDataset(df, DATA_ROOT, mode='training')
    val_ds = HABDataset(df, DATA_ROOT, mode='validation')
    test_ds = HABDataset(df, DATA_ROOT, mode='test')

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    # --- 2. SETUP MODEL ---
    # mode can be 'generic', 's2', or 'iw'
    # For IW, point to the .ckpt file generated from your previous run
    model = HABLightningModel(
        mode='iw', 
        lr=1e-4, 
        weights_path='./best-hab-epoch=21-val_f1=0.851.ckpt'
    )

    logger = CSVLogger(output_path, name="hab_ow_experiment")
    checkpoint_callback = ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=3, save_last=True, filename="best-hab-{epoch:02d}-{val_f1:.3f}")
    early_stop_callback = EarlyStopping(monitor="val_f1", patience=20, mode="max", verbose=True)    

    trainer = L.Trainer(
        max_epochs=60,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    
    print("Starting Training...")
    trainer.fit(model, train_loader, val_loader)

    # --- 3. ANALYSIS ---
    print(f"\nLoading Best Model from: {checkpoint_callback.best_model_path}")
    best_model = HABLightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.to(DEVICE)
    
    evaluate_split(best_model, val_loader, DEVICE, split_name="Validation", output_dir=output_path)
    evaluate_split(best_model, test_loader, DEVICE, split_name="Test", output_dir=output_path)