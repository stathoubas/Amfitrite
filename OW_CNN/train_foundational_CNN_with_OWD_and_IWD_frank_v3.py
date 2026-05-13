# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:05:28 2026

@author: K. Pikounis

train CNNs using both OW and IW datasest
fankenstein network resnet34: trained resnet 18 started from bigearth weights + trained resnet 34 from imagenet weightes 
freeze resnet 18 weights and then unfreeze them.

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
from sklearn.metrics import confusion_matrix, classification_report

from pytorch_lightning.callbacks import BaseFinetuning

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
# MODULE 2 changes: load an already sploit dataset and IMBALANCE CALCULATOR
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

class FrankFinetuningCallback(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # Freeze the entire network initially...
        self.freeze(pl_module.model)
        
        # ...Then UNFREEZE ONLY the deeper layers (Layer 3, Layer 4, and the Head)
        # This forces the R34 body to adapt to the frozen R18 stem!
        self.make_trainable(pl_module.model.layer3)
        self.make_trainable(pl_module.model.layer4)
        self.make_trainable(pl_module.model.fc)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When we hit the target epoch, THAW everything and drop the learning rate!
        if current_epoch == self.unfreeze_at_epoch:
            print(f"\n[Epoch {current_epoch}] Thawing the entire network for fine-tuning!")
            self.unfreeze_and_add_param_group(
                modules=pl_module.model,
                optimizer=optimizer,
                initial_denom_lr=10.0 # Divides the current learning rate by 10
            )

def build_water_cnn(architecture='resnet18', num_bands=10, mode='generic', weights_path=None, custom_weights_path=None):
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
        raise ValueError(f"Architecture {architecture} not supported. Use 'resnet18' or 'resnet34'.")

    # 2. Stem Surgery (Adjust input channels)
    model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 3. Load Pre-trained Weights based on Mode
    if mode == 'generic':
        print(f"[{architecture.upper()}] Mode: Generic - Inflating ImageNet weights to {num_bands} bands...")
        # Get standard ImageNet weights for the specific architecture
        temp_model = models.resnet18(weights=default_weights) if architecture == 'resnet18' else models.resnet34(weights=default_weights)
        
        with torch.no_grad():
            w_avg = temp_model.conv1.weight.mean(dim=1, keepdim=True)
            model.conv1.weight.copy_(w_avg.repeat(1, num_bands, 1, 1))
            
        state_dict = temp_model.state_dict()
        del state_dict['conv1.weight']
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False)

    elif mode == 's2':
        print(f"[{architecture.upper()}] Mode: S2 - Surgery on 13-band SSL4EO weights to {num_bands} bands...")
        state_dict = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('backbone.', '')
            if 'fc.' in name: continue 
            
            # Extract exactly the bands we need from the 13-band SSL4EO weight tensor
            if name == 'conv1.weight' and v.shape[1] == 13:
                if num_bands == 12:
                    keep_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
                elif num_bands == 10:
                    keep_indices = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
                v = v[:, keep_indices, :, :]
                
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)

    elif mode == 'bigearthnet':
        if num_bands != 10:
            raise ValueError("BigEarthNet strictly requires 10 bands.")
            
        print(f"[{architecture.upper()}] Mode: BigEarthNet - Loading 10-band weights...")
        if weights_path.endswith('.safetensors'):
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location='cpu')
            
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('backbone.', '').replace('model.vision_encoder.', '')
            if 'fc.' in name: continue
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)

    elif mode == 'frank_custom_r34':
        if architecture != 'resnet34' or num_bands != 10:
            raise ValueError("Frankenstein requires 'resnet34' and 10 bands.")
        if not custom_weights_path:
            raise ValueError("You must provide custom_weights_path for the trained R34 base.")
            
        print(f"[{architecture.upper()}] Mode: Frankenstein - Grafting BigEarthNet onto Custom Trained R34...")
        
        # A. Configure Head for 2 classes immediately
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        # B. Load Your Custom Trained R34 Weights (10 bands)
        custom_ckpt = torch.load(custom_weights_path, map_location='cpu')
        if 'state_dict' in custom_ckpt:
            # ONLY extract keys that actually belong to the ResNet (ignoring class_weights, etc.)
            custom_dict = {k.replace('model.', ''): v for k, v in custom_ckpt['state_dict'].items() if k.startswith('model.')}
        else:
            custom_dict = custom_ckpt
            
        model.load_state_dict(custom_dict, strict=False)
        print("-> Step 1: Loaded Custom R34 Base Weights.")
        
        # C. Load BigEarthNet R18 weights
        if weights_path.endswith('.safetensors'): state_dict = load_file(weights_path)
        else: state_dict = torch.load(weights_path, map_location='cpu')
            
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        
        bigearth_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('backbone.', '').replace('model.vision_encoder.', '').replace('model.', '')
            if 'fc.' in name: continue
            bigearth_dict[name] = v

        # D. SURGERY: Overwrite matching layers
        r34_dict = model.state_dict()
        overwritten_count = 0
        
        for name, r18_weight in bigearth_dict.items():
            if name in r34_dict and r34_dict[name].shape == r18_weight.shape:
                r34_dict[name] = r18_weight
                overwritten_count += 1
                
        model.load_state_dict(r34_dict)
        print(f"-> Step 2: Overwrote {overwritten_count} early layers with BigEarthNet weights.")
        
        # Return early because we already did the Head Surgery in Step A!
        return model

    # 4. Standard Head Surgery (Change to Binary Classification for non-Frankenstein models)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    return model

# ==============================================================================
# MODULE 5: ADVANCED PYTORCH LIGHTNING SYSTEM
# ==============================================================================

class HABLightningSystem(L.LightningModule):
    def __init__(self, architecture, num_bands, mode, weights_path, lr, class_weights, custom_weights_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        
        # 1. Build Model using Module 4
        self.model = build_water_cnn(
            architecture=architecture, 
            num_bands=num_bands, 
            mode=mode, 
            weights_path=weights_path,
            custom_weights_path=custom_weights_path
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
        
        # 4. Storage for granular sub-domain tracking
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
        active_parameters = filter(lambda p: p.requires_grad, self.parameters())        
        optimizer = torch.optim.AdamW(active_parameters, lr=self.hparams.lr)
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


def evaluate_split(model, loader, device, split_name, output_dir):
    """Runs final inference, prints classification report, and saves Confusion Matrix."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n--- Results for {split_name} ---")
    print(classification_report(all_labels, all_preds, target_names=["nonHAB", "HAB"], zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["nonHAB", "HAB"], yticklabels=["nonHAB", "HAB"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({split_name})")
    plt.savefig(os.path.join(output_dir, f"cm_{split_name.lower()}.png"))
    plt.close()

if __name__ == "__main__":
    # Define paths based on your inputs
    IW_DIR = "/home/kostas/AMFITRITE/data256"
    OW_DIR = "/home/kostas/AMFITRITE/OWdata"
    IW_EXCEL = "/home/kostas/AMFITRITE/dataset_summary_256x256pixels.xlsx"
    OW_CSV = "/home/kostas/AMFITRITE/OWdata/amfitrite_open_waters_master.csv"
    MASTER_OUTPUT = "/home/kostas/AMFITRITE/IW_and_OW_CNN/amfitrite_universal_split.csv"
    BASE_OUTPUT_DIR = "/home/kostas/AMFITRITE/IW_and_OW_CNN"
        
    # Ensure modules 1-4 are imported/defined above this!
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware initialized on: {DEVICE}")
    
    # --- GLOBAL CONFIGURATION ---
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    
    '''
    # Run Module 1
    harmonizer = DatasetHarmonizer(IW_DIR, OW_DIR, IW_EXCEL, OW_CSV)
    master_dataframe = harmonizer.create_master_registry()
    
    # Run Module 2
    split_df, class_weights = create_stratified_split_and_weights(master_dataframe, MASTER_OUTPUT)
    
     Define our experimental runs
    EXPERIMENTS = [
        {
            'exp_name': 'generic_resnet18',
            'architecture': 'resnet18',
            'num_bands': 12,
            'mode': 'generic',
            'weights_path': None
        },
        {
            'exp_name': 'generic_resnet34',
            'architecture': 'resnet34',
            'num_bands': 12,
            'mode': 'generic',
            'weights_path': None
        },
        {
            'exp_name': 's2_resnet18',
            'architecture': 'resnet18',
            'num_bands': 12,
            'mode': 's2',
            'weights_path': '/home/kostas/AMFITRITE/pretrained_model_weights/MoCo_ResNet18_S2-L1C_13_bands/B13_rn18_moco_0099_ckpt.pth'
        },
        {
            'exp_name': 'bigearthnet_resnet18',
            'architecture': 'resnet18',
            'num_bands': 10,
            'mode': 'bigearthnet',
            'weights_path': '/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0_resnet18-s2-v0.2.0/model.safetensors'
        }
    ]
    '''
    '''
    # Run load laready split dataste and extract weights 
    split_df, class_weights = load_existing_split_and_get_weights(MASTER_OUTPUT)
    
    EXPERIMENTS = [
        {
            'exp_name': 'generic_resnet34_10bands',
            'architecture': 'resnet34',
            'num_bands': 10,          # Configured strictly for 10 bands!
            'mode': 'generic',        # Inflates ImageNet weights to 10 bands
            'weights_path': None
        }
    ]
    '''
    # Run load laready split dataste and extract weights 
    split_df, class_weights = load_existing_split_and_get_weights(MASTER_OUTPUT)
    
    EXPERIMENTS = [
        {
            'exp_name': 'frank_v3_custom_resnet34_freeze_34_layers_first_try2',
            'architecture': 'resnet34',
            'num_bands': 10,
            'mode': 'frank_custom_r34',
            'weights_path': '/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/logs/version_0/checkpoints/best-hab-epoch=52-val_f1_macro=0.882.ckpt',
            'custom_weights_path': '/home/kostas/AMFITRITE/IW_and_OW_CNN/generic_resnet34_10bands/logs/version_0/checkpoints/best-hab-epoch=31-val_f1_macro=0.866.ckpt' 
        }
    ]
    
    
    # --- MAIN LOOP ---
    for exp in EXPERIMENTS:
        print("\n" + "="*60)
        print(f"STARTING EXPERIMENT: {exp['exp_name'].upper()}")
        print("="*60)
        
        output_dir = os.path.join(BASE_OUTPUT_DIR, exp['exp_name'])
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        # 1. Initialize Datasets Dynamically (num_bands determines the channel selection!)
        train_ds = UniversalWaterDataset(split_df, mode='training', num_bands=exp['num_bands'])
        val_ds   = UniversalWaterDataset(split_df, mode='validation', num_bands=exp['num_bands'])
        test_ds  = UniversalWaterDataset(split_df, mode='test', num_bands=exp['num_bands'])

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # 2. Initialize the Lightning Module
        model = HABLightningSystem(
            architecture=exp['architecture'],
            num_bands=exp['num_bands'],
            mode=exp['mode'],
            weights_path=exp['weights_path'],
            lr=1e-4,
            class_weights=class_weights,
            custom_weights_path=exp.get('custom_weights_path', None)
        )

        # 3. Callbacks & Logger
        logger = CSVLogger(output_dir, name="logs")
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_f1_macro", mode="max", save_top_k=10, save_last=True, 
            filename="best-hab-{epoch:02d}-{val_f1_macro:.3f}"
        )
        early_stop_callback = EarlyStopping(monitor="val_f1_macro", patience=20, mode="max", verbose=False)    
        
        finetune_cb = FrankFinetuningCallback(unfreeze_at_epoch=15) # Thaws at epoch 15

        # 4. Trainer Configuration
        trainer = L.Trainer(
            max_epochs=100, accelerator="gpu", devices=1, precision="32-true",
            logger=logger,  
            callbacks=[checkpoint_callback, early_stop_callback, finetune_cb], 
            enable_progress_bar=False
        )
        
        # 5. Execute Training
        trainer.fit(model, train_loader, val_loader)
        
        # 6. PLOTTING & FINAL EVALUATION
        print(f"\n[*] Generating Training Plots...")
        metrics_csv_path = os.path.join(logger.log_dir, "metrics.csv")
        plot_training_history(metrics_csv_path, output_dir)

        print(f"\n[*] Evaluating Best Model on all splits...")
        best_model = HABLightningSystem.load_from_checkpoint(checkpoint_callback.best_model_path, class_weights=class_weights)
        best_model.to(DEVICE)
        
        # Create a non-shuffled training loader so the confusion matrix is perfectly ordered
        train_eval_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        evaluate_split(best_model, train_eval_loader, DEVICE, "Train", output_dir)
        evaluate_split(best_model, val_loader, DEVICE, "Validation", output_dir)
        evaluate_split(best_model, test_loader, DEVICE, "Test", output_dir)      
        
        
        # Clean up Memory
        del model, trainer, train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*60)
