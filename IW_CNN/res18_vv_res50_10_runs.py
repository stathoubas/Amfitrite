# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:54:00 2026

@author: K. Pikounis

Architectural Capacity Duel: ResNet-18 vs ResNet-50

Runs 10 Monte Carlo Iterations evaluating 2 architectures:
1. ResNet-18 (Starting from BigEarthNet R18 weights)
2. ResNet-50 (Starting from BigEarthNet R50 weights)
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
                binary_label = 1; strat_group = 'iw_hab'
            else:
                binary_label = 0; strat_group = 'iw_nonhab'
                
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
            if row['tile_is_hab'] == True: suffix, binary_label, strat_group = "HAB", 1, "ow_hab"
            else:
                binary_label = 0
                if row['tile_status'] == 'land': suffix, strat_group = "land", "land"
                elif row['tile_status'] == 'clouds': suffix, strat_group = "clouds", "clouds"
                else: suffix, strat_group = "nonHAB", "ow_nonhab"
                    
            folder_path = os.path.join(self.ow_dir, f"{uid}_{suffix}")
            if os.path.isdir(folder_path):
                standardized_rows.append({
                    'uid': uid, 'folder_path': folder_path,
                    'binary_label': binary_label, 'strat_group': strat_group, 'source': 'OW'
                })
        return pd.DataFrame(standardized_rows)

    def create_master_registry(self):
        return pd.concat([self.process_iw_data(), self.process_ow_data()], ignore_index=True)

# ==============================================================================
# MODULE 2: SPLITTER & IMBALANCE CALCULATOR
# ==============================================================================

def create_stratified_split(master_df, output_registry_path, random_seed=42):
    print("\nPerforming 70/15/15 Stratified Split on Master Dataset...")
    df = master_df.copy()
    df['split'] = 'junk'
    
    train_idx, temp_idx = train_test_split(df.index, test_size=0.30, stratify=df['strat_group'], random_state=random_seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=df.loc[temp_idx, 'strat_group'], random_state=random_seed)
    
    df.loc[train_idx, 'split'] = 'training'
    df.loc[val_idx, 'split'] = 'validation'
    df.loc[test_idx, 'split'] = 'test'
    df.to_csv(output_registry_path, index=False)
    return df

def calculate_class_weights(df):
    train_df = df[df['split'] == 'training']
    count_nonhab = len(train_df[train_df['binary_label'] == 0])
    count_hab = len(train_df[train_df['binary_label'] == 1])
    total = count_nonhab + count_hab
    
    weight_nonhab = total / (2.0 * count_nonhab) if count_nonhab > 0 else 1.0
    weight_hab = total / (2.0 * count_hab) if count_hab > 0 else 1.0
    return torch.tensor([weight_nonhab, weight_hab], dtype=torch.float32)

# ==============================================================================
# MODULE 3: UNIVERSAL PYTORCH DATASET
# ==============================================================================

class UniversalWaterDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, mode='training', num_bands=10):
        self.df = dataframe[dataframe['split'] == mode].reset_index(drop=True)
        self.mode = mode
        self.num_bands = num_bands
        self.active_bands = ["B02_raw.tif", "B03_raw.tif", "B04_raw.tif", "B05_raw.tif", "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", "B8A_raw.tif", "B11_raw.tif", "B12_raw.tif"]

    def __len__(self): return len(self.df)

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
        if self.mode == 'training': tensor = self.apply_augmentations(tensor)
        return tensor, label, strat_group

# ==============================================================================
# MODULE 4: DYNAMIC MODEL BUILDER
# ==============================================================================

def build_water_resnet(architecture='resnet18', num_bands=10, weights_path=None):
    if architecture == 'resnet18':
        print("[MODEL] Building ResNet-18...")
        model = models.resnet18(weights=None)
    elif architecture == 'resnet50':
        print("[MODEL] Building ResNet-50...")
        model = models.resnet50(weights=None)
    else:
        raise ValueError("Unsupported architecture")

    model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)

    print(f"[MODEL] Loading BigEarthNet safetensors from: {weights_path}")
    if weights_path.endswith('.safetensors'): state_dict = load_file(weights_path)
    else: state_dict = torch.load(weights_path, map_location='cpu')
    
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    elif 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
    
    new_state_dict = {
        k.replace('module.', '').replace('backbone.', '').replace('model.vision_encoder.', ''): v 
        for k, v in state_dict.items() if not 'fc.' in k
    }
    model.load_state_dict(new_state_dict, strict=False)
    
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# ==============================================================================
# MODULE 5: ADVANCED PYTORCH LIGHTNING SYSTEM
# ==============================================================================

class HABLightningSystem(L.LightningModule):
    def __init__(self, architecture, num_bands, weights_path, lr, class_weights):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.model = build_water_resnet(architecture, num_bands, weights_path)
        self.register_buffer("class_weights", class_weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        self.train_f1 = MulticlassF1Score(num_classes=2, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=2, average='macro')
        self.train_acc = MulticlassAccuracy(num_classes=2, average='micro')
        self.val_acc = MulticlassAccuracy(num_classes=2, average='micro')
        self.train_bal_acc = MulticlassAccuracy(num_classes=2, average='macro')
        self.val_bal_acc = MulticlassAccuracy(num_classes=2, average='macro')
        self.validation_step_outputs = []

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
        x, y, strat_groups = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_f1(preds, y); self.val_acc(preds, y); self.val_bal_acc(preds, y) 
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_f1_macro", self.val_f1, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append({'preds': preds.cpu(), 'targets': y.cpu(), 'groups': strat_groups})
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_overall", self.val_acc.compute())
        self.log("val_bal_acc", self.val_bal_acc.compute()) 
        if len(self.validation_step_outputs) == 0: return
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        all_groups = [g for x in self.validation_step_outputs for g in x['groups']]
        
        if (all_targets == 1).any(): self.log("val_hab_acc", (all_preds[all_targets == 1] == all_targets[all_targets == 1]).float().mean())
        if (all_targets == 0).any(): self.log("val_nonhab_acc", (all_preds[all_targets == 0] == all_targets[all_targets == 0]).float().mean())

        for group in ['iw_hab', 'iw_nonhab', 'ow_hab', 'ow_nonhab', 'land', 'clouds']:
            indices = [i for i, g in enumerate(all_groups) if g == group]
            if len(indices) > 0: self.log(f"val_{group}_acc", (all_preds[indices] == all_targets[indices]).float().mean())
            else: self.log(f"val_{group}_acc", 0.0) 

        self.val_f1.reset(); self.val_acc.reset(); self.val_bal_acc.reset()
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1_macro"}}

# ==============================================================================
# MODULE 6: PLOTTING & EVALUATION
# ==============================================================================

def generate_comparison_plots(csv_path, base_output_dir):
    df = pd.read_csv(csv_path)
    
    # Removed clouds/land to prevent plotting empty metric datasets
    metrics = [
        ('f1', 'F1 Score (Macro)'), ('acc', 'Overall Accuracy'), ('bal_acc', 'Balanced Accuracy'),
        ('hab_acc', 'Global HAB Acc'), ('nonhab_acc', 'Global non-HAB Acc'),
        ('iw_hab_acc', 'IW HABs Acc'), ('iw_nonhab_acc', 'IW non-HABs Acc')
    ]
    
    duel_dir = os.path.join(base_output_dir, "duel_ResNet18_vs_ResNet50")
    os.makedirs(duel_dir, exist_ok=True)
    print(f"\n[*] Generating plots for: ResNet-18 vs ResNet-50")
    
    col_x = 'res18_'
    col_y = 'res50_'
    
    for m_suffix, title in metrics:
        m_x, m_y = col_x + m_suffix, col_y + m_suffix
        if m_x not in df.columns or m_y not in df.columns: continue
        valid_df = df.dropna(subset=[m_x, m_y])
        if len(valid_df) == 0: continue
            
        # Scatter Plot
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=valid_df[m_x], y=valid_df[m_y], s=150, color='blue', edgecolor='black', alpha=0.8)
        min_val = min(valid_df[m_x].min(), valid_df[m_y].min()) - 0.02
        max_val = max(valid_df[m_x].max(), valid_df[m_y].max()) + 0.02
        plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label="Parity (y = x)")
        plt.fill_between([min_val, max_val], [min_val, max_val], max_val, color='green', alpha=0.05, label="ResNet-50 Wins")
        plt.fill_between([min_val, max_val], min_val, [min_val, max_val], color='red', alpha=0.05, label="ResNet-18 Wins")
        plt.title(f"Monte Carlo Duel: {title}", fontsize=16, fontweight='bold')
        plt.xlabel("ResNet-18 Score", fontsize=14)
        plt.ylabel("ResNet-50 Score", fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.7); plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout(); plt.savefig(os.path.join(duel_dir, f"scatter_{m_suffix}.png"), dpi=300); plt.close()

        # Histogram
        pct_diff = 100 * (valid_df[m_y] - valid_df[m_x]) / valid_df[m_x]
        mean_diff = pct_diff.mean()
        plt.figure(figsize=(8, 6))
        sns.histplot(pct_diff, binwidth=0.1, kde=False, color='gray', edgecolor='black')
        plt.axvline(0, color='black', linestyle='--', linewidth=2, label="0% Difference")
        plt.axvline(mean_diff, color='blue', linestyle='-', linewidth=2, label=f"Mean: {mean_diff:+.2f}%")
        plt.title(f"Relative Improvement: ResNet-50 vs ResNet-18\nMetric: {title}", fontsize=14, fontweight='bold')
        plt.xlabel("% Improvement over ResNet-18", fontsize=12)
        plt.ylabel("Count (MC Iters)", fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7); plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout(); plt.savefig(os.path.join(duel_dir, f"hist_{m_suffix}.png"), dpi=300); plt.close()

def evaluate_split(model, loader, device, split_name, prefix, iter_results_dict):
    """Evaluates and stores metrics dynamically using prefix."""
    model.eval()
    all_preds, all_labels, all_groups = [], [], []

    with torch.no_grad():
        for images, labels, groups in loader:
            images = images.to(device)
            preds = torch.argmax(model(images), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_groups.extend(groups)

    results_df = pd.DataFrame({'Actual': all_labels, 'Predicted': all_preds, 'Group': all_groups})
    results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

    iter_results_dict[f'{prefix}_f1'] = f1_score(all_labels, all_preds, average='macro')
    iter_results_dict[f'{prefix}_acc'] = accuracy_score(all_labels, all_preds)
    iter_results_dict[f'{prefix}_bal_acc'] = balanced_accuracy_score(all_labels, all_preds)
    
    hab_data = results_df[results_df['Actual'] == 1]
    nonhab_data = results_df[results_df['Actual'] == 0]
    iter_results_dict[f'{prefix}_hab_acc'] = hab_data['Correct'].mean() if not hab_data.empty else np.nan
    iter_results_dict[f'{prefix}_nonhab_acc'] = nonhab_data['Correct'].mean() if not nonhab_data.empty else np.nan
    
    # Still calculates granular metrics if they exist safely; missing ones return NaN
    for group in ['iw_hab', 'iw_nonhab', 'ow_hab', 'ow_nonhab', 'clouds', 'land']:
        group_data = results_df[results_df['Group'] == group]
        iter_results_dict[f'{prefix}_{group}_acc'] = group_data['Correct'].mean() if not group_data.empty else np.nan


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    IW_DIR = "/home/kostas/AMFITRITE/data256"
    OW_DIR = "/home/kostas/AMFITRITE/OWdata"
    IW_EXCEL = "/home/kostas/AMFITRITE/dataset_summary_256x256pixels.xlsx"
    OW_CSV = "/home/kostas/AMFITRITE/OWdata/amfitrite_open_waters_master.csv"
    
    BASE_OUTPUT_DIR = "/home/kostas/AMFITRITE/IW/res18_vs_res50_2classes"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware initialized on: {DEVICE}")
    
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    TOTAL_ITERATIONS = 10
    
    WEIGHTS_R18 = '/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0_resnet18-s2-v0.2.0/model.safetensors'
    WEIGHTS_R50 = '/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0_resnet50-s2-v0.2.0/model.safetensors'
    
    harmonizer = DatasetHarmonizer(IW_DIR, OW_DIR, IW_EXCEL, OW_CSV)
    master_dataframe = harmonizer.create_master_registry()
    
    # --- RESUME LOGIC ---
    START_ITERATION = 1  # Change this to > 1 to safely resume without losing data
    backup_file = os.path.join(BASE_OUTPUT_DIR, "running_backup_results.csv")
    
    if os.path.exists(backup_file) and START_ITERATION > 1:
        print(f"\n[*] Resuming experiment. Loading past results from {backup_file}...")
        master_results = pd.read_csv(backup_file).to_dict('records')
    else:
        master_results = []

    for iteration in range(START_ITERATION, TOTAL_ITERATIONS + 1):
        print("\n" + "X"*70)
        print(f"XXX RESNET DUEL ITERATION {iteration}/{TOTAL_ITERATIONS} XXX")
        print("X"*70)
        
        L.seed_everything(1234 + iteration, workers=True)
        iter_dir = os.path.join(BASE_OUTPUT_DIR, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Split Master FIRST (to keep the exact same random seeds as previous experiments!)
        master_split_csv = os.path.join(iter_dir, f"master_split_iter_{iteration}.csv")
        master_split_df = create_stratified_split(master_dataframe, master_split_csv, random_seed=42+iteration)
        
        # CHRONOLOGICAL FIX: Extract STRICTLY the Inland Water (IW) data. No OW clouds/land.
        iw_strict_df = master_split_df[
            master_split_df['source'] == 'IW'
        ].reset_index(drop=True)
        iw_weights = calculate_class_weights(iw_strict_df)

        # DataLoaders strictly on pure IW
        iw_train_loader = torch.utils.data.DataLoader(UniversalWaterDataset(iw_strict_df, 'training'), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        iw_val_loader   = torch.utils.data.DataLoader(UniversalWaterDataset(iw_strict_df, 'validation'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        iw_test_loader  = torch.utils.data.DataLoader(UniversalWaterDataset(iw_strict_df, 'test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        iter_results = {'iteration': iteration}

        # --- MODEL 1: ResNet-18 ---
        print("\n[Phase 1] Training ResNet-18 strictly on IW...")
        m_r18 = HABLightningSystem('resnet18', 10, WEIGHTS_R18, 1e-4, iw_weights)
        cb_r18 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1)
        t_r18 = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="log_r18"), callbacks=[cb_r18, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        t_r18.fit(m_r18, iw_train_loader, iw_val_loader)
        
        best_r18 = HABLightningSystem.load_from_checkpoint(cb_r18.best_model_path, class_weights=iw_weights).to(DEVICE)
        evaluate_split(best_r18, iw_test_loader, DEVICE, "ResNet-18", "res18", iter_results)
        del m_r18, t_r18, best_r18; gc.collect(); torch.cuda.empty_cache()

        # --- MODEL 2: ResNet-50 ---
        print("\n[Phase 2] Training ResNet-50 strictly on IW...")
        m_r50 = HABLightningSystem('resnet50', 10, WEIGHTS_R50, 1e-4, iw_weights)
        cb_r50 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1)
        t_r50 = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="log_r50"), callbacks=[cb_r50, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        t_r50.fit(m_r50, iw_train_loader, iw_val_loader)
        
        best_r50 = HABLightningSystem.load_from_checkpoint(cb_r50.best_model_path, class_weights=iw_weights).to(DEVICE)
        evaluate_split(best_r50, iw_test_loader, DEVICE, "ResNet-50", "res50", iter_results)
        del m_r50, t_r50, best_r50; gc.collect(); torch.cuda.empty_cache()

        # --- End Iteration Logic ---
        master_results.append(iter_results)
        pd.DataFrame(master_results).to_csv(os.path.join(BASE_OUTPUT_DIR, "running_backup_results.csv"), index=False)
        print(f"\n[Iteration {iteration} Complete] Results Backed Up.")
        del iw_train_loader, iw_val_loader, iw_test_loader; gc.collect()

    print("\n" + "="*60)
    print("ALL 10 ITERATIONS COMPLETED!")
    print("="*60)
    
    final_csv_path = os.path.join(BASE_OUTPUT_DIR, "final_resnet_duel_results.csv")
    pd.DataFrame(master_results).to_csv(final_csv_path, index=False)
    generate_comparison_plots(final_csv_path, BASE_OUTPUT_DIR)