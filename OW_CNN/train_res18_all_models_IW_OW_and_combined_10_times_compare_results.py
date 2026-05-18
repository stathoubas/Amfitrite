# -*- coding: utf-8 -*-
"""
The "Master Ablation" Domain-Specific Fine-Tuning Pipeline
Target: Comprehensive Transfer Learning Evaluation (IW & OW)

Executes 10 Monte Carlo Iterations evaluating 7 distinct models per iteration:
1. Baseline OW (Trained strictly on OW)
2. Baseline IW (Trained on IW + clouds + land)
3. Universal Foundation (Trained on ALL data -> Zero-Shot evaluated on both)
4. Transfer to OW (Model 2 -> Fine-Tuned on OW)
5. FT Foundation OW (Model 3 -> Fine-Tuned on OW)
6. Transfer to IW (Model 1 -> Fine-Tuned on IW)
7. FT Foundation IW (Model 3 -> Fine-Tuned on IW)
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
        self.active_bands = [
            "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", 
            "B05_raw.tif", "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", 
            "B8A_raw.tif",                "B11_raw.tif", "B12_raw.tif"
        ]

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
# MODULE 4: MULTI-MODEL BUILDER
# ==============================================================================

def build_water_cnn(architecture='resnet18', num_bands=10, mode='generic', weights_path=None):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if mode == 'bigearthnet':
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

    elif mode == 'custom_r18':
        model.fc = nn.Linear(model.fc.in_features, 2)
        ckpt = torch.load(weights_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        clean_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_dict[k.replace('model.', '')] = v
        model.load_state_dict(clean_dict, strict=True)

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
        self.validation_step_outputs.append({
            'preds': preds.cpu(), 'targets': y.cpu(), 'groups': strat_groups
        })
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_overall", self.val_acc.compute())
        self.log("val_bal_acc", self.val_bal_acc.compute()) 
        if len(self.validation_step_outputs) == 0: return
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        all_groups = [g for x in self.validation_step_outputs for g in x['groups']]
        
        if (all_targets == 1).any():
            self.log("val_hab_acc", (all_preds[all_targets == 1] == all_targets[all_targets == 1]).float().mean())
        if (all_targets == 0).any():
            self.log("val_nonhab_acc", (all_preds[all_targets == 0] == all_targets[all_targets == 0]).float().mean())

        for group in ['iw_hab', 'iw_nonhab', 'ow_hab', 'ow_nonhab', 'land', 'clouds']:
            indices = [i for i, g in enumerate(all_groups) if g == group]
            if len(indices) > 0:
                self.log(f"val_{group}_acc", (all_preds[indices] == all_targets[indices]).float().mean())
            else:
                self.log(f"val_{group}_acc", 0.0) 

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
    metrics = [
        ('f1', 'F1 Score (Macro)'), ('acc', 'Overall Accuracy'), ('bal_acc', 'Balanced Accuracy'),
        ('hab_acc', 'Global HAB Acc'), ('nonhab_acc', 'Global non-HAB Acc'),
        ('iw_hab_acc', 'IW HABs Acc'), ('iw_nonhab_acc', 'IW non-HABs Acc'),
        ('ow_hab_acc', 'OW HABs Acc'), ('ow_nonhab_acc', 'OW non-HABs Acc'),
        ('land_acc', 'Land Acc'), ('clouds_acc', 'Clouds Acc')
    ]
    
    scenarios = [
        # OW Evaluations (Prefix: ow_)
        {'col_x': 'ow_baseline_', 'col_y': 'ow_zeroshot_', 'name_x': 'OW Baseline', 'name_y': 'Zero-Shot (All)', 'folder': 'duel_OW_1_Base_vs_Zero'},
        {'col_x': 'ow_baseline_', 'col_y': 'ow_ftfound_', 'name_x': 'OW Baseline', 'name_y': 'FT Foundation', 'folder': 'duel_OW_2_Base_vs_FTFound'},
        {'col_x': 'ow_baseline_', 'col_y': 'ow_transfer_from_iw_', 'name_x': 'OW Baseline', 'name_y': 'Transfer from IW', 'folder': 'duel_OW_3_Base_vs_TransferIW'},
        
        # IW Evaluations (Prefix: iw_)
        {'col_x': 'iw_baseline_', 'col_y': 'iw_zeroshot_', 'name_x': 'IW Baseline', 'name_y': 'Zero-Shot (All)', 'folder': 'duel_IW_1_Base_vs_Zero'},
        {'col_x': 'iw_baseline_', 'col_y': 'iw_ftfound_', 'name_x': 'IW Baseline', 'name_y': 'FT Foundation', 'folder': 'duel_IW_2_Base_vs_FTFound'},
        {'col_x': 'iw_baseline_', 'col_y': 'iw_transfer_from_ow_', 'name_x': 'IW Baseline', 'name_y': 'Transfer from OW', 'folder': 'duel_IW_3_Base_vs_TransferOW'}
    ]
    
    for sc in scenarios:
        duel_dir = os.path.join(base_output_dir, sc['folder'])
        os.makedirs(duel_dir, exist_ok=True)
        print(f"\n[*] Generating plots for: {sc['name_x']} vs {sc['name_y']}")
        
        for m_suffix, title in metrics:
            m_x, m_y = sc['col_x'] + m_suffix, sc['col_y'] + m_suffix
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
            plt.fill_between([min_val, max_val], [min_val, max_val], max_val, color='green', alpha=0.05, label=f"{sc['name_y']} Wins")
            plt.fill_between([min_val, max_val], min_val, [min_val, max_val], color='red', alpha=0.05, label=f"{sc['name_x']} Wins")
            plt.title(f"Monte Carlo Duel: {title}", fontsize=16, fontweight='bold')
            plt.xlabel(f"{sc['name_x']} Score", fontsize=14)
            plt.ylabel(f"{sc['name_y']} Score", fontsize=14)
            plt.grid(True, linestyle=':', alpha=0.7); plt.legend(loc='upper left', fontsize=12)
            plt.tight_layout(); plt.savefig(os.path.join(duel_dir, f"scatter_{m_suffix}.png"), dpi=300); plt.close()

            # Histogram
            pct_diff = 100 * (valid_df[m_y] - valid_df[m_x]) / valid_df[m_x]
            mean_diff = pct_diff.mean()
            plt.figure(figsize=(8, 6))
            sns.histplot(pct_diff, binwidth=0.1, kde=False, color='gray', edgecolor='black')
            plt.axvline(0, color='black', linestyle='--', linewidth=2, label="0% Difference")
            plt.axvline(mean_diff, color='blue', linestyle='-', linewidth=2, label=f"Mean: {mean_diff:+.2f}%")
            plt.title(f"Relative Improvement: {sc['name_y']} vs {sc['name_x']}\nMetric: {title}", fontsize=14, fontweight='bold')
            plt.xlabel(f"% Improvement over {sc['name_x']}", fontsize=12)
            plt.ylabel("Count (MC Iters)", fontsize=12)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7); plt.legend(loc='upper right', fontsize=12)
            plt.tight_layout(); plt.savefig(os.path.join(duel_dir, f"hist_{m_suffix}.png"), dpi=300); plt.close()

def evaluate_split(model, loader, device, split_name, prefix, iter_results_dict):
    """Evaluates and stores metrics dynamically using prefix (e.g., 'ow_baseline')."""
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
    
    for group in ['iw_hab', 'iw_nonhab', 'ow_hab', 'ow_nonhab', 'clouds', 'land']:
        group_data = results_df[results_df['Group'] == group]
        iter_results_dict[f'{prefix}_{group}_acc'] = group_data['Correct'].mean() if not group_data.empty else np.nan


# ==============================================================================
# MAIN EXECUTION (7 MODELS PER ITERATION)
# ==============================================================================

if __name__ == "__main__":
    IW_DIR = "/home/kostas/AMFITRITE/data256"
    OW_DIR = "/home/kostas/AMFITRITE/OWdata"
    IW_EXCEL = "/home/kostas/AMFITRITE/dataset_summary_256x256pixels.xlsx"
    OW_CSV = "/home/kostas/AMFITRITE/OWdata/amfitrite_open_waters_master.csv"
    
    BASE_OUTPUT_DIR = "/home/kostas/AMFITRITE/IW_and_OW_CNN/IW_OW_both_comparisons"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware initialized on: {DEVICE}")
    
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    TOTAL_ITERATIONS = 10
    
    BIGEARTH_WEIGHTS = '/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0_resnet18-s2-v0.2.0/model.safetensors'
    
    harmonizer = DatasetHarmonizer(IW_DIR, OW_DIR, IW_EXCEL, OW_CSV)
    master_dataframe = harmonizer.create_master_registry()
    
    START_ITERATION = 6
    backup_file = os.path.join(BASE_OUTPUT_DIR, "running_backup_results.csv")
    
    if os.path.exists(backup_file) and START_ITERATION > 1:
        print(f"\n[*] Resuming experiment. Loading past results from {backup_file}...")
        master_results = pd.read_csv(backup_file).to_dict('records')
    else:
        master_results = []

    for iteration in range(START_ITERATION, TOTAL_ITERATIONS + 1):
        print("\n" + "X"*70)
        print(f"XXX MASTER ABLATION ITERATION {iteration}/{TOTAL_ITERATIONS} XXX")
        print("X"*70)
        
        L.seed_everything(1234 + iteration, workers=True)
        iter_dir = os.path.join(BASE_OUTPUT_DIR, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # 1. Base Splitting
        master_split_csv = os.path.join(iter_dir, f"master_split_iter_{iteration}.csv")
        master_split_df = create_stratified_split(master_dataframe, master_split_csv, random_seed=42+iteration)
        master_weights = calculate_class_weights(master_split_df)

        # 2. Extract Subsets
        ow_df = master_split_df[master_split_df['source'] == 'OW'].reset_index(drop=True)
        ow_weights = calculate_class_weights(ow_df)

        iw_augmented_df = master_split_df[
            (master_split_df['source'] == 'IW') | (master_split_df['strat_group'].isin(['clouds', 'land']))
        ].reset_index(drop=True)
        iw_weights = calculate_class_weights(iw_augmented_df)

        # 3. DataLoaders
        ow_train_loader = torch.utils.data.DataLoader(UniversalWaterDataset(ow_df, 'training'), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        ow_val_loader   = torch.utils.data.DataLoader(UniversalWaterDataset(ow_df, 'validation'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        ow_test_loader  = torch.utils.data.DataLoader(UniversalWaterDataset(ow_df, 'test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        iw_train_loader = torch.utils.data.DataLoader(UniversalWaterDataset(iw_augmented_df, 'training'), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        iw_val_loader   = torch.utils.data.DataLoader(UniversalWaterDataset(iw_augmented_df, 'validation'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        iw_test_loader  = torch.utils.data.DataLoader(UniversalWaterDataset(iw_augmented_df, 'test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        master_train_loader = torch.utils.data.DataLoader(UniversalWaterDataset(master_split_df, 'training'), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        master_val_loader   = torch.utils.data.DataLoader(UniversalWaterDataset(master_split_df, 'validation'), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        iter_results = {'iteration': iteration}

        # --- MODEL 1: OW Baseline ---
        print("\n[Phase 1] Training Model 1: Baseline OW...")
        m1 = HABLightningSystem('resnet18', 10, 'bigearthnet', BIGEARTH_WEIGHTS, 1e-4, ow_weights)
        cb1 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1)
        t1 = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="log_m1_ow_base"), callbacks=[cb1, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        t1.fit(m1, ow_train_loader, ow_val_loader)
        ckpt_m1 = cb1.best_model_path
        best_m1 = HABLightningSystem.load_from_checkpoint(ckpt_m1, class_weights=ow_weights).to(DEVICE)
        evaluate_split(best_m1, ow_test_loader, DEVICE, "M1_OW_Base", "ow_baseline", iter_results)
        del m1, t1, best_m1; gc.collect(); torch.cuda.empty_cache()

        # --- MODEL 2: IW Baseline ---
        print("\n[Phase 2] Training Model 2: Baseline IW...")
        m2 = HABLightningSystem('resnet18', 10, 'bigearthnet', BIGEARTH_WEIGHTS, 1e-4, iw_weights)
        cb2 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1)
        t2 = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="log_m2_iw_base"), callbacks=[cb2, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        t2.fit(m2, iw_train_loader, iw_val_loader)
        ckpt_m2 = cb2.best_model_path
        best_m2 = HABLightningSystem.load_from_checkpoint(ckpt_m2, class_weights=iw_weights).to(DEVICE)
        evaluate_split(best_m2, iw_test_loader, DEVICE, "M2_IW_Base", "iw_baseline", iter_results)
        del m2, t2, best_m2; gc.collect(); torch.cuda.empty_cache()

        # --- MODEL 3: Universal Foundation (Zero-Shot) ---
        print("\n[Phase 3] Training Model 3: Universal Foundation (Master)...")
        m3 = HABLightningSystem('resnet18', 10, 'bigearthnet', BIGEARTH_WEIGHTS, 1e-4, master_weights)
        cb3 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1)
        t3 = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="log_m3_foundation"), callbacks=[cb3, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        t3.fit(m3, master_train_loader, master_val_loader)
        ckpt_m3 = cb3.best_model_path
        
        # Eval 3a: Zero-Shot OW
        best_m3_ow = HABLightningSystem.load_from_checkpoint(ckpt_m3, class_weights=ow_weights).to(DEVICE)
        evaluate_split(best_m3_ow, ow_test_loader, DEVICE, "M3_OW_ZeroShot", "ow_zeroshot", iter_results)
        # Eval 3b: Zero-Shot IW
        best_m3_iw = HABLightningSystem.load_from_checkpoint(ckpt_m3, class_weights=iw_weights).to(DEVICE)
        evaluate_split(best_m3_iw, iw_test_loader, DEVICE, "M3_IW_ZeroShot", "iw_zeroshot", iter_results)
        del m3, t3, best_m3_ow, best_m3_iw; gc.collect(); torch.cuda.empty_cache()

        # --- MODEL 4: Transfer IW -> OW ---
        print("\n[Phase 4] Training Model 4: Transfer (IW Model -> FT on OW)...")
        m4 = HABLightningSystem('resnet18', 10, 'custom_r18', ckpt_m2, 1e-5, ow_weights)
        cb4 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1)
        t4 = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="log_m4_transfer_to_ow"), callbacks=[cb4, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        t4.fit(m4, ow_train_loader, ow_val_loader)
        best_m4 = HABLightningSystem.load_from_checkpoint(cb4.best_model_path, class_weights=ow_weights).to(DEVICE)
        evaluate_split(best_m4, ow_test_loader, DEVICE, "M4_OW_Transfer", "ow_transfer_from_iw", iter_results)
        del m4, t4, best_m4; gc.collect(); torch.cuda.empty_cache()

        # --- MODEL 5: FT Foundation -> OW ---
        print("\n[Phase 5] Training Model 5: FT Foundation -> OW...")
        m5 = HABLightningSystem('resnet18', 10, 'custom_r18', ckpt_m3, 1e-5, ow_weights)
        cb5 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1)
        t5 = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="log_m5_ftfound_ow"), callbacks=[cb5, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        t5.fit(m5, ow_train_loader, ow_val_loader)
        best_m5 = HABLightningSystem.load_from_checkpoint(cb5.best_model_path, class_weights=ow_weights).to(DEVICE)
        evaluate_split(best_m5, ow_test_loader, DEVICE, "M5_OW_FTFound", "ow_ftfound", iter_results)
        del m5, t5, best_m5; gc.collect(); torch.cuda.empty_cache()

        # --- MODEL 6: Transfer OW -> IW ---
        print("\n[Phase 6] Training Model 6: Transfer (OW Model -> FT on IW)...")
        m6 = HABLightningSystem('resnet18', 10, 'custom_r18', ckpt_m1, 1e-5, iw_weights)
        cb6 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1)
        t6 = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="log_m6_transfer_to_iw"), callbacks=[cb6, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        t6.fit(m6, iw_train_loader, iw_val_loader)
        best_m6 = HABLightningSystem.load_from_checkpoint(cb6.best_model_path, class_weights=iw_weights).to(DEVICE)
        evaluate_split(best_m6, iw_test_loader, DEVICE, "M6_IW_Transfer", "iw_transfer_from_ow", iter_results)
        del m6, t6, best_m6; gc.collect(); torch.cuda.empty_cache()

        # --- MODEL 7: FT Foundation -> IW ---
        print("\n[Phase 7] Training Model 7: FT Foundation -> IW...")
        m7 = HABLightningSystem('resnet18', 10, 'custom_r18', ckpt_m3, 1e-5, iw_weights)
        cb7 = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1)
        t7 = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="log_m7_ftfound_iw"), callbacks=[cb7, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        t7.fit(m7, iw_train_loader, iw_val_loader)
        best_m7 = HABLightningSystem.load_from_checkpoint(cb7.best_model_path, class_weights=iw_weights).to(DEVICE)
        evaluate_split(best_m7, iw_test_loader, DEVICE, "M7_IW_FTFound", "iw_ftfound", iter_results)
        del m7, t7, best_m7; gc.collect(); torch.cuda.empty_cache()

        # --- End Iteration Logic ---
        master_results.append(iter_results)
        pd.DataFrame(master_results).to_csv(os.path.join(BASE_OUTPUT_DIR, "running_backup_results.csv"), index=False)
        print(f"\n[Iteration {iteration} Complete] Matrix Backed Up.")
        del ow_train_loader, ow_val_loader, ow_test_loader, iw_train_loader, iw_val_loader, iw_test_loader, master_train_loader, master_val_loader; gc.collect()

    print("\n" + "="*60)
    print("ALL 10 ITERATIONS COMPLETED!")
    print("="*60)
    
    final_csv_path = os.path.join(BASE_OUTPUT_DIR, "final_master_ablation_results.csv")
    pd.DataFrame(master_results).to_csv(final_csv_path, index=False)
    generate_comparison_plots(final_csv_path, BASE_OUTPUT_DIR)