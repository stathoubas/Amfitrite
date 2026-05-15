# -*- coding: utf-8 -*-
"""
The "Gold Standard" Domain-Specific Fine-Tuning Pipeline
Target: Inland Water (Augmented with Land/Clouds)

Runs 10 Monte Carlo Iterations evaluating 3 ResNet-18 variants:
1. Baseline: BigEarthNet -> Trained on IW
2. Zero-Shot: BigEarthNet -> Trained on Master (Universal) -> Evaluated on IW
3. Fine-Tuned: Universal Foundation -> Fine-Tuned on IW
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

def create_stratified_split(master_df, output_registry_path, random_seed=42):
    """Splits the Master dataframe and returns it. Weights are calculated separately later."""
    print("\nPerforming 70/15/15 Stratified Split on Master Dataset...")
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
    
    return df

def calculate_class_weights(df):
    """Calculates PyTorch class weights strictly from the Training set of the given dataframe."""
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
    if architecture != 'resnet18': raise ValueError("This script is constrained to ResNet-18 duels.")

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
        model.fc = nn.Linear(model.fc.in_features, 2)

    elif mode == 'custom_r18':
        print(f"[RESNET18] Mode: Universal Foundation - Loading weights from {weights_path}...")
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        ckpt = torch.load(weights_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        clean_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_dict[k.replace('model.', '')] = v
            
        model.load_state_dict(clean_dict, strict=True)
        print("-> Foundation Weights mapped successfully.")

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
        
        # Storage for highly granular metrics per epoch
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
        
        # Calculate Global Class Accuracies
        if (all_targets == 1).any():
            self.log("val_hab_acc", (all_preds[all_targets == 1] == all_targets[all_targets == 1]).float().mean())
        if (all_targets == 0).any():
            self.log("val_nonhab_acc", (all_preds[all_targets == 0] == all_targets[all_targets == 0]).float().mean())

        # Calculate Sub-Domain Accuracies
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
# PLOTTING ENGINE (MODULE 6)
# ==============================================================================

def generate_comparison_plots(csv_path, base_output_dir):
    """
    Generates scatter plots and relative difference histograms for all 3 model combinations.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        print("Please ensure the path is correct and the file exists.")
        return

    df = pd.read_csv(csv_path)
    
    # Base list of metrics to plot
    metrics = [
        ('f1', 'F1 Score (Macro)'),
        ('acc', 'Overall Accuracy'),
        ('bal_acc', 'Balanced Accuracy'),
        ('hab_acc', 'Global HAB Accuracy'),
        ('nonhab_acc', 'Global non-HAB Accuracy'),
        ('iw_hab_acc', 'IW HABs Accuracy'),
        ('iw_nonhab_acc', 'IW non-HABs Accuracy'),
        ('land_acc', 'Land Accuracy'),
        ('clouds_acc', 'Clouds Accuracy')
    ]
    
    # Define the 3 duels
    scenarios = [
        {
            'col_x': 'baseline_', 'col_y': 'zeroshot_', 
            'name_x': 'Baseline', 'name_y': 'Zero-Shot', 
            'folder': 'duel_1_baseline_vs_zeroshot'
        },
        {
            'col_x': 'baseline_', 'col_y': 'ftfound_', 
            'name_x': 'Baseline', 'name_y': 'Fine-Tuned', 
            'folder': 'duel_2_baseline_vs_ftfound'
        },
        {
            'col_x': 'zeroshot_', 'col_y': 'ftfound_', 
            'name_x': 'Zero-Shot', 'name_y': 'Fine-Tuned', 
            'folder': 'duel_3_zeroshot_vs_ftfound'
        }
    ]
    
    for sc in scenarios:
        duel_dir = os.path.join(base_output_dir, sc['folder'])
        os.makedirs(duel_dir, exist_ok=True)
        print(f"\n[*] Generating plots for: {sc['name_x']} vs {sc['name_y']}")
        
        for m_suffix, title in metrics:
            m_x = sc['col_x'] + m_suffix
            m_y = sc['col_y'] + m_suffix
            
            # Ensure both columns exist and drop NaNs
            if m_x not in df.columns or m_y not in df.columns:
                continue
                
            valid_df = df.dropna(subset=[m_x, m_y])
            if len(valid_df) == 0: 
                continue
                
            # ---------------------------------------------------------
            # 1. SCATTER PLOT
            # ---------------------------------------------------------
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
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend(loc='upper left', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(duel_dir, f"scatter_{m_suffix}.png"), dpi=300)
            plt.close()

            # ---------------------------------------------------------
            # 2. BINNED HISTOGRAM (Percentage Difference)
            # ---------------------------------------------------------
            pct_diff = 100 * (valid_df[m_y] - valid_df[m_x]) / valid_df[m_x]
            mean_diff = pct_diff.mean()
            
            plt.figure(figsize=(8, 6))
            ax = sns.histplot(pct_diff, binwidth=0.1, kde=False, color='gray', edgecolor='black')
            
            # Add a vertical line at 0% (No improvement baseline)
            plt.axvline(0, color='black', linestyle='--', linewidth=2, label="0% Difference")
            
            # Add a vertical line for the true Mean
            plt.axvline(mean_diff, color='blue', linestyle='-', linewidth=2, label=f"Mean: {mean_diff:+.2f}%")
            
            plt.title(f"Relative Improvement: {sc['name_y']} vs {sc['name_x']}\nMetric: {title}", fontsize=14, fontweight='bold')
            plt.xlabel(f"% Improvement over {sc['name_x']}", fontsize=12)
            plt.ylabel("Count (Monte Carlo Iterations)", fontsize=12)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Ensure legend displays both lines clearly
            plt.legend(loc='upper right', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(duel_dir, f"hist_{m_suffix}.png"), dpi=300)
            plt.close()
            
        print(f" -> Saved to {duel_dir}/")


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
    
    hab_data = results_df[results_df['Actual'] == 1]
    nonhab_data = results_df[results_df['Actual'] == 0]
    
    metrics_dict['hab_acc'] = hab_data['Correct'].mean() if not hab_data.empty else np.nan
    metrics_dict['nonhab_acc'] = nonhab_data['Correct'].mean() if not nonhab_data.empty else np.nan
    
    for group in ['iw_hab', 'iw_nonhab', 'ow_hab', 'ow_nonhab', 'clouds', 'land']:
        group_data = results_df[results_df['Group'] == group]
        metrics_dict[f'{group}_acc'] = group_data['Correct'].mean() if not group_data.empty else np.nan

    return metrics_dict


if __name__ == "__main__":
    IW_DIR = "/home/kostas/AMFITRITE/data256"
    OW_DIR = "/home/kostas/AMFITRITE/OWdata"
    IW_EXCEL = "/home/kostas/AMFITRITE/dataset_summary_256x256pixels.xlsx"
    OW_CSV = "/home/kostas/AMFITRITE/OWdata/amfitrite_open_waters_master.csv"
    
    BASE_OUTPUT_DIR = "/home/kostas/AMFITRITE/IW_and_OW_CNN/IW_after_OW_and_IW_comparisons"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware initialized on: {DEVICE}")
    
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    TOTAL_ITERATIONS = 10
    
    BIGEARTH_WEIGHTS = '/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0_resnet18-s2-v0.2.0/model.safetensors'
    
    # Harmonize Once
    harmonizer = DatasetHarmonizer(IW_DIR, OW_DIR, IW_EXCEL, OW_CSV)
    master_dataframe = harmonizer.create_master_registry()
    master_results = []

    # ==========================================================================
    # THE LEAK-FREE 10-ITERATION NESTED MONTE CARLO LOOP
    # ==========================================================================
    for iteration in range(1, TOTAL_ITERATIONS + 1):
        print("\n" + "X"*60)
        print(f"XXX MONTE CARLO ITERATION {iteration}/{TOTAL_ITERATIONS} XXX")
        print("X"*60)
        
        L.seed_everything(1234 + iteration, workers=True)
        iter_dir = os.path.join(BASE_OUTPUT_DIR, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # 1. Split the ENTIRE dataset (No Leakage)
        master_split_csv = os.path.join(iter_dir, f"master_split_iter_{iteration}.csv")
        master_split_df = create_stratified_split(master_dataframe, master_split_csv, random_seed=42+iteration)
        master_weights = calculate_class_weights(master_split_df)

        # 2. Filter the Split to get the IW Augmented subsets
        iw_augmented_df = master_split_df[
            (master_split_df['source'] == 'IW') | 
            (master_split_df['strat_group'].isin(['clouds', 'land']))
        ].reset_index(drop=True)
        iw_weights = calculate_class_weights(iw_augmented_df)

        # 3. Create DataLoaders
        # Foundation loaders (uses entire dataset)
        found_train_ds = UniversalWaterDataset(master_split_df, mode='training', num_bands=10)
        found_val_ds   = UniversalWaterDataset(master_split_df, mode='validation', num_bands=10)
        found_train_loader = torch.utils.data.DataLoader(found_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        found_val_loader   = torch.utils.data.DataLoader(found_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        # IW Specialized loaders (uses filtered dataset)
        iw_train_ds = UniversalWaterDataset(iw_augmented_df, mode='training', num_bands=10)
        iw_val_ds   = UniversalWaterDataset(iw_augmented_df, mode='validation', num_bands=10)
        iw_test_ds  = UniversalWaterDataset(iw_augmented_df, mode='test', num_bands=10) # <-- The ultimate proving ground
        
        iw_train_loader = torch.utils.data.DataLoader(iw_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        iw_val_loader   = torch.utils.data.DataLoader(iw_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        iw_test_loader  = torch.utils.data.DataLoader(iw_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        iter_results = {'iteration': iteration}

        # ----------------------------------------------------------------------
        # PHASE 1: Train Baseline (BigEarthNet -> Trained on IW)
        # ----------------------------------------------------------------------
        print("\n[PHASE 1] Training Baseline BigEarthNet ResNet-18 on IW...")
        model_base = HABLightningSystem(
            architecture='resnet18', num_bands=10, mode='bigearthnet',
            weights_path=BIGEARTH_WEIGHTS, lr=1e-4, class_weights=iw_weights
        )
        
        ckpt_cb_base = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1, filename="best-baseline-{epoch:02d}")
        trainer_base = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="logs_baseline"), 
                                 callbacks=[ckpt_cb_base, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        trainer_base.fit(model_base, iw_train_loader, iw_val_loader)
        
        best_base = HABLightningSystem.load_from_checkpoint(ckpt_cb_base.best_model_path, class_weights=iw_weights).to(DEVICE)
        metrics_base = evaluate_split(best_base, iw_test_loader, DEVICE, "Test_Baseline", iter_dir)
        for k, v in metrics_base.items(): iter_results[f"baseline_{k}"] = v
        
        del model_base, trainer_base, best_base; gc.collect(); torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # PHASE 2: Train Universal Foundation (BigEarthNet -> Trained on Master)
        # ----------------------------------------------------------------------
        print("\n[PHASE 2] Training Universal Foundation Model on Master Dataset...")
        model_found = HABLightningSystem(
            architecture='resnet18', num_bands=10, mode='bigearthnet',
            weights_path=BIGEARTH_WEIGHTS, lr=1e-4, class_weights=master_weights
        )
        
        ckpt_cb_found = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1, filename="best-foundation-{epoch:02d}")
        trainer_found = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="logs_foundation"), 
                                  callbacks=[ckpt_cb_found, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        trainer_found.fit(model_found, found_train_loader, found_val_loader)
        
        # Zero-Shot Evaluation (Test the Foundation Model against the unseen IW Test Set!)
        best_found_path = ckpt_cb_found.best_model_path
        best_found = HABLightningSystem.load_from_checkpoint(best_found_path, class_weights=iw_weights).to(DEVICE) # Note: eval uses IW weights
        
        print("\n[Evaluating Zero-Shot Foundation on IW Test Set]...")
        metrics_zero = evaluate_split(best_found, iw_test_loader, DEVICE, "Test_ZeroShot", iter_dir)
        for k, v in metrics_zero.items(): iter_results[f"zeroshot_{k}"] = v
        
        del model_found, trainer_found, best_found; gc.collect(); torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # PHASE 3: Train Specialized Model (Foundation -> Fine-Tuned on IW)
        # ----------------------------------------------------------------------
        print("\n[PHASE 3] Fine-Tuning Specialized Model from Foundation Checkpoint...")
        model_ft = HABLightningSystem(
            architecture='resnet18', num_bands=10, mode='custom_r18',
            weights_path=best_found_path, lr=1e-5, class_weights=iw_weights # Load straight from Phase 2 ckpt!
        )
        
        ckpt_cb_ft = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1, filename="best-ftfound-{epoch:02d}")
        trainer_ft = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=CSVLogger(iter_dir, name="logs_ftfound"), 
                               callbacks=[ckpt_cb_ft, EarlyStopping(monitor="val_f1_macro", patience=20, mode="max")], enable_progress_bar=False)
        trainer_ft.fit(model_ft, iw_train_loader, iw_val_loader)
        
        best_ft = HABLightningSystem.load_from_checkpoint(ckpt_cb_ft.best_model_path, class_weights=iw_weights).to(DEVICE)
        metrics_ft = evaluate_split(best_ft, iw_test_loader, DEVICE, "Test_FT_Foundation", iter_dir)
        for k, v in metrics_ft.items(): iter_results[f"ftfound_{k}"] = v
        
        del model_ft, trainer_ft, best_ft; gc.collect(); torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # Backup and clear iteration memory
        # ----------------------------------------------------------------------
        master_results.append(iter_results)
        backup_csv = os.path.join(BASE_OUTPUT_DIR, "running_backup_results.csv")
        pd.DataFrame(master_results).to_csv(backup_csv, index=False)
        print(f"\n[Iteration {iteration} Complete] Results backed up.")
        
        del found_train_loader, found_val_loader, iw_train_loader, iw_val_loader, iw_test_loader
        gc.collect()

    print("\n" + "="*60)
    print("ALL 10 ITERATIONS COMPLETED!")
    print("="*60)
    
    final_csv_path = os.path.join(BASE_OUTPUT_DIR, "final_duel_results.csv")
    pd.DataFrame(master_results).to_csv(final_csv_path, index=False)
    
    generate_comparison_plots(final_csv_path, BASE_OUTPUT_DIR)