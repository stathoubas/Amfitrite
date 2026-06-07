# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:44:43 2026

@author: K. Pikounis
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import timm
from torchvision import models
from safetensors.torch import load_file
import gc

# ==============================================================================
# UNIFIED DATASET (Handles both Masked and Unmasked flows)
# ==============================================================================
class HABMonteCarloDataset(Dataset):
    def __init__(self, dataframe, root_dir, mode='train', apply_mask=False):
        self.df = dataframe[dataframe['split'] == mode].reset_index(drop=True)
        self.root_dir = root_dir
        self.mode = mode
        self.apply_mask = apply_mask
        
        self.band_names = [
            "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", "B05_raw.tif", "B06_raw.tif",
            "B07_raw.tif", "B08_raw.tif", "B8A_raw.tif", "B11_raw.tif", "B12_raw.tif"
        ]
        self.label_map = {"Low": 0, "Moderate": 1, "High": 1}

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
        uid = str(row['uid'])
        label = self.label_map[row['indicative_class']]
        folder_path = os.path.join(self.root_dir, uid)
        
        band_data = []
        for b_name in self.band_names:
            with rasterio.open(os.path.join(folder_path, b_name)) as src:
                band_data.append(src.read(1).astype(np.float32))
        bands_stack = np.stack(band_data, axis=0)
        
        # Apply Water Mask conditionally
        if self.apply_mask:
            with rasterio.open(os.path.join(folder_path, "SCL_raw.tif")) as src:
                scl = src.read(1)
            mask = (scl == 6).astype(np.float32)
            bands_stack = bands_stack * mask
            
        tensor = torch.from_numpy(bands_stack)
        tensor = tensor.unsqueeze(0) 
        tensor = torch.nn.functional.interpolate(
            tensor, size=(256, 256), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        tensor = tensor / 10000.0
        
        if self.mode == 'training':
            tensor = self.apply_augmentations(tensor)
            
        return tensor, label

# ==============================================================================
# UNIFIED MODEL BUILDER
# ==============================================================================
class HABUnifiedLightningModel(L.LightningModule):
    def __init__(self, model_type, weights_path, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        
        self.model = self._build_model(model_type, weights_path)
        self.register_buffer("class_weights", torch.tensor([2.3, 1.0]))    
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        self.train_f1 = MulticlassF1Score(num_classes=2, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=2, average='macro')

    def _build_model(self, model_type, weights_path):
        # 1. TIMM Models (RDNet, ConvNeXt)
        if model_type in ['rdnet_base', 'convnextv2_base']:
            model = timm.create_model(model_type, pretrained=False, num_classes=2, in_chans=10)
            state_dict = load_file(weights_path) if weights_path.endswith('.safetensors') else torch.load(weights_path, map_location='cpu')
            
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '').replace('backbone.', '').replace('model.vision_encoder.', '')
                if "head." in name or "fc." in name: continue
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict, strict=False)

        # 2. Torchvision Model (ResNet18)
        elif model_type == 'resnet18':
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            state_dict = load_file(weights_path) if weights_path.endswith('.safetensors') else torch.load(weights_path, map_location='cpu')
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '').replace('backbone.', '').replace('model.vision_encoder.', '')
                if "fc." in name: continue
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict, strict=False)
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_f1(preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_f1(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"}}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def create_random_split(excel_path, data_root, seed):
    """Creates a new randomized split based on the given seed."""
    df = pd.read_excel(excel_path)
    df['split'] = 'junk'
    
    valid_indices = [idx for idx, row in df.iterrows() if os.path.isdir(os.path.join(data_root, str(row['uid'])))]
    valid_df = df.loc[valid_indices].copy()
    
    train_idx, temp_idx = train_test_split(
        valid_df.index, test_size=0.30, stratify=valid_df['indicative_class'], random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=valid_df.loc[temp_idx, 'indicative_class'], random_state=seed
    )
    
    df.loc[train_idx, 'split'] = 'training'
    df.loc[val_idx, 'split'] = 'validation'
    df.loc[test_idx, 'split'] = 'test'
    return df

def evaluate_metrics(model, loader, device):
    """Calculates evaluation metrics without Lightning overhead."""
    model.eval()
    
    acc_metric = MulticlassAccuracy(num_classes=2, average='micro').to(device)
    bal_acc_metric = MulticlassAccuracy(num_classes=2, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=2, average='macro').to(device)
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            acc_metric.update(preds, labels)
            bal_acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            
    return {
        'F1': round(f1_metric.compute().item(), 4),
        'Accuracy': round(acc_metric.compute().item(), 4),
        'Balanced_Accuracy': round(bal_acc_metric.compute().item(), 4)
    }

# ==============================================================================
# MAIN MONTE CARLO EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    EXCEL_PATH = "/home/kostas/AMFITRITE/dataset_summary.xlsx"
    DATA_ROOT = "/home/kostas/AMFITRITE/data"
    OUTPUT_DIR = "/home/kostas/AMFITRITE/monte_carlo_results"
    
    # Path mappings for weights
    WEIGHTS_PATHS = {
        'rdnet_base': "/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0rdnet_base-s2-v0.2.0/model.safetensors",
        'convnextv2_base': "/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0convnextv2_base-s2-v0.2.0/model.safetensors",
        'resnet18': "/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0_resnet18-s2-v0.2.0/model.safetensors"
    }

    # Model Configurations to evaluate
    MODELS_CONFIG = [
        {'name': 'RDNet_NoMask', 'model_type': 'rdnet_base', 'apply_mask': False, "BATCH_SIZE":8, "NUM_WORKERS":4},
        {'name': 'ConvNeXt_NoMask', 'model_type': 'convnextv2_base', 'apply_mask': False, "BATCH_SIZE":8, "NUM_WORKERS":4},
        {'name': 'ResNet18_Mask', 'model_type': 'resnet18', 'apply_mask': True, "BATCH_SIZE":64, "NUM_WORKERS":8},
        {'name': 'ResNet18_NoMask', 'model_type': 'resnet18', 'apply_mask': False, "BATCH_SIZE":64, "NUM_WORKERS":8}
    ]

    #BATCH_SIZE = 64
    #NUM_WORKERS = 8
    MAX_EPOCHS = 60
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_results = []

    print(f"Starting 10-Fold Monte Carlo Evaluation on {DEVICE}...")

    for iteration in range(1, 11):
        # Generate a distinct, deterministic seed for each iteration
        iter_seed = 4321 + iteration 
        print(f"\n{'='*50}\nITERATION {iteration}/10 (Seed: {iter_seed})\n{'='*50}")
        
        # 1. Generate unique split for this iteration
        split_df = create_random_split(EXCEL_PATH, DATA_ROOT, seed=iter_seed)
        
        for config in MODELS_CONFIG:
            print(f"\n---> Training Config: {config['name']}")
            
            # 2. Initialize Datasets and Loaders for this specific config (handling mask flags)
            train_ds = HABMonteCarloDataset(split_df, DATA_ROOT, mode='training', apply_mask=config['apply_mask'])
            val_ds = HABMonteCarloDataset(split_df, DATA_ROOT, mode='validation', apply_mask=config['apply_mask'])
            test_ds = HABMonteCarloDataset(split_df, DATA_ROOT, mode='test', apply_mask=config['apply_mask'])
            
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['NUM_WORKERS'], persistent_workers=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], persistent_workers=True)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], persistent_workers=True)
            eval_train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], persistent_workers=True)
            
            # 3. Setup Model, Callbacks, and Trainer
            model = HABUnifiedLightningModel(
                model_type=config['model_type'], 
                weights_path=WEIGHTS_PATHS[config['model_type']], 
                lr=1e-4
            )
            
            run_dir = os.path.join(OUTPUT_DIR, f"iter_{iteration}", config['name'])
            logger = CSVLogger(run_dir, name="logs")
            checkpoint_callback = ModelCheckpoint(
                monitor="val_f1", mode="max", save_top_k=1, filename="best-checkpoint"
            )
            early_stop_callback = EarlyStopping(
                monitor="val_f1", patience=20, mode="max", verbose=False
            )
            
            trainer = L.Trainer(
                max_epochs=MAX_EPOCHS,
                accelerator="gpu",
                devices=1,
                precision="32-true",
                logger=logger,
                callbacks=[checkpoint_callback, early_stop_callback],
                enable_progress_bar=False 
            )
            
            # 4. Train Model
            trainer.fit(model, train_loader, val_loader)
            
            # 5. Load Best Checkpoint and Evaluate
            best_model_path = checkpoint_callback.best_model_path
            best_model = HABUnifiedLightningModel.load_from_checkpoint(best_model_path)
            best_model.to(DEVICE)
            
            train_metrics = evaluate_metrics(best_model, eval_train_loader, DEVICE)
            val_metrics = evaluate_metrics(best_model, val_loader, DEVICE)
            test_metrics = evaluate_metrics(best_model, test_loader, DEVICE)
            
            # 6. Store Results
            for split_name, metrics in zip(['Train', 'Validation', 'Test'], [train_metrics, val_metrics, test_metrics]):
                master_results.append({
                    'Iteration': iteration,
                    'Seed': iter_seed,
                    'Model_Config': config['name'],
                    'Split': split_name,
                    'F1_Macro': metrics['F1'],
                    'Accuracy': metrics['Accuracy'],
                    'Balanced_Accuracy': metrics['Balanced_Accuracy']
                })
            
            # Clean up memory
            del model, best_model, trainer, train_loader, val_loader, test_loader, eval_train_loader
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ==============================================================================
    # EXPORT RESULTS TO EXCEL
    # ==============================================================================
    results_df = pd.DataFrame(master_results)
    excel_out_path = os.path.join(OUTPUT_DIR, "Master_Monte_Carlo_IWD_Results.xlsx")
    results_df.to_excel(excel_out_path, index=False)
    
    print("\n" + "="*50)
    print("MONTE CARLO EVALUATION COMPLETE!")
    print(f"Results successfully saved to: {excel_out_path}")
    print("="*50)