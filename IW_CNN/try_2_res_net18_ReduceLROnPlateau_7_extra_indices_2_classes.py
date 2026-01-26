# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:42:51 2026

@author: K. Pikounis
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
from pytorch_lightning.callbacks import ModelCheckpoint

from torchinfo import summary
from torchview import draw_graph
from pytorch_lightning.callbacks import EarlyStopping


class HABDataset(Dataset):
    def __init__(self, dataframe, root_dir, mode='train'):
        """
        Args:
            dataframe: The filtered DataFrame from our Registry (Block 1).
            root_dir: Path to the folder containing the UID subfolders.
            mode: 'train', 'validation', or 'test'.
        """
        self.df = dataframe[dataframe['split'] == mode].reset_index(drop=True)
        self.root_dir = root_dir
        self.mode = mode
        
        # Define the exact order of bands to ensure the 12-channel stack is consistent
        self.band_names = [
            "B01_raw.tif", "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", 
            "B05_raw.tif", "B06_raw.tif", "B07_raw.tif", "B08_raw.tif", 
            "B8A_raw.tif", "B09_raw.tif", "B11_raw.tif", "B12_raw.tif"
        ]
        
        # Map text labels to integers for the CNN
        self.label_map = {"Low": 0, "Moderate": 1, "High": 1}

    def __len__(self):
        return len(self.df)

    def apply_water_mask(self, bands_stack, scl_array):
        """
        Logic: SCL 6 is water, 7 is unclassified. 
        for now use only water
        Zeroes out everything else (Land, Clouds, etc.)
        """
        # Create a mask: 1.0 for water, 0.0 for others
        #mask = np.isin(scl_array, [6, 7]).astype(np.float32)
        mask = (scl_array == 6).astype(np.float32)
        
        # Multiply the whole 12-layer stack by the 2D mask
        # Broadfasting handles applying the 2D mask to all 12 layers
        masked_stack = bands_stack * mask
        return masked_stack

    def apply_augmentations(self, tensor):
        """
        Geometric augmentations that don't change the spectral values,
        only the orientation.
        """
        # 1. Random Horizontal Flip
        if random.random() > 0.5:
            tensor = TF.hflip(tensor)
            
        # 2. Random Vertical Flip
        if random.random() > 0.5:
            tensor = TF.vflip(tensor)
            
        # 3. Random Rotation (Fixed 90-degree steps to avoid interpolation artifacts)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            tensor = TF.rotate(tensor, angle)
            
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row['uid'])
        label = self.label_map[row['indicative_class']]
        folder_path = os.path.join(self.root_dir, uid)
        
        # 1. Load SCL for masking
        with rasterio.open(os.path.join(folder_path, "SCL_raw.tif")) as src:
            scl = src.read(1)
            
        # 2. Load all 12 bands
        band_data = []
        for b_name in self.band_names:
            with rasterio.open(os.path.join(folder_path, b_name)) as src:
                # Read as float32 immediately for math
                band_data.append(src.read(1).astype(np.float32))
        
        # Stack into (12, 365, 365)
        bands_stack = np.stack(band_data, axis=0)
        
        # 3. Masking
        masked_data = self.apply_water_mask(bands_stack, scl)
        
                
        # 4. Extract indices
        # Band Mapping based on self.band_names list:
        # 0:B01, 1:B02(Blue), 2:B03(Green), 3:B04(Red), 4:B05(RE1), 
        # 7:B08(NIR), 10:B11(SWIR)
        
        # Add epsilon to avoid division by zero
        eps = 1e-8
        
        b_blue  = masked_data[1]
        b_green = masked_data[2]
        b_red   = masked_data[3]
        b_re1   = masked_data[4]
        b_nir   = masked_data[7]
        b_swir  = masked_data[10]

        # 1. NDVI (Vegetation/Algae biomass)
        ndvi = (b_nir - b_red) / (b_nir + b_red + eps)
        
        # 2. NDWI (McFeeters - Water body delineation)
        ndwi = (b_green - b_nir) / (b_green + b_nir + eps)
        
        # 3. MNDWI (Modified NDWI - usually better for open water)
        mndwi = (b_green - b_swir) / (b_green + b_swir + eps)
        
        # 4. NDCI (Normalized Difference Chlorophyll Index - CRITICAL for HAB)
        ndci = (b_re1 - b_red) / (b_re1 + b_red + eps)
        
        # 5. NDre (Normalized Difference Red Edge - Good for vegetation stress/bloom)
        ndre = (b_nir - b_re1) / (b_nir + b_re1 + eps)
        
        # 6. NDTI (Normalized Difference Turbidity Index - Requested)
        ndti = (b_red - b_green) / (b_red + b_green + eps)
        
        # 7. SABI (Surface Algal Bloom Index - Requested)
        sabi = (b_nir - b_red) / (b_blue + b_green + eps)
        
        # Stack indices: Shape becomes (7, 365, 365)
        indices_stack = np.stack([ndvi, ndwi, mndwi, ndci, ndre, ndti, sabi], axis=0)
        
        # Concatenate with raw data: Shape becomes (19, 365, 365)
        # We assume masked_data is already float32.
        full_stack = np.concatenate([masked_data, indices_stack], axis=0)

        # 5. Convert to Torch Tensor
        tensor = torch.from_numpy(full_stack)
        
        # 6. Resize from 365x365 to 256x256
        # (Using unsqueeze because interpolate expects a batch dimension)
        tensor = tensor.unsqueeze(0) 
        tensor = torch.nn.functional.interpolate(
            tensor, size=(256, 256), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # 7. Normalization
        # Divide by 10,000 to bring Sentinel-2 DN values to roughly 0-1
        tensor = tensor / 10000.0
        
        # 8. Augmentation (Only for training!)
        if self.mode == 'training':
            tensor = self.apply_augmentations(tensor)
            
        return tensor, label


class HABLightningModel(L.LightningModule):
    def __init__(self, mode='generic', weights_path=None, lr=1e-4):
        super().__init__()
        # Save hyperparameters so they are logged and accessible
        self.save_hyperparameters()
        
        # 1. Build the Architecture with the Surgery logic
        self.model = self._build_model(self.hparams.mode, self.hparams.weights_path)
                
        # 2. Define Weights: [Low, Bloom]
        self.register_buffer("class_weights", torch.tensor([2.3, 1.0]))    
       
        # 3. Loss Function
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        
        # 3. Metrics Setup (Accuracy, F1, and Per-Class)
        def get_metrics(prefix):
            return torchmetrics.MetricCollection({
                # Standard Accuracy (biased towards majority)
                'acc': MulticlassAccuracy(num_classes=2, average='micro'),
                
                # Balanced Accuracy (Fair average of Recall_Low and Recall_Bloom)
                'bal_acc': MulticlassAccuracy(num_classes=2, average='macro'),
                
                # Macro F1 (The gold standard for stopping)
                'f1': MulticlassF1Score(num_classes=2, average='macro'),
                
                # Per-class breakdown
                'per_class': ClasswiseWrapper(
                    MulticlassAccuracy(num_classes=2, average=None),
                    labels=["Low", "Bloom"]
                )
            }, prefix=prefix)

        self.train_metrics = get_metrics('train_')
        self.val_metrics = get_metrics('val_')
    
    def _build_model(self, mode, weights_path):
        # Start with a "raw" ResNet18 structure
        model = models.resnet18(weights=None)
        
        # Total input channels = 12 (Raw) + 7 (Indices) = 19
        input_channels = 19
        
        # Step A: Stem Surgery
        # We define a new conv1 with 19 input filters
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if mode == 'generic':
            # Option A: Inflate ImageNet weights
            print(f"Mode: Generic - Inflating ImageNet weights to {input_channels} channels...")
            temp_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            with torch.no_grad():
                # Average the RGB weights (shape: 64, 3, 7, 7) -> (64, 1, 7, 7)
                w_avg = temp_resnet.conv1.weight.mean(dim=1, keepdim=True)
                # Repeat for all 19 channels
                model.conv1.weight.copy_(w_avg.repeat(1, input_channels, 1, 1))
            
            # Load the rest of the pretrained body
            state_dict = temp_resnet.state_dict()
            del state_dict['conv1.weight']
            model.load_state_dict(state_dict, strict=False)
                
        elif mode == 's2':
            # Option B: Load Sentinel-2 SSL4EO weights
            print(f"Mode: S2 - Performing weight surgery on {weights_path}...")
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '').replace('backbone.', '')
                
                # Check for the first layer weights
                if name == 'conv1.weight' and v.shape[1] == 13:
                    # 1. Extract the 12 spectral bands (as before)
                    # Keep 0-9, skip 10, keep 11-12
                    keep_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
                    w_spectral = v[:, keep_indices, :, :] # Shape: [64, 12, 7, 7]
                    
                    # 2. Initialize weights for the 7 indices
                    # We use the mean of the spectral weights to initialize the indices.
                    # This is better than random initialization.
                    w_mean = w_spectral.mean(dim=1, keepdim=True) # Shape: [64, 1, 7, 7]
                    w_indices = w_mean.repeat(1, 7, 1, 1) # Shape: [64, 7, 7, 7]
                    
                    # 3. Concatenate to make 19 channels
                    w_total = torch.cat([w_spectral, w_indices], dim=1) # Shape: [64, 19, 7, 7]
                    
                    new_state_dict[name] = w_total
                else:
                    new_state_dict[name] = v
            
            # Load everything
            model.load_state_dict(new_state_dict, strict=False)

        # Step B: Head Surgery (Change output from 1000 to 3 classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Track metrics
        self.train_metrics.update(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Log all accumulated training metrics at the end of the epoch
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Track metrics
        self.val_metrics.update(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Log all accumulated validation metrics (including per-class acc)
        output = self.val_metrics.compute()
        self.log_dict(output)
        self.val_metrics.reset()
    
     
    def configure_optimizers(self):
        # 1. Use self.hparams.lr to grab the value you passed in __init__
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        
        # 2. Define the Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',      # We want to maximize F1
            factor=0.1,      # Drop LR by 10x (e.g., 1e-4 -> 1e-5)
            patience=5,      # If no improvement for 5 epochs...
        )
        
        # 3. Return the specific dictionary Lightning expects
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1", # The metric to watch
            },
        }    
    
    '''
    def configure_optimizers(self):
        # Using AdamW as it is better for transformers/modern CNNs
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    '''

def prepare_dataset_registry_and_split(excel_path, data_root, output_registry_path):
    """
    Filters existing folders, performs stratified split, and saves a registry Excel.
    """
    print("Step 1: Loading master Excel and verifying files on disk...")
    df = pd.read_excel(excel_path)
    
    # Initialize the 'split' column as 'junk'
    df['split'] = 'junk'
    
    # Check physical existence of folders
    # We assume 'uid' column in Excel matches the folder names
    valid_indices = []
    for idx, row in df.iterrows():
        uid = str(row['uid'])
        folder_path = os.path.join(data_root, uid)
        
        if os.path.isdir(folder_path):
            valid_indices.append(idx)
        else:
            # This will remain as 'junk' in the final Excel
            pass

    print(f"  -> Found {len(valid_indices)} valid folders out of {len(df)} total rows.")
    
    # Create a sub-dataframe of only valid items for splitting
    valid_df = df.loc[valid_indices].copy()
    
    # Step 2: Stratified Split (70% Train, 15% Val, 15% Test)
    # First split: Separate Train (70%) from the rest (30%)
    train_idx, temp_idx = train_test_split(
        valid_df.index, 
        test_size=0.30, 
        stratify=valid_df['indicative_class'], 
        random_state=42
    )
    
    # Second split: Separate Temp into Val (50% of 30% = 15%) and Test (15%)
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.50, 
        stratify=valid_df.loc[temp_idx, 'indicative_class'], 
        random_state=42
    )
    
    # Step 3: Assign the labels back to the main dataframe
    df.loc[train_idx, 'split'] = 'training'
    df.loc[val_idx, 'split'] = 'validation'
    df.loc[test_idx, 'split'] = 'test'
    
    # Step 4: Save the Registry
    df.to_excel(output_registry_path, index=False)
    
    print(f"Success! Registry saved to: {output_registry_path}")
    print(df['split'].value_counts()) # Prints counts for Training, Val, Test, and Junk
    
    return df


def plot_training_history(csv_path, output_dir="plots"):
    """Reads the CSV log and saves 5 specific plots, robust to column naming."""
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    try:
        metrics = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Could not find log file at {csv_path}. Skipping plots.")
        return

    # Helper to find the actual column name in the CSV
    def find_col(prefix, label):
        # Looks for columns that contain both the prefix (e.g., 'train') and the label (e.g., 'Low')
        # This handles variations like 'train_per_class_Low' or 'train_per_class_MulticlassAccuracy_Low'
        candidates = [c for c in metrics.columns if prefix in c and label in c]
        if candidates:
            return candidates[0] # Return the first match
        return None

    # Helper to generate one plot
    def save_plot(train_col_candidate, val_col_candidate, title, filename):
        plt.figure(figsize=(10, 6))
        
        # Resolve actual column names
        # For simple metrics like loss/f1, we use exact names. For per-class, we use the finder.
        if "per_class" in train_col_candidate:
            # Extract the label we are looking for (e.g., "Low") from the candidate string
            label = train_col_candidate.split("_")[-1] 
            metric_train = find_col("train", label)
            metric_val = find_col("val", label)
        else:
            metric_train = train_col_candidate
            metric_val = val_col_candidate

        if metric_train and metric_val and metric_train in metrics.columns and metric_val in metrics.columns:
            clean_train = metrics[[metric_train, 'epoch']].dropna()
            clean_val = metrics[[metric_val, 'epoch']].dropna()
            
            plt.plot(clean_train['epoch'], clean_train[metric_train], label='Train', marker='o')
            plt.plot(clean_val['epoch'], clean_val[metric_val], label='Validation', marker='o')
            
            plt.title(title)
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"{output_dir}/{filename}")
            plt.close()
            print(f"Saved: {filename}")
        else:
            print(f"Skipping {filename}: Could not find columns for {title}")

    # 1. Loss Plot
    save_plot('train_loss_epoch', 'val_loss', "Overall Loss", "1_loss_curve.png")

    # 2. F1 Score Plot (Macro)
    save_plot('train_f1', 'val_f1', "Macro F1 Score (Balance)", "2_f1_curve.png")

    # --- PLOT 3: Balanced vs Standard Accuracy ---
    plt.figure(figsize=(10, 6))
    if 'val_acc' in metrics.columns and 'val_bal_acc' in metrics.columns:
        clean_data = metrics[['epoch', 'val_acc', 'val_bal_acc']].dropna()
        plt.plot(clean_data['epoch'], clean_data['val_acc'], label='Standard Accuracy (Micro)', marker='o', linestyle='--')
        plt.plot(clean_data['epoch'], clean_data['val_bal_acc'], label='Balanced Accuracy (Macro)', marker='o', linewidth=2)
        plt.title("Standard vs Balanced Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/3_acc_comparison.png")
        plt.close()
        print("Saved: 3_acc_comparison.png")

    # --- PLOT 4: Accuracy Low Class ---
    save_plot('train_per_class_Low', 'val_per_class_Low', 
              "Accuracy: Low Class (Specificity  - True Negative Rate)", "4_acc_low_curve.png")

    # --- PLOT 5: Accuracy Bloom Class ---
    # Note: Label is 'Bloom' because we renamed it in the MetricsCollection
    save_plot('train_per_class_Bloom', 'val_per_class_Bloom', 
             "Accuracy: Bloom Class (Recall - True Positive Rate)", "5_acc_bloom_curve.png")

def evaluate_split(model, loader, device, split_name="Test", output_dir="plots"):
    """Runs inference and generates a confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\n--- Evaluating on {split_name} Set ---")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Text Report with correct names
    # Class 0 = Low, Class 1 = Bloom
    print(classification_report(all_labels, all_preds, target_names=["Low", "Bloom"]))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Low", "Bloom"], 
                yticklabels=["Low", "Bloom"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({split_name})")
    plt.savefig(f"{output_dir}/cm_{split_name.lower()}.png")
    plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {DEVICE}")
    
    # --- 1. SETUP DATA ---
    # Ensure Block 1 functions (prepare_dataset...) are defined above or imported
    EXCEL_PATH = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\dataset_summary.xlsx"
    DATA_ROOT = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\data"
    registry_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\dataset_summary_with_splits.xlsx"
    output_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\res18_2classes_7_extra_indices\results4"
    Logger_path = output_path
    batch_size = 32
    num_workers = 8

    df = prepare_dataset_registry_and_split(EXCEL_PATH, DATA_ROOT, registry_path)
    
    train_ds = HABDataset(df, DATA_ROOT, mode='training')
    val_ds = HABDataset(df, DATA_ROOT, mode='validation')
    test_ds = HABDataset(df, DATA_ROOT, mode='test')
    
    # Batch Size 8 for 4GB GPU
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    # --- 2. SETUP MODEL & LOGGER ---
    #model = HABLightningModel(mode="generic", weights_path = None, lr=1e-4)
    model = HABLightningModel(mode='s2', lr=1e-4, weights_path=r'C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\pretrained_model_weights\MoCo_ResNet18_S2-L1C 13 bands\B13_rn18_moco_0099_ckpt.pth')
    
    logger = CSVLogger(output_path, name="hab_experiment")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="best-hab-{epoch:02d}-{val_f1:.3f}"
    )
    
    # 2. Define Early Stopping (Stops training if no improvement)
    early_stop_callback = EarlyStopping(
        monitor="val_f1",  # Watch the F1 score
        patience=20,       # Wait 10 epochs for an improvement before stopping
        mode="max",        # Higher is better
        verbose=True       # Print a message when it stops
    )    

    print("\n--- Model Summary ---")
    # Input size: (Batch_Size, Channels, Height, Width)
    # We use Batch=batch_size, Channels=12, Size=256x256
    summary(model, input_size=(batch_size, 19, 256, 256))

    print("\n--- Generating Architecture Diagram ---")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    try:
        # This creates a visual graph of the flow
        model_graph = draw_graph(
            model, 
            input_size=(batch_size, 19, 256, 256), 
            expand_nested=True,
            graph_name='HAB_ResNet18_Arch',
            save_graph=True,  # Saves a PDF/PNG
            directory=output_path # Saves it into your plots folder
        )
        print("Architecture diagram saved to 'plots/HAB_ResNet18_Arch.gv.pdf'")
    except Exception as e:
        print(f"Skipping visualization (Graphviz not found or error): {e}")

    trainer = L.Trainer(
        max_epochs=60,             # Increased for generic mode
        accelerator="gpu",
        devices=1,
        precision="32-true",
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback], # Ensure early_stop is here!
        log_every_n_steps=10       # This is fine, leave it.
    )
    
    print("Starting Training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training Complete!")

    # --- 4. ANALYSIS & PLOTTING ---
    print("\nGenerating Plots...")
    # Path to metrics.csv
    metrics_path = f"{logger.log_dir}/metrics.csv"
    plot_training_history(metrics_path, output_dir=output_path)
    
    # Load Best Model
    print(f"\nLoading Best Model from: {checkpoint_callback.best_model_path}")
    best_model = HABLightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.to(DEVICE)
    best_model.eval()
    
    # Run Final Evaluation
    # Note: Using validation set here just to demo, ideally run on Test
    train_eval_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    
    evaluate_split(best_model, train_eval_loader, DEVICE, split_name="Train", output_dir=output_path)
    evaluate_split(best_model, val_loader, DEVICE, split_name="Validation", output_dir=output_path)
    evaluate_split(best_model, test_loader, DEVICE, split_name="Test", output_dir=output_path)
    
    print("\n--- Saving Optimal Model Weights ---")
    # Get the path for the raw weights
    weights_path = os.path.join(output_path, "optimal_weights.pth")
    
    # 2. Extract the state_dict from the LightningModule
    # We use best_model.model because we wrapped the ResNet inside self.model
    torch.save(best_model.model.state_dict(), weights_path)
    
    print(f"Optimal weights saved to: {weights_path}")
    
    
    print("\n--- Project Complete. Check the 'plots' folder for your 5 curves and matrices! ---")
    