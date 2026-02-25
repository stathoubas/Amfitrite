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
import timm

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
        
        # Define the exact order of bands to ensure the 10-channel stack is consistent
        self.band_names = [
            "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", "B05_raw.tif", "B06_raw.tif",
            "B07_raw.tif", "B08_raw.tif", "B8A_raw.tif", "B11_raw.tif", "B12_raw.tif"
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
        
        # Multiply the whole 10-layer stack by the 2D mask
        # Broadfasting handles applying the 2D mask to all 10 layers
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
            
        # 2. Load all 10 bands
        band_data = []
        for b_name in self.band_names:
            with rasterio.open(os.path.join(folder_path, b_name)) as src:
                # Read as float32 immediately for math
                band_data.append(src.read(1).astype(np.float32))
        
        # Stack into (10, 365, 365)
        bands_stack = np.stack(band_data, axis=0)
        
        # 3. Masking
        masked_data = self.apply_water_mask(bands_stack, scl)
        
        # 4. Convert to Torch Tensor
        tensor = torch.from_numpy(masked_data)
        
        # 5. Resize from 365x365 to 224x224
        # (Using unsqueeze because interpolate expects a batch dimension)
        tensor = tensor.unsqueeze(0) 
        tensor = torch.nn.functional.interpolate(
            tensor, size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # 6. Normalization
        # Divide by 10,000 to bring Sentinel-2 DN values to roughly 0-1
        tensor = tensor / 10000.0
        
        # 7. Augmentation (Only for training!)
        if self.mode == 'training':
            tensor = self.apply_augmentations(tensor)
            
        return tensor, label

class HABLightningModel(L.LightningModule):
    def __init__(self, mode='generic', weights_path=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Build the ConvNeXt Architecture
        self.model = self._build_model(self.hparams.mode, self.hparams.weights_path)
                
        # 2. Define Weights: [Low, Bloom]
        self.register_buffer("class_weights", torch.tensor([2.3, 1.0]))    
       
        # 3. Loss Function
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # 4. Metrics Setup
        def get_metrics(prefix):
            return torchmetrics.MetricCollection({
                'acc': MulticlassAccuracy(num_classes=2, average='micro'),
                'bal_acc': MulticlassAccuracy(num_classes=2, average='macro'),
                'f1': MulticlassF1Score(num_classes=2, average='macro'),
                'per_class': ClasswiseWrapper(
                    MulticlassAccuracy(num_classes=2, average=None),
                    labels=["Low", "Bloom"]
                )
            }, prefix=prefix)

        self.train_metrics = get_metrics('train_')
        self.val_metrics = get_metrics('val_')

    def _build_model(self, mode, weights_path):
        print(f"--- Building ViT Base Patch 8 (Mode: {mode}) ---")
        
        # 1. Create ViT Skeleton (Patch 8)
        # We use standard 224x224 input size here
        model = timm.create_model(
            'vit_base_patch8_224',
            pretrained=False, 
            num_classes=2, 
            in_chans=10
        )
        
        if mode == 'bigearthnet':
            print(f"Loading Pretrained Weights from {weights_path}...")
            
            if weights_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
            else:
                state_dict = torch.load(weights_path, map_location='cpu')
            
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '').replace('backbone.', '').replace('model.vision_encoder.', '')
                
                # Skip Head
                if "head." in name or "fc." in name:
                    continue
                
                # --- POSITIONAL EMBEDDINGS (120x120 -> 224x224) ---
                if 'pos_embed' in name and v.shape != model.pos_embed.shape:
                    print(f"Resizing pos_embed: {v.shape} -> {model.pos_embed.shape}")
                    
                    # v is (1, 226, 768). 
                    # 226 = 1 class token + 225 image tokens (15x15 grid)
                    cls_token = v[:, 0:1, :]
                    pos_tokens = v[:, 1:, :]
                    
                    # 1. Reshape flat tokens to 2D grid (15x15)
                    # Shape: (1, 225, 768) -> (1, 15, 15, 768) -> (1, 768, 15, 15)
                    pos_tokens = pos_tokens.reshape(1, 15, 15, -1).permute(0, 3, 1, 2)
                    
                    # 2. Interpolate to target grid (28x28)
                    # Why 28? Because input is 224, patch is 8. 224/8 = 28.
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(28, 28), mode='bicubic', align_corners=False
                    )
                    
                    # 3. Flatten back
                    # Shape: (1, 768, 28, 28) -> (1, 28, 28, 768) -> (1, 784, 768)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    
                    # 4. Recombine with Class Token
                    new_v = torch.cat((cls_token, pos_tokens), dim=1)
                    new_state_dict[name] = new_v
                    continue
                    
                new_state_dict[name] = v
            
            # Load weights
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            
            print("Weights Loaded.") 
            if 'patch_embed.proj.weight' in missing:
                print("CRITICAL WARNING: Patch Embeddings were NOT loaded!")
            else:
                print("SUCCESS: ViT Weights loaded and resized!")

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
    #EXCEL_PATH = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\dataset_summary.xlsx"
    #DATA_ROOT = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\data"
    #registry_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\dataset_summary_with_splits.xlsx"
    #output_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\convnextv2_base\results1"
    #weights_path = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\pretrained_model_weights\BIFOLD-BigEarthNetv2-0convnextv2_base-s2-v0.2.0\model.safetensors"
    #Logger_path = output_path
    #batch_size = 16
    #num_workers = 8
    
    EXCEL_PATH = "/home/kostas/AMFITRITE/dataset_summary.xlsx"
    DATA_ROOT = "/home/kostas/AMFITRITE/data"
    registry_path = "/home/kostas/AMFITRITE/dataset_summary_with_splits.xlsx"
    output_path = "/home/kostas/AMFITRITE/vit_base/results1"
    weights_path = "/home/kostas/AMFITRITE/pretrained_model_weights/BIFOLD-BigEarthNetv2-0vit_base_patch8_224-s2-v0.2.0/model.safetensors"
    Logger_path = output_path
    batch_size = 16
    num_workers = 4

    df = prepare_dataset_registry_and_split(EXCEL_PATH, DATA_ROOT, registry_path)
    
    train_ds = HABDataset(df, DATA_ROOT, mode='training')
    val_ds = HABDataset(df, DATA_ROOT, mode='validation')
    test_ds = HABDataset(df, DATA_ROOT, mode='test')
    
    # Batch Size 8 for 4GB GPU
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True)#, prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, pin_memory=True)#, prefetch_factor=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, pin_memory=True)#, prefetch_factor=4)

    # --- 2. SETUP MODEL & LOGGER ---
    model = HABLightningModel(mode='bigearthnet', lr=1e-4, weights_path=weights_path)

    
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
    summary(model, input_size=(batch_size, 10, 224, 224))

    print("\n--- Generating Architecture Diagram ---")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    try:
        # This creates a visual graph of the flow
        model_graph = draw_graph(
            model, 
            input_size=(batch_size, 10, 224, 224), 
            expand_nested=True,
            graph_name='convnextv2_base',
            save_graph=True,  # Saves a PDF/PNG
            directory=output_path # Saves it into your plots folder
        )
        print("Architecture diagram saved to 'plots/HAB_ResNet18_Arch.gv.pdf'")
    except Exception as e:
        print(f"Skipping visualization (Graphviz not found or error): {e}")

    '''
    # --- 3. TRAIN ---
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    '''
    trainer = L.Trainer(
        max_epochs=60,             # Increased for generic mode
        accelerator="gpu",
        devices=1,
        precision="16-mixed", #"32-true",
        #accumulate_grad_batches=2,
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
    
