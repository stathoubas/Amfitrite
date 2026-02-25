# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 16:22:41 2026

@author: K. Pikounis
"""


import os
import pandas as pd
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import models
import pytorch_lightning as L
import timm
from tqdm import tqdm

# --- CONFIGURATION SECTION ---
# UPDATE THESE PATHS BEFORE RUNNING
DATA_ROOT = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD\data"
EXCEL_PATH = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\dataset_summary_with_splits_from_Server.xlsx"
OUTPUT_EXCEL_PATH = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\dataset_summary_with_splits_from_Server_with_predictions.xlsx"

# Dictionary mapping your 4 scenarios to their saved weight files
# Keys must match the scenario names used in the main loop below
MODEL_PATHS = {
    "res18_scl": r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\res18_2classes\results11\best_epoch_16.pth",
    "res18_no_scl": r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\res18_2classes\results12\best_epoch_26.pth",
    "convnext_scl": r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\convnextv2_base\results3_new\best_epoch_15.pth",
    "rdnet_no_scl": r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\rdnet_base\results3\best_epoch_35.pth"
}

# --- 1. DATASET CLASS ---
class HABDataset(Dataset):
    def __init__(self, dataframe, root_dir, use_mask=False, img_size=256):
        """
        Args:
            dataframe: The full pandas DataFrame.
            root_dir: Path to data.
            use_mask: Boolean. If True, loads SCL and applies water mask.
            img_size: Target image size (e.g., 256 or 224).
        """
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.use_mask = use_mask
        self.img_size = img_size
        
        self.band_names = [
            "B02_raw.tif", "B03_raw.tif", "B04_raw.tif", "B05_raw.tif", "B06_raw.tif",
            "B07_raw.tif", "B08_raw.tif", "B8A_raw.tif", "B11_raw.tif", "B12_raw.tif"
        ]
        
        # We don't strictly need labels for inference, but we keep the map for consistency
        self.label_map = {"Low": 0, "Moderate": 1, "High": 1}

    def __len__(self):
        return len(self.df)

    def apply_water_mask(self, bands_stack, scl_array):
        # SCL 6 is water. 
        mask = (scl_array == 6).astype(np.float32)
        return bands_stack * mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row['uid'])
        folder_path = os.path.join(self.root_dir, uid)
        
        # 1. Load Bands
        band_data = []
        for b_name in self.band_names:
            p = os.path.join(folder_path, b_name)
            # Handle missing files gracefully (optional safety)
            if not os.path.exists(p):
                # Return zero tensor if file missing (shouldn't happen in valid df)
                return torch.zeros((10, self.img_size, self.img_size)), 0 
                
            with rasterio.open(p) as src:
                band_data.append(src.read(1).astype(np.float32))
        
        bands_stack = np.stack(band_data, axis=0)
        
        # 2. Apply Mask (If requested)
        if self.use_mask:
            scl_path = os.path.join(folder_path, "SCL_raw.tif")
            if os.path.exists(scl_path):
                with rasterio.open(scl_path) as src:
                    scl = src.read(1)
                bands_stack = self.apply_water_mask(bands_stack, scl)
            else:
                pass # If SCL missing, just use unmasked
        
        # 3. To Tensor
        tensor = torch.from_numpy(bands_stack)
        
        # 4. Resize
        tensor = tensor.unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(self.img_size, self.img_size), 
            mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # 5. Normalize
        tensor = tensor / 10000.0
        
        return tensor, 0 # Return dummy label 0

# --- 2. MODEL CLASS ---
class HABLightningModel(L.LightningModule):
    def __init__(self, arch_name='resnet18', num_classes=2, in_chans=10):
        super().__init__()
        self.model = self._build_model(arch_name, num_classes, in_chans)

    def _build_model(self, arch, num_classes, in_chans):
        print(f"Building Architecture: {arch}")
        
        if arch == 'resnet18':
            # Standard ResNet18 construction
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
            
        elif arch == 'convnextv2_base':
            model = timm.create_model(
                'convnextv2_base', pretrained=False, num_classes=num_classes, in_chans=in_chans
            )
            return model
            
        elif arch == 'rdnet_base':
            model = timm.create_model(
                'rdnet_base', pretrained=False, num_classes=num_classes, in_chans=in_chans
            )
            return model
        
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, x):
        return self.model(x)

# --- 3. MAIN EVALUATION LOGIC ---
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # 1. Load Dataframe
    print(f"Loading DataFrame from {EXCEL_PATH}...")
    df = pd.read_excel(EXCEL_PATH) if EXCEL_PATH.endswith('.xlsx') else pd.read_csv(EXCEL_PATH)
    
    # 2. Define the 4 Scenarios
    # Structure: (Column Name, Arch Name, Use Mask?, Image Size)
    scenarios = [
        ("pred_res18_scl",     "resnet18",        True,  256),
        ("pred_res18_no_scl",  "resnet18",        False, 256),
        ("pred_convnext_scl",  "convnextv2_base", True,  256), # User asked for masked ConvNeXt
        ("pred_rdnet_no_scl",  "rdnet_base",      False, 256)
    ]
    
    for col_name, arch, use_mask, img_size in scenarios:
        print(f"\n--- Processing: {col_name} ---")
        print(f"Architecture: {arch} | Masking: {use_mask} | Size: {img_size}")
        
        # A. Setup Dataset & Loader
        # We process ALL rows, so we don't filter by 'split'
        ds = HABDataset(df, DATA_ROOT, use_mask=use_mask, img_size=img_size)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        # B. Setup Model
        model_wrapper = HABLightningModel(arch_name=arch)
        
        # C. Load Weights
        # Map the scenario column name to the keys in MODEL_PATHS
        # (Assuming the keys in MODEL_PATHS match the col_name for simplicity, 
        #  or mapped simply like 'res18_scl')
        # Let's clean the column name to match keys: 'pred_res18_scl' -> 'res18_scl'
        key = col_name.replace("pred_", "")
        ckpt_path = MODEL_PATHS.get(key)
        
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"CRITICAL WARNING: Weights not found for {key} at {ckpt_path}. Skipping!")
            continue
            
        print(f"Loading weights from: {ckpt_path}")
        
        # Handle loading: determine if it's a Lightning Checkpoint (.ckpt) or raw state_dict (.pth)
        if ckpt_path.endswith('.ckpt'):
            # Load from Lightning Checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            # Clean keys if necessary (remove 'model.' prefix if present in checkpoint)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model_wrapper.model.load_state_dict(state_dict, strict=False)
        else:
            # Assume raw state_dict (.pth)
            state_dict = torch.load(ckpt_path, map_location='cpu')
            model_wrapper.model.load_state_dict(state_dict, strict=False)
            
        model_wrapper.to(device)
        model_wrapper.eval()
        
        # D. Inference Loop
        preds_list = []
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc=f"Predicting {col_name}"):
                imgs = imgs.to(device)
                outputs = model_wrapper(imgs)
                # Get predicted class index (0 or 1)
                _, preds = torch.max(outputs, 1)
                preds_list.extend(preds.cpu().numpy())
        
        # E. Map Int -> String ("yes"/"no")
        # 0 = Low (No), 1 = Bloom (Yes)
        str_preds = ["yes" if p == 1 else "no" for p in preds_list]
        
        # F. Add to DataFrame
        df[col_name] = str_preds
        
    # 3. Save Final Result
    print(f"\nSaving results to {OUTPUT_EXCEL_PATH}...")
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print("Done!")

if __name__ == "__main__":
    run_evaluation()