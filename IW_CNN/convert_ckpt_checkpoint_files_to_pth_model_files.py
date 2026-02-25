# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 11:19:30 2026

@author: K. Pikounis
"""

import torch
import pytorch_lightning as L
import timm
from torchvision import models
import torch.nn as nn
import os

# --- 1. DEFINE THE MODEL CLASS (Must match training exactly) ---
# Paste your HABLightningModel class here so Python knows how to load it.
# (I am including the generic structure based on your previous codes)

class HABLightningModel(L.LightningModule):
    def __init__(self, mode='generic', weights_path=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._build_model(self.hparams.mode, self.hparams.weights_path)
        self.register_buffer("class_weights", torch.tensor([2.3, 1.0]))    
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
    def _build_model(self, mode, weights_path):
        # We define a "dummy" build because load_from_checkpoint 
        # will automatically overwrite these weights anyway.
        # We just need the structure to match.
        
        # LOGIC TO DETECT ARCHITECTURE based on 'mode' 
        # (You might need to adjust this if you used different classes for different models)
        if 'convnext' in str(weights_path).lower() or 'convnext' in mode:
             model = timm.create_model('convnextv2_base', pretrained=False, num_classes=2, in_chans=10)
        elif 'rdnet' in str(weights_path).lower() or 'rdnet' in mode:
             model = timm.create_model('rdnet_base', pretrained=False, num_classes=2, in_chans=10)
        elif 'vit' in str(weights_path).lower():
             # Remember to match the patch size/img size you used!
             model = timm.create_model('vit_base_patch8_224', pretrained=False, num_classes=2, in_chans=10)
        else:
             # Default to ResNet18
             model = models.resnet18(weights=None)
             model.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
             model.fc = nn.Linear(model.fc.in_features, 2)
             
        return model

# --- 2. CONVERSION SETTINGS ---
# Path to the .ckpt file you want to convert
CKPT_PATH = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\convnextv2_base\results3_new\hab_experiment\version_0\checkpoints\best-hab-epoch=15-val_f1=0.857.ckpt"

# Where to save the new .pth file
OUTPUT_DIR = r"C:\Users\KostasPikounis\OneDrive_Inlecom_Personal\OneDrive - INLECOM\Amfitrite\task2\IWD CNN\convnextv2_base\results3_new"
OUTPUT_NAME = "best_epoch_15.pth"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_NAME)

if __name__ == "__main__":
    print(f"Processing: {CKPT_PATH}")
    
    # 1. Load the heavy Lightning checkpoint
    # strict=False allows ignoring minor mismatches if you changed the class slightly
    print("Loading Lightning Checkpoint...")
    lightning_model = HABLightningModel.load_from_checkpoint(CKPT_PATH)
    
    # 2. Extract ONLY the model weights
    # We access .model to strip away the Lightning wrapper
    state_dict = lightning_model.model.state_dict()
    
    # 3. Save as standard PyTorch .pth
    print(f"Saving lightweight weights to: {OUTPUT_PATH}")
    torch.save(state_dict, OUTPUT_PATH)
    
    # 4. Verify Size Reduction
    ckpt_size = os.path.getsize(CKPT_PATH) / (1024 * 1024)
    pth_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    
    print(f"\nDone!")
    print(f"Original .ckpt size: {ckpt_size:.2f} MB")
    print(f"Converted .pth size: {pth_size:.2f} MB")
    print(f"Reduction: {ckpt_size/pth_size:.1f}x smaller")