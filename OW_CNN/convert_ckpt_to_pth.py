# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:25:09 2026

"""

import torch
import os

def convert_ckpt_to_raw_pth(ckpt_path, output_pth_path):
    """
    Strips PyTorch Lightning metadata (Optimizers, Epochs, etc.) from a .ckpt
    and saves purely the standardized neural network weights to a .pth file.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find checkpoint at {ckpt_path}")
        
    print(f"Loading Lightning checkpoint: {ckpt_path}")
    
    # Load to CPU to avoid GPU memory spikes during conversion
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Extract the state dictionary
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    clean_dict = {}
    skipped_keys = []
    
    # Clean the keys to match standard torchvision ResNets
    for key, value in state_dict.items():
        # Only keep the actual neural network weights
        if key.startswith('model.'):
            # Remove the 'model.' prefix
            clean_key = key.replace('model.', '')
            clean_dict[clean_key] = value
        else:
            skipped_keys.append(key)

    print(f"\nExtraction complete:")
    print(f" -> Kept {len(clean_dict)} model weight tensors.")
    print(f" -> Skipped {len(skipped_keys)} lightning/custom artifacts (e.g., {', '.join(skipped_keys[:3])}...)")

    # Save the purely standardized weights
    torch.save(clean_dict, output_pth_path)
    
    # Compare sizes
    ckpt_size = os.path.getsize(ckpt_path) / (1024 * 1024)
    pth_size = os.path.getsize(output_pth_path) / (1024 * 1024)
    
    print(f"\nFile Size Reduction:")
    print(f" -> Original .ckpt: {ckpt_size:.2f} MB")
    print(f" -> New .pth:       {pth_size:.2f} MB")
    print(f"Saved clean standard PyTorch weights to: {output_pth_path}")

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Point this to your best AMFITRITE checkpoint
    INPUT_CKPT = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/logs/version_0/checkpoints/best-hab-epoch=52-val_f1_macro=0.882.ckpt"
    
    # Define where you want the clean file saved
    OUTPUT_PTH = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18/amfitrite_resnet18_bigearth_best.pth"

    INPUT_CKPT = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18_focal_loss/logs/version_0/checkpoints/best-hab-epoch=29-val_f1_macro=0.864.ckpt"
    OUTPUT_PTH = "/home/kostas/AMFITRITE/IW_and_OW_CNN/bigearthnet_resnet18_focal_loss/resnet18_bigearth_focal_loss.pth"
    
    convert_ckpt_to_raw_pth(INPUT_CKPT, OUTPUT_PTH)
