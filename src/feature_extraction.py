import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List

from dataset.cholec80.util_dataset import Cholec80Dataset
from dataset.transform import get_basic_transforms
from models.tool_resnet import ToolCNN

def extract_features(csv_path: str, ckpt_path: str, out_file: str, device: str = "cuda", batch_size: int = 64):
    """
    Extracts features from a dataset specified by a CSV file and saves them to a file.
    """
    print(f"Loading dataset from: {csv_path}")
    ds = Cholec80Dataset(csv_path, transform=get_basic_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Loading model...")
    model = ToolCNN(num_tools=7, num_stages=7)

    # Load state dict with key remapping for compatibility
    ckpt = torch.load(ckpt_path, map_location=device)
    renamed_ckpt = {}
    for key, value in ckpt.items():
        if key.startswith("backbone."):
            new_key = key.replace("backbone.", "_full_resnet.", 1)
            renamed_ckpt[new_key] = value
        else:
            renamed_ckpt[key] = value
    
    model.load_state_dict(renamed_ckpt, strict=False)
    model.to(device)
    model.eval()

    all_feats = []
    all_tools = []
    all_phases = []

    print("Extracting features...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            imgs = batch['image'].to(device)
            tool_targets = batch['tools']
            phase_targets = batch['phase']
            
            tool_logits, stage_logits, feats = model(imgs, return_features=True)
            
            all_feats.append(feats.cpu())
            all_tools.append(tool_targets)
            all_phases.append(phase_targets)

    all_feats = torch.cat(all_feats, dim=0)
    all_tools = torch.cat(all_tools, dim=0)
    all_phases = torch.cat(all_phases, dim=0)

    print(f"Saving features to: {out_file}")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    torch.save(
        {"features": all_feats, "tools": all_tools, "phases": all_phases},
        out_file,
    )
    print("Extraction complete.")

if __name__ == "__main__":
    # --- Configuration ---
    # !!! USER: PLEASE VERIFY THE PATH TO YOUR TRAINING CSV FILE !!!
    CSV_TRAIN_PATH = "/Users/omarahmed/Desktop/UCL/Year 3/Term 1/Deep Learning/SAR-Podcast-Bot/src/dataset/cholec80/cholec80_manifest.csv"  # <-- PLEASE VERIFY THIS PATH
    CSV_VAL_PATH = "/Users/omarahmed/Desktop/UCL/Year 3/Term 1/Deep Learning/SAR-Podcast-Bot/src/dataset/cholec80/cholec80_val.csv"
    CKPT_PATH = "/Users/omarahmed/Desktop/UCL/Year 3/Term 1/Deep Learning/SAR-Podcast-Bot/src/tool_results/tool_detection_model.pth"
    OUT_DIR = "/Users/omarahmed/Desktop/UCL/Year 3/Term 1/Deep Learning/SAR-Podcast-Bot/src/tool_results"
    DEVICE = "cpu"

    # --- Run Extraction ---
    # Extract for training data
    print("--- Extracting Training Features ---")
    extract_features(
        csv_path=CSV_TRAIN_PATH,
        ckpt_path=CKPT_PATH,
        out_file=os.path.join(OUT_DIR, "cholec80_cnn_train_feats.pt"),
        device=DEVICE
    )

    # Extract for validation data
    print("\n--- Extracting Validation Features ---")
    extract_features(
        csv_path=CSV_VAL_PATH,
        ckpt_path=CKPT_PATH,
        out_file=os.path.join(OUT_DIR, "cholec80_cnn_val_feats.pt"),
        device=DEVICE
    )
