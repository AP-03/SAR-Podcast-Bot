import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.cholec80 import Cholec80ToolDataset
from dataset.transforms import get_basic_transforms
from models.tool_cnn import ToolCNN

def extract_features(csv_path, ckpt_path, out_dir, batch_size=64, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    ds = Cholec80ToolDataset(csv_path, transform=get_basic_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ToolCNN(num_tools=7)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    all_feats = []
    all_tools = []

    with torch.no_grad():
        for imgs, tool_targets in tqdm(loader):
            imgs = imgs.to(device)
            logits, probs, feats = model(imgs, return_features=True)
            all_feats.append(feats.cpu())
            all_tools.append(tool_targets)

    all_feats = torch.cat(all_feats, dim=0)  # [N, D]
    all_tools = torch.cat(all_tools, dim=0)  # [N, 7]

    torch.save(
        {"features": all_feats, "tools": all_tools},
        os.path.join(out_dir, "cholec80_tools_feats.pt"),
    )

if __name__ == "__main__":
    print("Call extract_features(...) after you have a trained checkpoint.")
