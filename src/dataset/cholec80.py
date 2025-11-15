import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

class Cholec80ToolDataset(Dataset):
    TOOL_COLS = [
        "Grasper", "Bipolar", "Hook",
        "Scissors", "Clipper", "Irrigator", "SpecimenBag",
    ]

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        img = img.float() / 255.0  # CHW tensor

        tools = torch.tensor(row[self.TOOL_COLS].values.astype("float32"))
        return img, tools
