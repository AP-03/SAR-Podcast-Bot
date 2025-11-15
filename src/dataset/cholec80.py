import os
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd

class Cholec80Dataset(Dataset):
    def __init__(self, csv_path, transform=None, task="both"):
        """
        task: "tools", "phase", or "both"
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        assert task in ["tools", "phase", "both"]
        self.task = task

        self.tool_cols = ["Grasper","Bipolar","Hook","Scissors","Clipper","Irrigator","SpecimenBag"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        image = image.float() / 255.0  # (C,H,W) after transform

        out = [image]

        if self.task in ["tools", "both"]:
            tools = torch.tensor(row[self.tool_cols].values.astype("float32"))
            out.append(tools)

        if self.task in ["phase", "both"]:
            phase = torch.tensor(int(row["phase"]), dtype=torch.long)
            out.append(phase)

        if self.task == "tools":
            return out[0], out[1]
        elif self.task == "phase":
            return out[0], out[1]
        else:
            return out[0], out[1], out[2]
