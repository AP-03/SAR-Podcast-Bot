import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from dataset.cholec80 import Cholec80ToolDataset
from dataset.transforms import get_basic_transforms
from models.tool_cnn import ToolCNN

def train_tools(
    train_csv,
    val_csv,
    epochs=10,
    batch_size=32,
    lr=1e-4,
    device="cuda",
):
    transform = get_basic_transforms()

    train_ds = Cholec80ToolDataset(train_csv, transform=transform)
    val_ds   = Cholec80ToolDataset(val_csv,   transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    model = ToolCNN(num_tools=7).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # ----- train -----
        model.train()
        running_loss = 0.0
        for imgs, tool_targets in train_loader:
            imgs = imgs.to(device)
            tool_targets = tool_targets.to(device)

            logits, probs = model(imgs)    # logits used for loss
            loss = criterion(logits, tool_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ----- validate -----
        model.eval()
        val_loss = 0.0
        # later: accumulate predictions for F1/mAP
        with torch.no_grad():
            for imgs, tool_targets in val_loader:
                imgs = imgs.to(device)
                tool_targets = tool_targets.to(device)

                logits, probs = model(imgs)
                loss = criterion(logits, tool_targets)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # TODO: compute per-tool precision/recall/F1 here and save checkpoints

if __name__ == "__main__":
    print("Wire this up with real CSV paths once data is ready.")
