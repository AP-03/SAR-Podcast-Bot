import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm
import os
import argparse
from dataset.cholec80.util_dataset import Cholec80Dataset, TOOL_NAMES
from dataset.transform import get_basic_transforms
from models.tool_resnet import ToolCNN


def train_tools(train_csv, val_csv, epochs=10, batch_size=32, lr=1e-4,device="cuda"):
    print("Loading transforms...")
    train_transform = get_basic_transforms()
    val_transform   = get_basic_transforms()

    print("Loading training dataset...")
    train_ds = Cholec80Dataset(train_csv, transform=train_transform)
    print(f"✓ Loaded {len(train_ds)} training samples")
    
    print("Loading validation dataset...")
    val_ds   = Cholec80Dataset(val_csv,   transform=val_transform)
    print(f"✓ Loaded {len(val_ds)} validation samples")

    print("Creating data loaders...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("Initializing model...")
    model = ToolCNN(num_tools=7, num_stages=7).to(device)
    print("✓ Model loaded to device")
    
    # Calculate class weights for imbalanced tools
    print("\nCalculating class weights for imbalanced tools...")
    pos_counts = torch.zeros(7)
    total_samples = len(train_ds)
    
    for idx in range(len(train_ds)):
        sample = train_ds[idx]
        pos_counts += sample['tools']
    
    # pos_weight = (num_negative) / (num_positive) for each class
    # This upweights the loss for minority (rare) tools
    neg_counts = total_samples - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-5)  # avoid division by zero
    pos_weight = pos_weight.to(device)
    
    print(f"Tool frequencies:")
    for i, name in enumerate(TOOL_NAMES):
        freq = pos_counts[i].item() / total_samples * 100
        print(f"  {name:15s}: {freq:5.1f}% (weight: {pos_weight[i].item():.2f})")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    stage_criterion = nn.CrossEntropyLoss()
    lambda_stage = 1.0

    warmup_epochs = 2
    model.freeze_backbone()
    print(f"\n✓ Freezing backbone for first {warmup_epochs} epochs")

    head_lr = lr
    backbone_lr = lr * 0.1

    optimizer = optim.Adam(
        [
            {"params": model.head_parameters(),      "lr": head_lr},
            {"params": model.backbone_parameters(),  "lr": 0.0},
        ],
        weight_decay=1e-4,
    )

    print("\n✓ Setup complete, starting training...\n")

    # Track metrics for plotting
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # ----- train -----
        if epoch==warmup_epochs:
            model.unfreeze_backbone()
            print(f"\n✓ Unfreezing backbone for fine-tuning from epoch {epoch+1} onwards")
            optimizer.param_groups[1]['lr'] = backbone_lr
        model.train()
        running_loss = 0.0
        train_preds_epoch = []
        train_targets_epoch = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            imgs = batch['image'].to(device)
            tool_targets  = batch['tools'].to(device)
            # CHANGE 'phase' here if your dataset uses a different key name
            stage_targets = batch['phase'].to(device).long()

            tool_logits, stage_logits = model(imgs)

            tool_loss  = criterion(tool_logits, tool_targets)
            stage_loss = stage_criterion(stage_logits, stage_targets)
            loss = tool_loss + lambda_stage * stage_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(tool_logits)
            train_preds_epoch.append(probs.detach().cpu().numpy())
            train_targets_epoch.append(tool_targets.cpu().numpy())

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                't_loss': f"{tool_loss.item():.4f}",
                's_loss': f"{stage_loss.item():.4f}",
            })


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            probs=torch.sigmoid(tool_logits)
            train_preds_epoch.append(probs.detach().cpu().numpy())
            train_targets_epoch.append(tool_targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Calculate training F1 for this epoch
        train_preds_epoch = np.vstack(train_preds_epoch)
        train_targets_epoch = np.vstack(train_targets_epoch)
        train_binary = (train_preds_epoch > 0.5).astype(int)
        train_f1s = [f1_score(train_targets_epoch[:, i], train_binary[:, i], zero_division=0) for i in range(7)]
        train_f1 = np.mean(train_f1s)

        # ----- validate -----
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        val_stage_correct = 0
        val_stage_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            for batch in pbar:
                imgs = batch['image'].to(device)
                tool_targets  = batch['tools'].to(device)
                # CHANGE 'phase' here if your dataset uses a different key name
                stage_targets = batch['phase'].to(device).long()

                tool_logits, stage_logits = model(imgs)

                tool_loss  = criterion(tool_logits, tool_targets)
                stage_loss = stage_criterion(stage_logits, stage_targets)
                loss = tool_loss + lambda_stage * stage_loss
                val_loss += loss.item() * imgs.size(0)

                # Tool predictions for metrics
                probs = torch.sigmoid(tool_logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(tool_targets.cpu().numpy())

                # Stage accuracy (optional but useful)
                _, stage_pred = stage_logits.max(dim=1)   # [B]
                val_stage_correct += (stage_pred == stage_targets).sum().item()
                val_stage_total   += stage_targets.size(0)

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    't_loss': f"{tool_loss.item():.4f}",
                    's_loss': f"{stage_loss.item():.4f}",
                })

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        val_stage_acc = val_stage_correct / val_stage_total


        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate per-tool F1 scores
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        binary_preds = (all_preds > 0.5).astype(int)
        
        f1_per_tool = []
        for i in range(7):
            f1 = f1_score(all_targets[:, i], binary_preds[:, i], zero_division=0)
            f1_per_tool.append(f1)
        
        avg_f1 = np.mean(f1_per_tool)
        
        # Print with both train and val F1
        print(
        f"Epoch {epoch+1}/{epochs} | "
        f"train_loss={train_loss:.4f} | train_F1={train_f1:.4f} | "
        f"val_loss={val_loss:.4f} | val_F1={avg_f1:.4f} | "
        f"val_stage_acc={val_stage_acc:.4f}"
        )

    
    # After training, create visualizations
    plot_training_results(train_losses, val_losses, all_preds, all_targets, model, val_loader, device)
    
    return model

def plot_training_results(train_losses, val_losses, all_preds, all_targets, model, val_loader, device):
    """
    Create essential visualizations of training results
    """
    # Convert to binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    f1_scores = []
    precisions = []
    recalls = []
    accuracies = []
    
    for i in range(7):
        f1 = f1_score(all_targets[:, i], binary_preds[:, i], zero_division=0)
        f1_scores.append(f1)
        
        tp = np.sum((binary_preds[:, i] == 1) & (all_targets[:, i] == 1))
        fp = np.sum((binary_preds[:, i] == 1) & (all_targets[:, i] == 0))
        fn = np.sum((binary_preds[:, i] == 0) & (all_targets[:, i] == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        acc = np.mean(binary_preds[:, i] == all_targets[:, i])
        
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(acc)
    
    # Create figure with 3 key plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss curves - most important for tracking training
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Per-tool F1 scores - key performance metric
    colors = plt.cm.viridis(np.linspace(0, 1, 7))
    bars = axes[1].bar(range(7), f1_scores, color=colors, alpha=0.8)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(TOOL_NAMES, rotation=45, ha='right')
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Per-Tool F1 Scores', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Precision & Recall - important for understanding model behavior
    x = np.arange(7)
    width = 0.35
    axes[2].bar(x - width/2, precisions, width, label='Precision', alpha=0.8, color='steelblue')
    axes[2].bar(x + width/2, recalls, width, label='Recall', alpha=0.8, color='coral')
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels(TOOL_NAMES, rotation=45, ha='right')
    axes[2].set_ylabel('Score', fontsize=12)
    axes[2].set_title('Precision & Recall per Tool', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save to tool_results directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tool_results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'tool_detection_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training results saved to '{plot_path}'")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FINAL VALIDATION METRICS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Average F1 Score:     {np.mean(f1_scores):.4f}")
    print(f"  Average Precision:    {np.mean(precisions):.4f}")
    print(f"  Average Recall:       {np.mean(recalls):.4f}")
    print(f"  Average Accuracy:     {np.mean(accuracies):.4f}")
    
    print(f"\nPer-Tool Metrics:")
    print(f"{'Tool':<25} {'F1':>8} {'Precision':>12} {'Recall':>10} {'Accuracy':>10}")
    print("-" * 70)
    for i, name in enumerate(TOOL_NAMES):
        print(f"{name:<25} {f1_scores[i]:>8.4f} {precisions[i]:>12.4f} {recalls[i]:>10.4f} {accuracies[i]:>10.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import os

    # Command-line config so you can force `cuda` or `cpu` easily
    parser = argparse.ArgumentParser(description="Train tool detection model")
    parser.add_argument('--device', type=str, default=None, help="Device to use: 'cuda' or 'cpu'. If omitted, auto-detects CUDA.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    # Configuration - separate train and validation sets
    # Use absolute paths to ensure we load the correct CSVs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(script_dir, "dataset/cholec80/cholec80_manifest.csv")  # seed=42, 5 videos
    val_csv = os.path.join(script_dir, "dataset/cholec80/cholec80_val.csv")         # seed=100, 5 videos

    # Create results directory
    results_dir = os.path.join(script_dir, 'tool_results')
    os.makedirs(results_dir, exist_ok=True)

    # Training parameters (can be overridden via CLI)
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("CHOLEC80 TOOL DETECTION TRAINING")
    print("="*60)
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV:   {val_csv}")
    print(f"Epochs:    {epochs}")
    print(f"Batch:     {batch_size}")
    print(f"LR:        {lr}")
    print(f"Device:    {device}")
    print(f"Results:   {results_dir}")
    print("="*60 + "\n")
    
    # Start training
    model = train_tools(
        train_csv=train_csv,
        val_csv=val_csv,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    )
    
    # Save final model to tool_results directory
    model_path = os.path.join(results_dir, "tool_detection_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to '{model_path}'")
