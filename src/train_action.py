import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from models.action_LSTM import create_action_lstm
from dataset.cholec80.util_dataset import FeatureSequenceDataset, PHASE_NAMES

# --- Training and Evaluation Functions ---

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for features, labels in tqdm(dataloader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            features, labels = features.to(device), labels.to(device)
            logits, _, _ = model(features)
            loss = criterion(logits, labels)
            running_loss += loss.item() * features.size(0)
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_labels)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_results(train_losses, val_losses, train_accs, val_accs, y_true, y_pred, save_dir):
    """Generates and saves plots for loss, accuracy, and confusion matrix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss over Epochs', fontsize=16)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_title('Accuracy over Epochs', fontsize=16)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'lstm_learning_curves.png')
    plt.savefig(plot_path)
    print(f"Learning curves saved to {plot_path}")
    plt.close()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=PHASE_NAMES, yticklabels=PHASE_NAMES)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(save_dir, 'lstm_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() and not config['no_cuda'] else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_feature_file = os.path.join(base_dir, 'tool_results/cholec80_cnn_train_feats.pt')
    val_feature_file = os.path.join(base_dir, 'tool_results/cholec80_cnn_val_feats.pt')
    save_dir = os.path.join(base_dir, 'phase_results')
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = FeatureSequenceDataset(train_feature_file, sequence_length=config['sequence_length'])
    val_dataset = FeatureSequenceDataset(val_feature_file, sequence_length=config['sequence_length'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    model = create_action_lstm(num_actions=7, feature_dim=config['feature_dim'], model_type='attention',
                               hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
                               dropout=config['dropout'], bidirectional=True).to(device)
    
    print("\nModel architecture:")
    print(model)

    # Calculate class weights for imbalanced phases
    print("\nCalculating class weights for imbalanced phases...")
    all_phases_train = torch.cat([seq_info['label'].unsqueeze(0) for seq_info in train_dataset.sequences], dim=0)
    class_counts = torch.bincount(all_phases_train, minlength=7)
    total_samples_train = len(all_phases_train)
    num_classes = 7 # Assuming 7 phases based on PHASE_NAMES
    
    # Calculate inverse frequency weights
    # weight = 1.0 / (class_count + epsilon)
    # A common formula for CrossEntropyLoss is total_samples / (num_classes * class_count)
    class_weights = total_samples_train / (num_classes * class_counts.float())
    class_weights = class_weights.to(device)
    
    print(f"Phase frequencies and weights:")
    for i, name in enumerate(PHASE_NAMES):
        count = class_counts[i].item()
        weight = class_weights[i].item()
        print(f"  {name:30s}: Count={count:6d}, Weight={weight:.4f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    best_val_acc = 0.0

    print("\n--- Starting Training ---")
    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, final_preds, final_labels = validate_one_epoch(model, val_loader, criterion, device)
        
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_lstm_attention_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("\n--- Training Finished ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    print("\n--- Final Validation Metrics ---")
    report = classification_report(final_labels, final_preds, target_names=PHASE_NAMES)
    print(report)

    print("\n--- Generating Plots ---")
    plot_results(train_loss_history, val_loss_history, train_acc_history, val_acc_history, final_labels, final_preds, save_dir)


if __name__ == "__main__":
    # Load configuration from YAML file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'hype/LSTM.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("--- Configuration ---")
    for key, value in config.items():
        print(f"{key:<20}: {value}")
    print("-" * 23)

    main(config)