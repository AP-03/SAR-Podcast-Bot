import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Assuming the model is in the models directory sibling to src
from models.action_LSTM import create_action_lstm

# --- 1. Dataset Class ---
# This is a placeholder. You will need to adapt it to your specific data structure.
class SurgicalActionDataset(Dataset):
    """
    Custom Dataset for loading surgical action sequences from pre-extracted features.

    Args:
        annotations_file (str): Path to the CSV file (e.g., cholec80_manifest.csv).
        features_dir (str): Directory where frame-wise features are stored.
        sequence_length (int): The length of the sequences to be fed into the LSTM.
        num_classes (int): The number of phase/action classes.
    """
    def __init__(self, annotations_file, features_dir, sequence_length=16, num_classes=7):
        self.features_dir = features_dir
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Load annotations and create sequences
        self.annotations = pd.read_csv(annotations_file)
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        # Group by video to create sequences within each video
        for video_id, group in self.annotations.groupby('video_id'):
            video_frames = group.sort_values('frame_id')
            
            # Create overlapping sequences
            for i in range(len(video_frames) - self.sequence_length + 1):
                start_idx = video_frames.index[i]
                end_idx = video_frames.index[i + self.sequence_length - 1]
                
                # The label for the sequence is the phase of the LAST frame
                label = video_frames.loc[end_idx, 'phase']
                
                # Get the frame IDs for the full sequence
                frame_ids = video_frames.iloc[i : i + self.sequence_length]['frame_id'].tolist()
                
                sequences.append({
                    'video_id': video_id,
                    'frame_ids': frame_ids,
                    'label': label
                })
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        video_id = sequence_info['video_id']
        frame_ids = sequence_info['frame_ids']
        label = sequence_info['label']
        
        # Load the sequence of features
        feature_sequence = []
        for frame_id in frame_ids:
            # Construct the path to the feature file for each frame
            feature_path = os.path.join(
                self.features_dir,
                f"video_{video_id:02d}",
                f"frame_{frame_id:06d}.npy"
            )
            try:
                feature = np.load(feature_path)
                feature_sequence.append(feature)
            except FileNotFoundError:
                print(f"Warning: Feature file not found at {feature_path}. Using zeros.")
                # Use a zero vector if a feature is missing
                # The feature_dim needs to be known here; assuming 2048 from ResNet
                feature_dim = 2048 
                feature_sequence.append(np.zeros(feature_dim))
        
        # Stack features into a single tensor
        features_tensor = torch.from_numpy(np.array(feature_sequence)).float()
        
        # Ensure the label is a tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features_tensor, label_tensor


# --- 2. Training and Evaluation Functions ---

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for features, labels in tqdm(dataloader, desc="Training"):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, _, _ = model(features) # Assuming attention model
        loss = criterion(logits, labels)

        # Backward pass and optimization
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
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            features, labels = features.to(device), labels.to(device)

            logits, _, _ = model(features) # Assuming attention model
            loss = criterion(logits, labels)

            running_loss += loss.item() * features.size(0)

            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


# --- 3. Main Execution ---

def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Create Datasets and DataLoaders
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # All Cholec80 datasets have 7 phase classes
    num_actions = 7
    print(f"Number of action classes: {num_actions}")

    train_dataset = SurgicalActionDataset(
        annotations_file=os.path.join(script_dir, "dataset/cholec80/cholec80_manifest.csv"),
        features_dir=args.features_dir,
        sequence_length=args.sequence_length,
        num_classes=num_actions
    )
    val_dataset = SurgicalActionDataset(
        annotations_file=os.path.join(script_dir, "dataset/cholec80/cholec80_val.csv"),
        features_dir=args.features_dir,
        sequence_length=args.sequence_length,
        num_classes=num_actions
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = create_action_lstm(
        num_actions=num_actions,
        feature_dim=args.feature_dim,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True
    ).to(device)
    
    print("Model architecture:")
    print(model)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_action_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("\n--- Training Finished ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM for surgical action recognition.")
    
    # Data and Model paths
    parser.add_argument('--features_dir', type=str, default='../data/features', help='Directory where CNN features are stored.')
    parser.add_argument('--train_annotations', type=str, default='../data/train_annotations.csv', help='Path to training annotations CSV file.')
    parser.add_argument('--val_annotations', type=str, default='../data/val_annotations.csv', help='Path to validation annotations CSV file.')
    parser.add_argument('--save_dir', type=str, default='../models/saved', help='Directory to save the best model.')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='attention', choices=['standard', 'attention'], help='Type of LSTM model to use.')
    parser.add_argument('--feature_dim', type=int, default=2048, help='Dimension of the input features from the CNN.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of the LSTM.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for.')
    parser.add_argument('--sequence_length', type=int, default=16, help='Length of the feature sequences.')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA training.')

    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    main(args)

