"""
Test script for vision models (CNN + LSTM) on m2cai16-workflow test dataset
Evaluates tool detection (CNN) and phase recognition (LSTM) performance
"""
import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..')
project_root = os.path.join(src_dir, '..')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from models.tool_resnet import ToolCNN
from models.action_LSTM import ActionLSTMWithAttention
from dataset.transform import get_basic_transforms
from configs.labels import PHASES, TOOLS


# Map m2cai16 test labels to training labels
# TrocarPlacement not in training set, map to Preparation (most similar)
LABEL_MAPPING = {
    "TrocarPlacement": "Preparation",
    "Preparation": "Preparation",
    "CalotTriangleDissection": "CalotTriangleDissection",
    "ClippingCutting": "ClippingCutting",
    "GallbladderDissection": "GallbladderDissection",
    "GallbladderRetraction": "GallbladderRetraction",
    "CleaningCoagulation": "CleaningCoagulation",
    "GallbladderPackaging": "GallbladderPackaging",
}


def load_ground_truth_labels(label_file, sample_rate=1):
    """
    Load ground truth phase labels from m2cai16 format and map to training labels
    
    Args:
        label_file: Path to .txt file with format "Frame\tPhase"
        sample_rate: Sample every Nth frame to match video processing
    
    Returns:
        List of phase labels per frame (mapped to training labels, sampled)
    """
    labels = []
    unmapped_labels = set()
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
        # Skip header line
        for frame_idx, line in enumerate(lines[1:]):
            # Only keep labels for sampled frames
            if frame_idx % sample_rate != 0:
                continue
                
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                original_label = parts[1]
                
                # Map to training label
                if original_label in LABEL_MAPPING:
                    labels.append(LABEL_MAPPING[original_label])
                else:
                    # Unknown label - keep original and warn
                    labels.append(original_label)
                    unmapped_labels.add(original_label)
    
    if unmapped_labels:
        print(f"  WARNING: Found unmapped labels: {unmapped_labels}")
    
    return labels


def process_video_cnn(video_path, model, transform, device, sample_rate=1):
    """
    Process video through CNN to extract features and predictions
    
    Args:
        video_path: Path to video file
        model: Trained ToolCNN model
        transform: Image transformation pipeline
        device: torch device
        sample_rate: Process every Nth frame
    
    Returns:
        Dictionary with frame-level predictions and features
    """
    print(f"\nProcessing video: {Path(video_path).name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Sample rate: every {sample_rate} frame(s)")
    
    results = {
        'frame_indices': [],
        'cnn_phase_predictions': [],
        'cnn_phase_confidences': [],
        'features': []
    }
    
    frame_idx = 0
    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=total_frames // sample_rate, desc="  CNN processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                # Preprocess
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                
                # Forward pass
                tool_logits, phase_logits, features = model(input_tensor, return_features=True)
                
                # Get phase predictions
                phase_probs = torch.softmax(phase_logits, dim=1).cpu().numpy()[0]
                predicted_phase_idx = phase_probs.argmax()
                
                results['frame_indices'].append(frame_idx)
                results['cnn_phase_predictions'].append(PHASES[predicted_phase_idx])
                results['cnn_phase_confidences'].append(phase_probs[predicted_phase_idx])
                results['features'].append(features.cpu().numpy()[0])
                
                pbar.update(1)
            
            frame_idx += 1
        
        pbar.close()
    
    cap.release()
    
    results['features'] = np.array(results['features'])
    print(f"  Extracted {len(results['frame_indices'])} frames")
    
    return results


def process_lstm(features, model, device, window_size=16, stride=8):
    """
    Process CNN features through LSTM for temporal phase recognition
    
    Args:
        features: numpy array of CNN features [num_frames, feature_dim]
        model: Trained LSTM model
        device: torch device
        window_size: Number of frames per window
        stride: Stride between windows
    
    Returns:
        Frame-level LSTM predictions
    """
    print(f"\n  LSTM processing with window_size={window_size}, stride={stride}")
    
    num_frames = len(features)
    
    # Initialize vote counting for each frame
    vote_counts = [{phase: 0.0 for phase in PHASES} for _ in range(num_frames)]
    
    model.eval()
    with torch.no_grad():
        # Sliding window
        for start_idx in tqdm(range(0, num_frames - window_size + 1, stride), 
                             desc="  LSTM windows"):
            end_idx = start_idx + window_size
            
            # Extract window
            window_features = features[start_idx:end_idx]
            window_tensor = torch.from_numpy(window_features).float().unsqueeze(0).to(device)
            
            # Predict
            logits, probs, _ = model(window_tensor, return_sequence=False)
            
            pred_idx = torch.argmax(probs, dim=1).cpu().item()
            confidence = probs[0, pred_idx].cpu().item()
            predicted_phase = PHASES[pred_idx]
            
            # Vote for all frames in window
            for frame_idx in range(start_idx, end_idx):
                vote_counts[frame_idx][predicted_phase] += confidence
    
    # Aggregate votes to get final predictions
    lstm_predictions = []
    lstm_confidences = []
    
    for votes in vote_counts:
        if sum(votes.values()) > 0:
            pred = max(votes, key=votes.get)
            conf = votes[pred] / sum(votes.values())
        else:
            pred = PHASES[0]
            conf = 0.0
        
        lstm_predictions.append(pred)
        lstm_confidences.append(conf)
    
    print(f"  Generated {len(lstm_predictions)} LSTM predictions")
    
    return lstm_predictions, lstm_confidences


def evaluate_predictions(ground_truth, predictions, model_name):
    """
    Calculate evaluation metrics
    
    Args:
        ground_truth: List of true labels
        predictions: List of predicted labels
        model_name: Name for printing (e.g., "CNN", "LSTM")
    
    Returns:
        Dictionary of metrics
    """
    print(f"  Ground truth length: {len(ground_truth)}")
    print(f"  Predictions length: {len(predictions)}")
    
    # Ensure same length (should already match if sampled correctly)
    if len(ground_truth) != len(predictions):
        print(f"  WARNING: Length mismatch! Truncating to shorter length.")
    min_len = min(len(ground_truth), len(predictions))
    ground_truth = ground_truth[:min_len]
    predictions = predictions[:min_len]
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(ground_truth, predictions, average=None, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(ground_truth, predictions, labels=PHASES)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': dict(zip(PHASES, precision_per_class)),
        'recall_per_class': dict(zip(PHASES, recall_per_class)),
        'f1_per_class': dict(zip(PHASES, f1_per_class)),
        'support_per_class': dict(zip(PHASES, support_per_class)),
        'confusion_matrix': conf_matrix
    }
    
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Print confusion matrix
    print(f"\n{model_name} Confusion Matrix:")
    print(f"  {'':>20}", end='')
    for phase in PHASES:
        print(f"{phase[:12]:>12}", end='')
    print()
    print("  " + "-" * (20 + 12 * len(PHASES)))
    
    for i, true_phase in enumerate(PHASES):
        print(f"  {true_phase:>20}", end='')
        for j in range(len(PHASES)):
            print(f"{conf_matrix[i][j]:>12}", end='')
        print()
    
    return metrics


def plot_confusion_matrix(conf_matrix, model_name, video_name, save_dir):
    """
    Plot and save confusion matrix as image
    
    Args:
        conf_matrix: Confusion matrix array
        model_name: Name of the model (for title)
        video_name: Video name (for filename)
        save_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix for better visualization
    conf_matrix_norm = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Plot
    sns.heatmap(
        conf_matrix_norm, 
        annot=conf_matrix,  # Show actual counts
        fmt='d',
        cmap='Blues',
        xticklabels=PHASES,
        yticklabels=PHASES,
        cbar_kws={'label': 'Normalized Frequency'},
        vmin=0,
        vmax=1
    )
    
    plt.title(f'{model_name} Confusion Matrix - {video_name}', fontsize=14, pad=20)
    plt.xlabel('Predicted Phase', fontsize=12)
    plt.ylabel('True Phase', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    output_path = save_dir / f'{video_name}_{model_name.lower()}_confusion_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved confusion matrix to: {output_path}")


def parse_video_nums(video_nums_str):
    """Parse video numbers from string"""
    if video_nums_str.lower() == 'all':
        return list(range(1, 15))  # 1-14
    
    video_nums = []
    for part in video_nums_str.split(','):
        part = part.strip()
        if '-' in part:
            # Range like "1-5"
            start, end = map(int, part.split('-'))
            video_nums.extend(range(start, end + 1))
        else:
            # Single number
            video_nums.append(int(part))
    
    return sorted(set(video_nums))


def evaluate_single_video(video_num, args, cnn_model, lstm_model, device, results_dir):
    """Evaluate a single video"""
    # Construct paths
    video_name = f"test_workflow_video_{video_num:02d}"
    video_path = Path(args.test_dir) / f"{video_name}.mp4"
    label_path = Path(args.test_dir) / f"{video_name}.txt"
    
    if not video_path.exists():
        print(f"  WARNING: Video not found: {video_path}")
        return None
    if not label_path.exists():
        print(f"  WARNING: Labels not found: {label_path}")
        return None
    
    print("\n" + "="*60)
    print(f"EVALUATING VIDEO {video_num:02d}")
    print("="*60)
    
    # Load ground truth
    gt_labels = load_ground_truth_labels(label_path, sample_rate=args.sample_rate)
    print(f"  Loaded {len(gt_labels)} ground truth labels (sampled at 1/{args.sample_rate})")
    
    # Process video through CNN
    print("\n  [CNN] Processing video...")
    transform = get_basic_transforms()
    cnn_results = process_video_cnn(video_path, cnn_model, transform, device, args.sample_rate)
    
    # Evaluate CNN predictions
    print("\n  [CNN] Evaluating predictions...")
    cnn_metrics = evaluate_predictions(gt_labels, cnn_results['cnn_phase_predictions'], "CNN")
    
    # Plot CNN confusion matrix
    plot_confusion_matrix(cnn_metrics['confusion_matrix'], "CNN", video_name, results_dir)
    
    # LSTM evaluation
    lstm_metrics = None
    lstm_predictions = None
    lstm_confidences = None
    
    if not args.skip_lstm:
        print("\n  [LSTM] Processing features...")
        lstm_predictions, lstm_confidences = process_lstm(
            cnn_results['features'],
            lstm_model,
            device,
            args.window_size,
            args.stride
        )
        
        print("\n  [LSTM] Evaluating predictions...")
        lstm_metrics = evaluate_predictions(gt_labels, lstm_predictions, "LSTM")
        
        # Plot LSTM confusion matrix
        plot_confusion_matrix(lstm_metrics['confusion_matrix'], "LSTM", video_name, results_dir)
        
        # Print comparison
        print(f"\n  COMPARISON:")
        print(f"    Accuracy:  CNN={cnn_metrics['accuracy']:.4f}, LSTM={lstm_metrics['accuracy']:.4f} ({(lstm_metrics['accuracy']-cnn_metrics['accuracy'])*100:+.2f}%)")
        print(f"    F1-Score:  CNN={cnn_metrics['f1']:.4f}, LSTM={lstm_metrics['f1']:.4f} ({(lstm_metrics['f1']-cnn_metrics['f1'])*100:+.2f}%)")
    
    # Save results
    output_file = results_dir / f'{video_name}_evaluation.npz'
    
    if args.skip_lstm:
        np.savez_compressed(
            output_file,
            video_name=video_name,
            frame_indices=cnn_results['frame_indices'],
            ground_truth=gt_labels,
            cnn_predictions=cnn_results['cnn_phase_predictions'],
            cnn_confidences=cnn_results['cnn_phase_confidences'],
            cnn_accuracy=cnn_metrics['accuracy'],
            cnn_f1=cnn_metrics['f1']
        )
    else:
        np.savez_compressed(
            output_file,
            video_name=video_name,
            frame_indices=cnn_results['frame_indices'],
            ground_truth=gt_labels,
            cnn_predictions=cnn_results['cnn_phase_predictions'],
            cnn_confidences=cnn_results['cnn_phase_confidences'],
            lstm_predictions=lstm_predictions,
            lstm_confidences=lstm_confidences,
            cnn_accuracy=cnn_metrics['accuracy'],
            cnn_f1=cnn_metrics['f1'],
            lstm_accuracy=lstm_metrics['accuracy'],
            lstm_f1=lstm_metrics['f1']
        )
    
    print(f"\n  Results saved to: {output_file}")
    
    return {
        'video_num': video_num,
        'video_name': video_name,
        'cnn_metrics': cnn_metrics,
        'lstm_metrics': lstm_metrics
    }


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test vision models on m2cai16 test dataset")
    parser.add_argument('--test-dir', type=str, 
                       default='/Volumes/LaCie/m2cai16-workflow/test_dataset',
                       help='Path to test dataset directory')
    parser.add_argument('--video-nums', type=str, default='1',
                       help='Video numbers to test (e.g., "1" or "1,3,5" or "1-5" or "all")')
    parser.add_argument('--cnn-model', type=str, 
                       default='results/tool_results/tool_detection_model_best.pth',
                       help='Path to trained CNN checkpoint')
    parser.add_argument('--lstm-model', type=str, 
                       default='results/phase_results/best_lstm_attention_model.pth',
                       help='Path to trained LSTM checkpoint')
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='Process every Nth frame')
    parser.add_argument('--window-size', type=int, default=16,
                       help='LSTM window size')
    parser.add_argument('--stride', type=int, default=8,
                       help='LSTM stride')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--skip-lstm', action='store_true',
                       help='Skip LSTM evaluation')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse video numbers
    video_nums = parse_video_nums(args.video_nums)
    print(f"\nTesting on {len(video_nums)} video(s): {video_nums}")
    
    # Create results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / '..' / 'results' / 'vision_test_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CNN model (once)
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    print("\n[CNN] Loading model...")
    cnn_model_path = script_dir / '..' / args.cnn_model
    
    cnn_model = ToolCNN(
        num_tools=len(TOOLS),
        num_stages=len(PHASES),
        pretrained=False
    )
    
    checkpoint = torch.load(cnn_model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        cnn_model.load_state_dict(checkpoint)
    
    cnn_model.to(device)
    print("  CNN model loaded successfully")
    
    # Load LSTM model (once)
    lstm_model = None
    if not args.skip_lstm:
        print("\n[LSTM] Loading model...")
        lstm_model_path = script_dir / '..' / args.lstm_model
        
        lstm_model = ActionLSTMWithAttention(
            num_actions=len(PHASES),
            feature_dim=2048,
            hidden_dim=128,
            num_layers=2,
            dropout=0.5,
            bidirectional=True
        )
        
        checkpoint = torch.load(lstm_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            lstm_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            lstm_model.load_state_dict(checkpoint)
        
        lstm_model.to(device)
        print("  LSTM model loaded successfully")
    
    # Evaluate each video
    all_results = []
    for video_num in video_nums:
        result = evaluate_single_video(video_num, args, cnn_model, lstm_model, device, results_dir)
        if result is not None:
            all_results.append(result)
    
    # Print summary
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("SUMMARY ACROSS ALL VIDEOS")
        print("="*60)
        
        avg_cnn_acc = np.mean([r['cnn_metrics']['accuracy'] for r in all_results])
        avg_cnn_f1 = np.mean([r['cnn_metrics']['f1'] for r in all_results])
        
        print(f"\nCNN Average Performance:")
        print(f"  Accuracy:  {avg_cnn_acc:.4f}")
        print(f"  F1-Score:  {avg_cnn_f1:.4f}")
        
        if not args.skip_lstm:
            avg_lstm_acc = np.mean([r['lstm_metrics']['accuracy'] for r in all_results])
            avg_lstm_f1 = np.mean([r['lstm_metrics']['f1'] for r in all_results])
            
            print(f"\nLSTM Average Performance:")
            print(f"  Accuracy:  {avg_lstm_acc:.4f}")
            print(f"  F1-Score:  {avg_lstm_f1:.4f}")
            
            print(f"\nImprovement (LSTM vs CNN):")
            print(f"  Accuracy:  {(avg_lstm_acc - avg_cnn_acc)*100:+.2f}%")
            print(f"  F1-Score:  {(avg_lstm_f1 - avg_cnn_f1)*100:+.2f}%")
        
        print(f"\nPer-Video Results:")
        print(f"{'Video':<10} {'CNN Acc':<10} {'CNN F1':<10}", end='')
        if not args.skip_lstm:
            print(f"{'LSTM Acc':<10} {'LSTM F1':<10}")
        else:
            print()
        print("-"*60)
        
        for r in all_results:
            print(f"{r['video_num']:<10} {r['cnn_metrics']['accuracy']:<10.4f} {r['cnn_metrics']['f1']:<10.4f}", end='')
            if not args.skip_lstm:
                print(f"{r['lstm_metrics']['accuracy']:<10.4f} {r['lstm_metrics']['f1']:<10.4f}")
            else:
                print()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
