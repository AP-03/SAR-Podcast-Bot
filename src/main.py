"""
Main pipeline for SAR-Podcast-Bot
Processes surgical videos through trained models to generate podcast content
"""
import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tool_resnet import ToolCNN
from models.action_LSTM import ActionLSTMWithAttention
from dataset.transform import get_basic_transforms
from configs.labels import PHASES, TOOLS


class VideoProcessor:
    """Process surgical video through trained CNN to extract frame-level predictions"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize video processor with trained CNN model
        
        Args:
            model_path: Path to trained tool detection model (.pth file)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Get transforms
        self.transform = get_basic_transforms()
        
        # Label mappings
        self.phases = PHASES
        self.tools = TOOLS
        
    def _load_model(self, model_path):
        """Load trained ToolCNN model"""
        print(f"Loading model from: {model_path}")
        
        # Initialize model architecture
        model = ToolCNN(
            num_tools=len(TOOLS),
            num_stages=len(PHASES),
            pretrained=False  # We're loading trained weights
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        print("Model loaded successfully!")
        return model
    
    def process_video(self, video_path, sample_rate=1):
        """
        Process video through CNN to get frame-level predictions
        
        Args:
            video_path: Path to input video file
            sample_rate: Process every Nth frame (default: 1 = all frames)
            
        Returns:
            results: Dict containing frame-level predictions
        """
        print(f"\nProcessing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames @ {fps:.2f} FPS")
        print(f"Sample rate: every {sample_rate} frame(s)")
        
        # Storage for results
        results = {
            'frame_indices': [],
            'timestamps': [],
            'tool_predictions': [],
            'tool_confidences': [],
            'phase_predictions': [],
            'phase_confidences': [],
            'features': []
        }
        
        frame_idx = 0
        processed_count = 0
        
        with torch.no_grad():
            pbar = tqdm(total=total_frames // sample_rate, desc="Processing frames")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_idx % sample_rate == 0:
                    # Preprocess frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                    
                    # Run model
                    tool_logits, phase_logits, features = self.model(input_tensor, return_features=True)
                    
                    # Convert to predictions
                    tool_probs = torch.sigmoid(tool_logits).cpu().numpy()[0]  # Multi-label
                    phase_probs = torch.softmax(phase_logits, dim=1).cpu().numpy()[0]  # Single-label
                    
                    # Get predictions (threshold tools at 0.5)
                    active_tools = [self.tools[i] for i, prob in enumerate(tool_probs) if prob > 0.5]
                    predicted_phase = self.phases[phase_probs.argmax()]
                    
                    # Store results
                    results['frame_indices'].append(frame_idx)
                    results['timestamps'].append(frame_idx / fps)
                    results['tool_predictions'].append(active_tools)
                    results['tool_confidences'].append(tool_probs.tolist())
                    results['phase_predictions'].append(predicted_phase)
                    results['phase_confidences'].append(phase_probs.tolist())
                    results['features'].append(features.cpu().numpy()[0])
                    
                    processed_count += 1
                    pbar.update(1)
                
                frame_idx += 1
            
            pbar.close()
        
        cap.release()
        
        print(f"\nProcessed {processed_count} frames")
        
        # Convert lists to numpy arrays
        results['features'] = np.array(results['features'])
        
        return results
    
    def save_results(self, results, output_path):
        """Save processing results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy archive
        np.savez_compressed(
            output_path,
            frame_indices=results['frame_indices'],
            timestamps=results['timestamps'],
            tool_predictions=results['tool_predictions'],
            tool_confidences=results['tool_confidences'],
            phase_predictions=results['phase_predictions'],
            phase_confidences=results['phase_confidences'],
            features=results['features']
        )
        print(f"Results saved to: {output_path}")


class ActionPredictor:
    """Process CNN features through trained LSTM to predict surgical actions"""
    
    def __init__(self, model_path, num_actions, device='cuda'):
        """
        Initialize action predictor with trained LSTM model
        
        Args:
            model_path: Path to trained LSTM model (.pth file)
            num_actions: Number of action/phase classes
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading LSTM model on device: {self.device}")
        
        # Load model
        self.model = self._load_lstm_model(model_path, num_actions)
        self.model.eval()
        
        # Action labels
        self.action_labels = PHASES  # Using phase labels as actions
        
    def _load_lstm_model(self, model_path, num_actions):
        """Load trained LSTM model"""
        print(f"Loading LSTM from: {model_path}")
        
        # Initialize model architecture
        model = ActionLSTMWithAttention(
            num_actions=num_actions,
            feature_dim=2048,
            hidden_dim=512,
            num_layers=2,
            dropout=0.5,
            bidirectional=True
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        print("LSTM model loaded successfully!")
        return model
    
    def predict_sequence(self, features, window_size=16, stride=8):
        """
        Predict actions from feature sequence using sliding window
        
        Args:
            features: numpy array of shape [num_frames, feature_dim]
            window_size: Number of frames to process at once
            stride: Number of frames to slide between windows
            
        Returns:
            predictions: List of predicted actions per window
            confidences: List of confidence scores per window
            attention_weights: List of attention weights per window
        """
        print(f"\nPredicting actions from {len(features)} feature vectors...")
        print(f"Window size: {window_size}, Stride: {stride}")
        
        num_frames = len(features)
        predictions = []
        confidences = []
        attention_weights_list = []
        window_indices = []
        
        with torch.no_grad():
            # Slide window over sequence
            for start_idx in tqdm(range(0, num_frames - window_size + 1, stride), 
                                 desc="Processing windows"):
                end_idx = start_idx + window_size
                
                # Extract window of features
                window_features = features[start_idx:end_idx]
                
                # Convert to tensor [1, window_size, feature_dim]
                window_tensor = torch.from_numpy(window_features).float().unsqueeze(0).to(self.device)
                
                # Predict
                logits, probs, attn_weights = self.model(window_tensor, return_sequence=False)
                
                # Get prediction
                pred_idx = torch.argmax(probs, dim=1).cpu().item()
                confidence = probs[0, pred_idx].cpu().item()
                
                predictions.append(self.action_labels[pred_idx])
                confidences.append(confidence)
                attention_weights_list.append(attn_weights.cpu().numpy()[0])
                window_indices.append((start_idx, end_idx))
        
        print(f"Processed {len(predictions)} windows")
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'attention_weights': attention_weights_list,
            'window_indices': window_indices
        }
    
    def aggregate_predictions(self, predictions_dict, num_frames):
        """
        Aggregate sliding window predictions to frame-level predictions
        
        Args:
            predictions_dict: Output from predict_sequence
            num_frames: Total number of frames in video
            
        Returns:
            frame_predictions: List of predicted action per frame
            frame_confidences: List of confidence per frame
        """
        # Initialize vote count for each frame
        vote_counts = [{label: 0 for label in self.action_labels} for _ in range(num_frames)]
        
        # Accumulate votes from each window
        for pred, conf, (start_idx, end_idx) in zip(
            predictions_dict['predictions'],
            predictions_dict['confidences'],
            predictions_dict['window_indices']
        ):
            for frame_idx in range(start_idx, end_idx):
                if frame_idx < num_frames:
                    vote_counts[frame_idx][pred] += conf
        
        # Get final prediction per frame
        frame_predictions = []
        frame_confidences = []
        
        for votes in vote_counts:
            if sum(votes.values()) > 0:
                pred = max(votes, key=votes.get)
                conf = votes[pred] / sum(votes.values())
            else:
                pred = self.action_labels[0]  # Default to first action
                conf = 0.0
            
            frame_predictions.append(pred)
            frame_confidences.append(conf)
        
        return frame_predictions, frame_confidences


def main():
    """Main pipeline execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process surgical video through CNN + LSTM pipeline")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--cnn-model', type=str, default='results/tool_results/tool_detection_model_best.pth',
                       help='Path to trained CNN checkpoint')
    parser.add_argument('--lstm-model', type=str, default='results/phase_results/best_lstm_attention_model.pth',
                       help='Path to trained LSTM checkpoint')
    parser.add_argument('--output', type=str, default='results/final_predictions.npz',
                       help='Output path for predictions')
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='Process every Nth frame (default: 1)')
    parser.add_argument('--window-size', type=int, default=16,
                       help='LSTM window size (number of frames)')
    parser.add_argument('--stride', type=int, default=8,
                       help='LSTM stride (frames between windows)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--skip-lstm', action='store_true',
                       help='Skip LSTM processing (CNN only)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SAR-PODCAST-BOT PIPELINE")
    print("="*60)
    
    # Step 1: Process video through CNN
    print("\n[STEP 1] Processing video through CNN...")
    cnn_processor = VideoProcessor(
        model_path=args.cnn_model,
        device=args.device
    )
    
    cnn_results = cnn_processor.process_video(
        video_path=args.video,
        sample_rate=args.sample_rate
    )
    
    # Step 2: Process features through LSTM (if not skipped)
    if not args.skip_lstm:
        print("\n[STEP 2] Processing CNN features through LSTM...")
        action_predictor = ActionPredictor(
            model_path=args.lstm_model,
            num_actions=len(PHASES),
            device=args.device
        )
        
        lstm_results = action_predictor.predict_sequence(
            features=cnn_results['features'],
            window_size=args.window_size,
            stride=args.stride
        )
        
        # Aggregate to frame-level
        frame_actions, frame_action_conf = action_predictor.aggregate_predictions(
            lstm_results, 
            len(cnn_results['frame_indices'])
        )
        
        # Add to results
        cnn_results['lstm_actions'] = frame_actions
        cnn_results['lstm_confidences'] = frame_action_conf
        cnn_results['lstm_windows'] = lstm_results
    
    # Save final results
    print("\n[STEP 3] Saving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.skip_lstm:
        np.savez_compressed(
            output_path,
            frame_indices=cnn_results['frame_indices'],
            timestamps=cnn_results['timestamps'],
            tool_predictions=cnn_results['tool_predictions'],
            tool_confidences=cnn_results['tool_confidences'],
            phase_predictions=cnn_results['phase_predictions'],
            phase_confidences=cnn_results['phase_confidences'],
            features=cnn_results['features']
        )
    else:
        np.savez_compressed(
            output_path,
            frame_indices=cnn_results['frame_indices'],
            timestamps=cnn_results['timestamps'],
            tool_predictions=cnn_results['tool_predictions'],
            tool_confidences=cnn_results['tool_confidences'],
            phase_predictions=cnn_results['phase_predictions'],
            phase_confidences=cnn_results['phase_confidences'],
            lstm_actions=cnn_results['lstm_actions'],
            lstm_confidences=cnn_results['lstm_confidences'],
            features=cnn_results['features']
        )
    
    print(f"Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Total frames processed: {len(cnn_results['frame_indices'])}")
    print(f"Video duration: {cnn_results['timestamps'][-1]:.2f}s")
    
    print(f"\n[CNN] Phase distribution:")
    from collections import Counter
    phase_counts = Counter(cnn_results['phase_predictions'])
    for phase, count in phase_counts.most_common():
        print(f"  {phase}: {count} frames ({count/len(cnn_results['phase_predictions'])*100:.1f}%)")
    
    if not args.skip_lstm:
        print(f"\n[LSTM] Action distribution:")
        action_counts = Counter(cnn_results['lstm_actions'])
        for action, count in action_counts.most_common():
            print(f"  {action}: {count} frames ({count/len(cnn_results['lstm_actions'])*100:.1f}%)")
        
        print(f"\nLSTM windows processed: {len(lstm_results['predictions'])}")
        print(f"Average confidence: {np.mean(lstm_results['confidences']):.3f}")
    
    print("="*60)
    print("\nPipeline complete! Ready for GPT-4o generation.")
    print("="*60)


if __name__ == "__main__":
    main()
