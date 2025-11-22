'''
Cholec80 Dataset Preparation for PyTorch
Creates manifest CSV mapping frames to annotations for training
'''

import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import random

# Phase mapping
PHASE_MAPPING = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderRetraction": 4,
    "CleaningCoagulation": 5,
    "GallbladderPackaging": 6
}


def parse_phase_annotations(phase_file):
    """Parse phase annotation file and return frame->phase mapping"""
    df = pd.read_csv(phase_file, sep='\t')
    phase_dict = {}
    for _, row in df.iterrows():
        frame_id = int(row['Frame'])
        phase_name = row['Phase'].strip()
        phase_dict[frame_id] = PHASE_MAPPING[phase_name]
    return phase_dict


def parse_tool_annotations(tool_file):
    """Parse tool annotation file and return frame->tools mapping"""
    df = pd.read_csv(tool_file, sep='\t')
    tool_dict = {}
    for _, row in df.iterrows():
        frame_id = int(row['Frame'])
        tools = [
            int(row['Grasper']),
            int(row['Bipolar']),
            int(row['Hook']),
            int(row['Scissors']),
            int(row['Clipper']),
            int(row['Irrigator']),
            int(row['SpecimenBag'])
        ]
        tool_dict[frame_id] = tools
    return tool_dict


def create_manifest(data_root, output_csv, max_videos=None, seed=42):
    """
    Create a CSV manifest file with columns:
    frame_path, video_id, frame_id, phase, grasper, bipolar, hook, scissors, clipper, irrigator, specimen_bag
    
    Args:
        data_root: Path to cholec80 directory
        output_csv: Path to output CSV file
        max_videos: Maximum number of videos to process (None = all videos)
    """
    
    frames_dir = os.path.join(data_root, "frames")
    phase_dir = os.path.join(data_root, "phase_annotations")
    tool_dir = os.path.join(data_root, "tool_annotations")
    
    # Get all video directories
    video_dirs = sorted(glob(os.path.join(frames_dir, "video*")))
    
    # Limit number of videos if specified
    if max_videos is not None:
        random.seed(seed)  # Set seed for reproducibility
        video_dirs = random.sample(video_dirs, min(max_videos, len(video_dirs)))
        video_dirs = sorted(video_dirs)  # Sort for consistent processing order
        print(f"Randomly selected {len(video_dirs)} videos (seed={seed})")
        print(f"Selected video IDs: {[int(os.path.basename(v).replace('video', '')) for v in video_dirs]}")
    
    manifest_data = []
    
    print("\nCreating dataset manifest...")
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        video_name = os.path.basename(video_dir)  # e.g., "video01"
        video_num = int(video_name.replace("video", ""))  # e.g., 1
        
        # Load annotations for this video
        phase_file = os.path.join(phase_dir, f"{video_name}-phase.txt")
        tool_file = os.path.join(tool_dir, f"{video_name}-tool.txt")
        
        if not os.path.exists(phase_file) or not os.path.exists(tool_file):
            print(f"Warning: Missing annotations for {video_name}, skipping...")
            continue
        
        phase_dict = parse_phase_annotations(phase_file)
        tool_dict = parse_tool_annotations(tool_file)
        
        # Get all frames for this video
        frame_files = sorted(glob(os.path.join(video_dir, "*.png")))
        
        for frame_file in frame_files:
            # Extract frame number from filename (e.g., video01_000123.png -> 123)
            frame_basename = os.path.basename(frame_file)
            png_number = int(frame_basename.split('_')[1].replace('.png', ''))
            
            # Frames are extracted at 1 FPS from 25 FPS videos
            # PNG numbering: 1, 2, 3... corresponds to original frames: 0, 25, 50...
            # So we need to convert: original_frame_id = (png_number - 1) * 25
            original_frame_id = (png_number - 1) * 25
            
            # Get phase (all frames have phase annotations)
            phase = phase_dict.get(original_frame_id, -1)  # -1 if missing
            
            # Get tools (tool annotations are sparse, so use nearest frame)
            if original_frame_id in tool_dict:
                tools = tool_dict[original_frame_id]
            else:
                # Find nearest annotated frame
                annotated_frames = sorted(tool_dict.keys())
                if annotated_frames:
                    nearest_frame = min(annotated_frames, key=lambda x: abs(x - original_frame_id))
                    tools = tool_dict[nearest_frame]
                else:
                    tools = [0, 0, 0, 0, 0, 0, 0]  # Default: no tools
            
            # Add to manifest
            manifest_data.append({
                'frame_path': frame_file,
                'video_id': video_num,
                'frame_id': original_frame_id,  # Store original 25 FPS frame ID
                'phase': phase,
                'grasper': tools[0],
                'bipolar': tools[1],
                'hook': tools[2],
                'scissors': tools[3],
                'clipper': tools[4],
                'irrigator': tools[5],
                'specimen_bag': tools[6]
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(manifest_data)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Manifest created successfully!")
    print(f"  Total frames: {len(df)}")
    print(f"  Total videos: {len(df['video_id'].unique())}")
    print(f"  Saved to: {output_csv}")
    print(f"\n  Phase distribution:")
    for phase_id, count in df['phase'].value_counts().sort_index().items():
        phase_name = [k for k, v in PHASE_MAPPING.items() if v == phase_id][0]
        print(f"    {phase_id}: {phase_name:30s} - {count:6d} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Cholec80 dataset manifest for PyTorch')
    parser.add_argument("--data_root", required=True,
                        help="Path to cholec80 directory (contains frames/, phase_annotations/, tool_annotations/)")
    parser.add_argument("--output_csv", default=None,
                        help="Path for output manifest CSV (default: saves in cholec80 module directory)")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Maximum number of videos to process (default: all 80 videos)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for video selection (default: 42). Use different seeds for train/val/test splits")
    
    args = parser.parse_args()
    
    # Set output path - default to the directory where this script is located
    if args.output_csv:
        manifest_csv = args.output_csv
    else:
        # Save in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        manifest_csv = os.path.join(script_dir, "cholec80_manifest.csv")
    
    # Verify data exists
    if not os.path.exists(args.data_root):
        print(f"ERROR: Data directory not found: {args.data_root}")
        print("Please provide the path to the extracted cholec80 directory.")
        exit(1)
    
    required_dirs = ['frames', 'phase_annotations', 'tool_annotations']
    for dir_name in required_dirs:
        dir_path = os.path.join(args.data_root, dir_name)
        if not os.path.exists(dir_path):
            print(f"ERROR: Required directory not found: {dir_path}")
            print(f"Expected structure: {args.data_root}/{{frames,phase_annotations,tool_annotations}}")
            exit(1)
    
    print(f"Found Cholec80 data at: {args.data_root}")
    
    # Create manifest CSV
    create_manifest(args.data_root, manifest_csv, max_videos=args.max_videos, seed=args.seed)
    
    print(f"\n{'='*60}")
    print("DATASET READY FOR PYTORCH!")
    print(f"{'='*60}")
    print(f"\nTo use this dataset in PyTorch:")
    print(f"  from dataset.cholec80.util_dataset import Cholec80Dataset")
    print(f"  dataset = Cholec80Dataset('{manifest_csv}')")
    print(f"{'='*60}\n")