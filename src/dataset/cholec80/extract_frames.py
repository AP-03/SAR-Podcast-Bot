"""
Extract frames from Cholec80 videos.

This script processes video files and extracts frames at a specified rate,
organizing them into subdirectories for each video. The extracted frames
are then compatible with cholec80_prepare.py for creating dataset manifests.

Usage:
    python extract_frames.py --videos_dir ./videos --output_dir ./frames --fps 1
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames(videos_dir, output_dir, fps=1, video_pattern="video*.mp4"):
    """
    Extract frames from videos.
    
    Args:
        videos_dir (str): Path to directory containing video files
        output_dir (str): Path to output frames directory
        fps (int): Frames per second to extract (1 = every 1 second)
        video_pattern (str): Glob pattern for video files (default: "video*.mp4")
    """
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)
    
    if not videos_dir.exists():
        print(f"Error: Videos directory '{videos_dir}' not found!")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = sorted(videos_dir.glob(video_pattern))
    
    if not video_files:
        print(f"Error: No video files matching '{video_pattern}' found in '{videos_dir}'")
        return
    
    print(f"Found {len(video_files)} video(s) to process\n")
    
    total_frames_saved = 0
    
    for video_file in video_files:
        video_name = video_file.stem
        frame_dir = output_dir / video_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_file))
        
        if not cap.isOpened():
            print(f"Warning: Could not open video '{video_file}'")
            continue
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps == 0:
            print(f"Warning: Could not determine FPS for '{video_name}'")
            cap.release()
            continue
        
        frame_interval = max(1, int(video_fps / fps))
        expected_frames = total_frames // frame_interval
        
        print(f"Extracting frames from {video_name}...")
        print(f"  Video FPS: {video_fps:.2f}, Total frames: {total_frames}")
        print(f"  Extraction rate: {fps} fps, Frame interval: {frame_interval}")
        
        frame_count = 0
        saved_count = 0
        
        with tqdm(total=expected_frames, unit="frames", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_path = frame_dir / f"frame_{saved_count+1:06d}.png"
                    cv2.imwrite(str(frame_path), frame)
                    saved_count += 1
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        print(f"  ✓ Saved {saved_count} frames to {frame_dir}\n")
        total_frames_saved += saved_count
    
    print(f"=" * 60)
    print(f"Frame extraction complete!")
    print(f"Total frames saved: {total_frames_saved}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from Cholec80 videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 1 frame per second from all videos
  python extract_frames.py --videos_dir ./videos --output_dir ./frames

  # Extract 2 frames per second
  python extract_frames.py --videos_dir ./videos --output_dir ./frames --fps 2

  # Process only video01-video05
  python extract_frames.py --videos_dir ./videos --output_dir ./frames --pattern "video0[1-5].mp4"
        """
    )
    
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="./videos",
        help="Path to directory containing video files (default: ./videos)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./frames",
        help="Path to output frames directory (default: ./frames)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second to extract (default: 1)"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="video*.mp4",
        help="Glob pattern for video files (default: video*.mp4)"
    )
    
    args = parser.parse_args()
    
    if args.fps < 1:
        print("Error: fps must be >= 1")
        return
    
    extract_frames(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        video_pattern=args.pattern
    )


if __name__ == "__main__":
    main()
