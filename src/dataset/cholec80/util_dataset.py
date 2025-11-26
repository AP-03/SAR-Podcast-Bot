'''
PyTorch Dataset for Cholec80 Surgical Videos
Loads frames and annotations from manifest CSV file
'''

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from typing import Optional, List, Callable


# Dataset constants
_H = 480  # frame height
_W = 854  # frame width
_C = 3    # channels
_N_VIDEOS = 80

PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderRetraction",
    "CleaningCoagulation",
    "GallbladderPackaging"
]

TOOL_NAMES = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag"
]


class Cholec80Dataset(Dataset):
    """
    PyTorch Dataset for Cholec80 surgical videos
    
    Args:
        manifest_csv: Path to the manifest CSV file (created by cholec80_prepare.py)
        transform: Optional torchvision transform to apply to images
        video_ids: Optional list of video IDs to include (1-80). If None, uses all videos.
        return_video_info: If True, returns video_id and frame_id in the output dict
    
    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ...     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ... ])
        >>> dataset = Cholec80Dataset('cholec80_manifest.csv', transform=transform)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    """
    
    def __init__(
        self, 
        manifest_csv: str,
        transform: Optional[Callable] = None,
        video_ids: Optional[List[int]] = None,
        return_video_info: bool = False
    ):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.return_video_info = return_video_info
        
        # Filter by video IDs if specified
        if video_ids is not None:
            self.df = self.df[self.df['video_id'].isin(video_ids)]
            self.df = self.df.reset_index(drop=True)
        
        print(f"Loaded Cholec80 dataset:")
        print(f"  - {len(self.df)} frames")
        print(f"  - {len(self.df['video_id'].unique())} videos")
        print(f"  - Phase distribution:")
        for phase_id, count in self.df['phase'].value_counts().sort_index().items():
            if 0 <= phase_id < len(PHASE_NAMES):
                print(f"      {phase_id}: {PHASE_NAMES[phase_id]:30s} - {count:6d} frames")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image and immediately convert to numpy to release file handle
        with Image.open(row['frame_path']) as pil_img:
            img = pil_img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Get labels
        phase = torch.tensor(row['phase'], dtype=torch.long)
        
        # Tools as binary vector
        tools = torch.tensor([
            row['grasper'],
            row['bipolar'],
            row['hook'],
            row['scissors'],
            row['clipper'],
            row['irrigator'],
            row['specimen_bag']
        ], dtype=torch.float32)
        
        result = {
            'image': img,
            'phase': phase,
            'tools': tools,
        }
        
        if self.return_video_info:
            result['video_id'] = row['video_id']
            result['frame_id'] = row['frame_id']
        
        return result
    
    def get_phase_weights(self):
        """
        Calculate class weights for imbalanced phases (useful for weighted loss)
        
        Returns:
            torch.Tensor: Weight for each phase class
        """
        phase_counts = self.df['phase'].value_counts().sort_index()
        total = len(self.df)
        weights = total / (len(phase_counts) * phase_counts)
        return torch.tensor(weights.values, dtype=torch.float32)
    
    def get_tool_pos_weights(self):
        """
        Calculate positive class weights for tools (useful for BCEWithLogitsLoss)
        
        Returns:
            torch.Tensor: Positive weight for each tool (7 values)
        """
        tool_columns = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'specimen_bag']
        pos_weights = []
        
        for tool in tool_columns:
            pos_count = self.df[tool].sum()
            neg_count = len(self.df) - pos_count
            # Weight = neg_count / pos_count (to balance positive class)
            weight = neg_count / max(pos_count, 1)  # Avoid division by zero
            pos_weights.append(weight)
        
        return torch.tensor(pos_weights, dtype=torch.float32)


def make_cholec80(
    manifest_csv: str,
    batch_size: int = 32,
    transform: Optional[Callable] = None,
    video_ids: Optional[List[int]] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataloader_kwargs
):
    """
    Convenience function to create a DataLoader for Cholec80
    
    Args:
        manifest_csv: Path to manifest CSV file
        batch_size: Batch size for DataLoader
        transform: Optional image transform
        video_ids: Optional list of video IDs to use
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        **dataloader_kwargs: Additional arguments for DataLoader
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset
    
    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ... ])
        >>> train_loader = make_cholec80(
        ...     'cholec80_manifest.csv',
        ...     batch_size=32,
        ...     transform=transform,
        ...     video_ids=list(range(1, 61)),  # Videos 1-60 for training
        ...     shuffle=True
        ... )
    """
    dataset = Cholec80Dataset(manifest_csv, transform=transform, video_ids=video_ids)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        **dataloader_kwargs
    )


if __name__ == "__main__":
    # Example usage
    print("Example: Loading Cholec80 Dataset")
    print("=" * 60)
    
    try:
        from torchvision import transforms
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load dataset
        dataset = Cholec80Dataset(
            manifest_csv='cholec80_manifest.csv',
            transform=transform,
            video_ids=[1, 2, 3],  # Use only first 3 videos for demo
            return_video_info=True
        )
        
        # Test loading a sample
        sample = dataset[0]
        print(f"\nSample data:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Phase: {sample['phase'].item()} ({PHASE_NAMES[sample['phase']]})")
        print(f"  Tools present: {sample['tools']}")
        tools_present = [TOOL_NAMES[i] for i, present in enumerate(sample['tools']) if present > 0.5]
        print(f"  Tool names: {', '.join(tools_present) if tools_present else 'None'}")
        print(f"  Video ID: {sample['video_id']}, Frame ID: {sample['frame_id']}")
        
        # Create DataLoader
        dataloader = make_cholec80(
            'cholec80_manifest.csv',
            batch_size=8,
            transform=transform,
            video_ids=[1, 2, 3],
            shuffle=True,
            num_workers=0  # Set to 0 for demo
        )
        
        print(f"\nDataLoader created:")
        print(f"  Total batches: {len(dataloader)}")
        
        # Get phase weights
        phase_weights = dataset.get_phase_weights()
        print(f"\nPhase weights for loss: {phase_weights}")
        
        # Get tool weights
        tool_weights = dataset.get_tool_pos_weights()
        print(f"Tool positive weights: {tool_weights}")
        
    except FileNotFoundError:
        print("\nERROR: cholec80_manifest.csv not found!")
        print("Please run: python cholec80_prepare.py --data_rootdir <path> first")
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure you have torchvision installed: pip install torchvision")