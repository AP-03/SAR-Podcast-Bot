# Cholec80 Dataset - PyTorch Usage Guide

## Overview
The Cholec80 dataset has been prepared for PyTorch training. The dataset contains 80 surgical videos with frame-level annotations for:
- **Surgical phases** (7 classes): Preparation, CalotTriangleDissection, ClippingCutting, GallbladderDissection, GallbladderRetraction, CleaningCoagulation, GallbladderPackaging
- **Surgical tools** (7 binary labels): Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag

## Setup

### 1. Prepare the Dataset

Run the preparation script to create a manifest CSV from the extracted dataset:

```bash
cd src/dataset/cholec80

# Process all 80 videos
python cholec80_prepare.py --data_root "/Volumes/LaCie/cholec80 /cholec80"

# Process only first 5 videos (for testing)
python cholec80_prepare.py --data_root "/Volumes/LaCie/cholec80 /cholec80" --max_videos 5

# Save to custom location
python cholec80_prepare.py --data_root "/Volumes/LaCie/cholec80 /cholec80" \
    --output_csv "./cholec80_manifest_custom.csv"

# Combine options: 10 videos, custom output
python cholec80_prepare.py --data_root "/Volumes/LaCie/cholec80 /cholec80" \
    --max_videos 10 \
    --output_csv "./cholec80_manifest_10videos.csv"
```

**Options:**
- `--data_root`: **(Required)** Path to cholec80 directory (contains frames/, phase_annotations/, tool_annotations/)
- `--output_csv`: Custom path for manifest CSV (default: saves next to data_root as `cholec80_manifest.csv`)
- `--max_videos`: Limit number of videos to process (default: all 80 videos)

**Note:** Running the script again with the same output path will **overwrite** the existing CSV. Use `--output_csv` to keep multiple versions.

### 2. Use in PyTorch

```python
from dataset.cholec80.util_dataset import Cholec80Dataset, make_cholec80
from torchvision import transforms
from torch.utils.data import DataLoader

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Option 1: Create dataset directly
dataset = Cholec80Dataset(
    manifest_csv='cholec80_manifest.csv',
    transform=transform,
    video_ids=list(range(1, 61))  # Use videos 1-60 for training
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Option 2: Use convenience function
train_loader = make_cholec80(
    manifest_csv='cholec80_manifest.csv',
    batch_size=32,
    transform=transform,
    video_ids=list(range(1, 61)),  # Training videos
    shuffle=True,
    num_workers=4
)

# Validation loader with different videos
val_loader = make_cholec80(
    manifest_csv='cholec80_manifest.csv',
    batch_size=32,
    transform=transform,
    video_ids=list(range(61, 71)),  # Validation videos 61-70
    shuffle=False,
    num_workers=4
)
```

### 3. Using the Data

```python
# Iterate over batches
for batch in train_loader:
    images = batch['image']      # Shape: [batch_size, 3, 224, 224]
    phases = batch['phase']      # Shape: [batch_size], values: 0-6
    tools = batch['tools']       # Shape: [batch_size, 7], binary values
    
    # Your training code here
    outputs = model(images)
    loss = criterion(outputs, phases)
```

### 4. Class Weights for Imbalanced Data

```python
# Get phase class weights for weighted loss
phase_weights = dataset.get_phase_weights()
criterion = torch.nn.CrossEntropyLoss(weight=phase_weights)

# Get tool positive weights for binary classification
tool_pos_weights = dataset.get_tool_pos_weights()
criterion_tools = torch.nn.BCEWithLogitsLoss(pos_weight=tool_pos_weights)
```

## Dataset Structure

### Manifest CSV Columns
- `frame_path`: Absolute path to PNG frame
- `video_id`: Video number (1-80)
- `frame_id`: Frame number within video
- `phase`: Phase label (0-6)
- `grasper`, `bipolar`, `hook`, `scissors`, `clipper`, `irrigator`, `specimen_bag`: Binary tool presence (0 or 1)

### Dataset Split Recommendation
- Training: Videos 1-60 (75%)
- Validation: Videos 61-70 (12.5%)
- Test: Videos 71-80 (12.5%)

Note: Use video-level splits to avoid data leakage!

## Files
- `cholec80_prepare.py`: Downloads data and creates manifest
- `util_dataset.py`: PyTorch Dataset class
- `cholec80_manifest.csv`: Frame-to-annotation mapping (created by prepare script)

## Original Format Notes
- Images: 480x854 RGB PNG files
- Frame rate: 1 FPS (extracted from surgical videos)
- Phase annotations: Dense (every frame)
- Tool annotations: Sparse (every 25 frames, interpolated to all frames)
