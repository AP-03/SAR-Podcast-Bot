# SAR-Podcast-Bot

Pipeline for turning surgical video into narrated, podcast-style output. The project couples a vision stack (multi-task CNN + action LSTM) with a GPT-2 core model fine-tuned on dialogue and surgical robotics knowledge.

## Overview
- **Frame-level perception:** `ToolCNN` (ResNet-50) jointly predicts surgical tools (multi-label) and phases (single-label) with Cholec80 data (`configs/labels.py`).
- **Temporal smoothing:** `ActionLSTMWithAttention` consumes CNN features to produce smoother phase/action timelines.
- **Language generation:** A GPT-2 model (with optional LoRA) is fine-tuned on DailyDialog + generated surgical robotics Q&A pairs to verbalize results.
- **End-to-end script:** `src/main.py` processes a video, runs CNN→LSTM (optional), and saves predictions to an `.npz` archive for downstream GPT usage.

## Repository Layout (selected)
- `src/main.py` — CLI entrypoint for video processing.
- `src/models/` — `tool_resnet.py`, `action_LSTM.py`, `GPT2.py`, SOTA baselines.
- `src/training/` — training scripts for tools (`train_tool.py`), actions (`train_action.py`), GPT-2 core (`train_Core.py`), plus dummy baselines.
- `src/utils/` — feature extraction (`feature_extraction.py`), metrics, and GPT-2 evaluation (`eval_core.py`).
- `src/dataset/` — Cholec80 utilities and README, DailyDialog copy, surgical robotics data generator and knowledge bases.
- `src/results/` — stored checkpoints, plots, and evaluation summaries (tool, phase/LSTM, GPT-2 core).
- `configs/labels.py` — canonical phase/tool label lists and index mappings.
- `notebooks/` — sanity checks and exploratory training runs.

## Setup
1) Create and activate a Python environment (3.9+ recommended).
2) Install dependencies (core set):
   ```bash
   pip install torch torchvision torchaudio opencv-python tqdm pillow numpy matplotlib seaborn scikit-learn pyyaml transformers peft
   ```
3) Ensure GPU drivers/CUDA are available if you want accelerated training/inference.

## Data Preparation
- **Cholec80:** Follow `src/dataset/cholec80/README.md` to extract frames and build a manifest CSV via `cholec80_prepare.py`. Splits are video-based to avoid leakage.
- **Robotics Q&A:** Generate instruction/response pairs from the provided knowledge bases:
  ```bash
  python src/dataset/Surgical_Robotics/data_generator.py
  ```
  This writes `robot_control_train.json`, which `train_Core.py` and `eval_core.py` consume.

## Training Workflow
The default checkpoints under `src/results/` were produced with the following steps:

1) **Train tool/phase CNN**
   ```bash
   python src/training/train_tool.py --train_csv /path/to/cholec80_train.csv --val_csv /path/to/cholec80_val.csv --device cuda
   ```
   - Outputs weights to `src/results/tool_results/tool_detection_model_best.pth` plus plots/metrics.

2) **Extract CNN features for LSTM**
   ```bash
   python src/utils/feature_extraction.py \
     --csv_path /path/to/cholec80_train.csv \
     --ckpt_path src/results/tool_results/tool_detection_model_best.pth \
     --out_file src/results/tool_results/cholec80_cnn_train_feats.pt
   ```
   Repeat for validation CSV to create `cholec80_cnn_val_feats.pt`.

3) **Train action/phase LSTM (with attention)**
   ```bash
   python src/training/train_action.py --config src/hype/LSTM.yaml
   ```
   - Saves best weights to `src/results/phase_results/best_lstm_attention_model.pth` and plots confusion matrices/learning curves.

4) **Fine-tune GPT-2 core**
   ```bash
   python src/training/train_Core.py
   ```
   - Uses `src/hype/Core.yaml` for hyperparameters and paths.
   - Best checkpoint lands in `src/results/core_results/gpt2_best_model_intial/`.

## Running the Video Pipeline
With the shipped checkpoints:
```bash
python src/main.py --video /path/to/video.mp4 \
  --cnn-model src/results/tool_results/tool_detection_model_best.pth \
  --lstm-model src/results/phase_results/best_lstm_attention_model.pth \
  --output src/results/final_predictions.npz \
  --sample-rate 1 --window-size 16 --stride 8 --device cuda
```
- Add `--skip-lstm` to export only CNN predictions.
- Output `.npz` includes per-frame tools, phases, confidences, extracted features, and (when enabled) LSTM-smoothed actions and attention weights.

## Core Model Evaluation Snapshot
Latest `src/results/core_results/evaluation_results.json`:
- Distinct-1/2: 0.67 / 0.93
- Avg response length: 34.9 tokens
- Token accuracy on validation sample: 0.658 (5,308 tokens)
- Example robotics response: tremor filtration for safer clipping/cutting, visual SLAM for preparation.

## Tips & Notes
- Label definitions live in `configs/labels.py`; keep CNN/LSTM and downstream mapping in sync.
- Training scripts assume Cholec80-style CSVs; adjust keys if your manifest differs.
- Default transforms are in `src/dataset/transform.py` (224×224, ImageNet normalization).
- GPU strongly recommended for both CNN/LSTM training and GPT-2 fine-tuning.
