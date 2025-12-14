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

## Running the Main Pipeline

The `src/main.py` script provides an end-to-end pipeline with vision processing (CNN + LSTM) and interactive Q&A capabilities.

### Basic Video Processing

Process a surgical video and save predictions:
```bash
python src/main.py --video /path/to/video.mp4 \
  --sample-rate 25 \
  --device cuda
```

**Key Arguments:**
- `--video` - Path to input video file
- `--sample-rate` - Process every Nth frame (e.g., 25 = 1 FPS for 25 FPS video)
- `--device` - `cuda` or `cpu`
- `--cnn-model` - Path to CNN checkpoint (default: `results/tool_results/tool_detection_model_best.pth`)
- `--lstm-model` - Path to LSTM checkpoint (default: `results/phase_results/best_lstm_attention_model.pth`)
- `--output` - Output NPZ path (default: `results/final_results/predictions.npz`)
- `--skip-lstm` - Skip LSTM processing (CNN only)

**Output:** Saves `.npz` file with frame indices, timestamps, tool predictions, phase predictions, LSTM actions, confidences, and CNN features.

### Interactive Q&A Mode

Have a conversation with the bot about the processed video:

```bash
python src/main.py --video /path/to/video.mp4 \
  --sample-rate 25 \
  --interactive-qa \
  --model-type sota \
  --device cuda
```

**Language Model Options (`--model-type`):**
- `dummy` - Simple LSTM baseline
- `core` - Fine-tuned GPT-2 with LoRA (default)
- `sota` - GPT-4o via OpenAI API (requires `OPENAI_API_KEY`)

**Interactive Commands:**
- `/summary` - Show video summary
- `/phases` - List all detected phases
- `/tools` - List all detected tools
- `/phase <name>` - Show tools used in a specific phase
- `/system` - Show system architecture and capabilities
- `/tts` - Toggle text-to-speech on/off
- `/voice` - Toggle voice input on/off
- `/save` - Save conversation to JSON
- `/quit` - Exit Q&A session

### Text-to-Speech (TTS)

Enable spoken responses (macOS only):

```bash
python src/main.py --video /path/to/video.mp4 \
  --interactive-qa \
  --enable-tts \
  --model-type sota
```

The bot will speak its responses aloud using macOS `say` command. Toggle on/off during session with `/tts`.

### Voice Input (Speech-to-Text)

Speak your questions instead of typing:

```bash
python src/main.py --video /path/to/video.mp4 \
  --interactive-qa \
  --enable-voice-input \
  --model-type sota
```

**Requirements:**
```bash
pip install openai-whisper sounddevice scipy
```

**How it works:**
- Speak your question (up to 10 seconds)
- Whisper transcribes your speech
- Bot responds with answer
- Toggle on/off during session with `/voice`

**Full Podcast Experience:**
```bash
python src/main.py --video /path/to/video.mp4 \
  --interactive-qa \
  --enable-tts \
  --enable-voice-input \
  --model-type sota
```

### Fast Q&A with Pre-computed Results

Skip video processing and load existing NPZ file for instant Q&A:

```bash
python src/main.py --load-npz results/final_results/predictions.npz \
  --interactive-qa \
  --enable-tts \
  --model-type sota
```

**Benefits:**
- ⚡ Instant startup (~5 seconds vs 6+ minutes)
- 💾 Reuse previously processed videos
- 🔄 Multiple Q&A sessions on same video
- 🎯 Test different language models quickly

**Example Workflow:**
```bash
# First time: Process video (slow, ~6 minutes)
python src/main.py --video test.mp4 --sample-rate 25

# Later: Instant Q&A sessions (fast, <5 seconds)
python src/main.py --load-npz results/final_results/predictions.npz \
  --interactive-qa --enable-tts --model-type sota
```

### Complete CLI Reference

```bash
python src/main.py [OPTIONS]

Video Processing:
  --video PATH              Input video file
  --load-npz PATH          Load pre-computed NPZ instead of processing
  --cnn-model PATH         CNN checkpoint path
  --lstm-model PATH        LSTM checkpoint path
  --output PATH            Output NPZ path
  --sample-rate N          Process every Nth frame
  --window-size N          LSTM window size (default: 16)
  --stride N               LSTM stride (default: 8)
  --device {cuda,cpu}      Device for inference
  --skip-lstm              Skip LSTM processing

Interactive Q&A:
  --interactive-qa         Enable interactive Q&A mode
  --model-type {dummy,core,sota}  Language model to use
  --lm-model-path PATH     Path to language model checkpoint
  --enable-tts             Enable text-to-speech (macOS)
  --enable-voice-input     Enable voice input (requires Whisper)
```

## Performance Metrics

Real performance on M2CAI16 test dataset (5 videos):

**CNN Phase Recognition:**
- Overall accuracy: 34.8%
- High variance across videos (4.6% to 77.1%)

**LSTM Phase Recognition:**
- Overall accuracy: 63.7%
- Improvement over CNN: +28.9%
- Best video: 91.5% accuracy

**Inference Speed (CPU):**
- CNN: ~10.5 FPS
- LSTM: 481-946 windows/second
- Total pipeline: 4-7 minutes for 40-min video at 1 FPS sampling

See `src/KnowledgeBases/models_info.json` for detailed metrics.

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
- For SOTA model, set `OPENAI_API_KEY` environment variable.
- Voice input uses Whisper "tiny" model (~39MB, good balance of speed and accuracy).
