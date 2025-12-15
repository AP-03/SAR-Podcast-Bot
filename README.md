# SAR-Podcast-Bot

Multimodal pipeline that turns surgical video into grounded, podcast-style narration. Vision stack (ToolCNN + ActionLSTM) detects tools/phases, and a language model explains what happened using only video-derived facts.

## Quick Start
- Install deps (Python 3.9+):  
  `pip install torch torchvision torchaudio opencv-python tqdm pillow numpy matplotlib seaborn scikit-learn pyyaml transformers peft`
- Process a video and save predictions (sample every 25th frame for speed):
  ```bash
  python src/main.py --video /path/to/video.mp4 --sample-rate 25 \
    --cnn-model src/results/tool_results/tool_detection_model_best.pth \
    --lstm-model src/results/phase_results/best_lstm_attention_model.pth \
    --output results/video_predictions.npz
  ```
- Interactive Q&A on the processed video (uses grounded routing/guardrails):
  ```bash
  python src/main.py --load-npz results/video_predictions.npz \
    --interactive-qa --model-type llama  # or dummy, sota
  ```
  Commands: `/summary`, `/phases`, `/tools`, `/phase <name>`, `/context`, `/tts`, `/voice`, `/save`, `/quit`.

## Components
- Vision: `src/models/tool_resnet.py` (tool + phase multi-task CNN), `src/models/action_LSTM.py` (attention LSTM), orchestrated in `src/main.py`.
- Narration/QA: grounded routing in `src/main.py` interactive mode; text generation wrapper in `src/narration_generator.py`.
- Data & configs: `configs/labels.py`, `src/dataset/` (Cholec80 prep, Surgical_Robotics knowledge base).
- Training scripts: `src/training/train_tool.py`, `train_action.py`, `train_dummy.py` (light baseline). Core LM is a fine-tuned Llama (see `src/results/model_final/`).

## Using Existing Checkpoints
- CNN: `src/results/tool_results/tool_detection_model_best.pth`
- LSTM: `src/results/phase_results/best_lstm_attention_model.pth`
- Llama core model (weights + tokenizer): `src/results/model_final/best_model`

## Narration from NPZ
If you have a pipeline NPZ with predictions, generate a narration script:
```bash
python src/narration_generator.py --npz results/video_predictions.npz --model src/results/model_final/best_model --output podcast_script.txt
```
Add `--no-gpt` to use only the built-in knowledge base.

## Training (optional)
1) Tool/phase CNN:  
   `python src/training/train_tool.py --train_csv cholec80_train.csv --val_csv cholec80_val.csv --device cuda`
2) Feature extraction for LSTM:  
   `python src/utils/feature_extraction.py --csv_path cholec80_train.csv --ckpt_path src/results/tool_results/tool_detection_model_best.pth --out_file src/results/tool_results/cholec80_cnn_train_feats.pt`
3) Action/phase LSTM:  
   `python src/training/train_action.py --config src/hype/LSTM.yaml`
4) Core Llama:  
   Use the bundled `src/results/model_final/best_model` (fine-tuned Llama). If re-training is needed, follow your Llama finetune workflow with DailyDialog + Surgical_Robotics Q&A.

## Notes
- Grounding: Q&A answers pull tool/phase lists deterministically from the NPZ; LLM is only used to explain, with guardrails against inventing tools/phases.
- For llama or other local models, ensure the tokenizer/model files are available; for SOTA/GPT-4o, set `OPENAI_API_KEY`.
- Use higher `--sample-rate` (e.g., 25) for quick runs; lower for denser predictions if time/GPU allows.
