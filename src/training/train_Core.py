"""
GPT-2 Core Model Training Script

Trains a GPT-2 model on combined datasets:
- DailyDialog: Conversational dialog with emotions and acts
- Surgical Robotics: Robot control instruction-response pairs

Features:
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Relative path handling (works on any machine)
- Weight decay regularization
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping
- Automatic output directory creation
- Fixed token accuracy calculation for causal LM

Requirements:
- pip install transformers torch peft
"""

import json
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import yaml

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels to project root

# Load hyperparameters from YAML using relative path
config_path = os.path.join(script_dir, "../hype/Core.yaml")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at: {config_path}")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"Loaded hyperparameters from: {config_path}")

# Convert relative paths in config to absolute paths
src_dir = os.path.dirname(script_dir)  # src/ directory
config['output_dir'] = os.path.join(src_dir, config['output_dir'])
config['best_model_path'] = os.path.join(src_dir, config['best_model_path'])
config['final_model_path'] = os.path.join(src_dir, config['final_model_path'])
config['plot_path'] = os.path.join(src_dir, config['plot_path'])
config['robot_control_path'] = os.path.join(src_dir, config['robot_control_path'])

# Create output directories if they don't exist
os.makedirs(config['output_dir'], exist_ok=True)
os.makedirs(os.path.dirname(config['best_model_path']), exist_ok=True)

# Verify robot control data exists
if not os.path.exists(config['robot_control_path']):
    raise FileNotFoundError(f"Robot control data not found at: {config['robot_control_path']}")

print(f"Output directory: {config['output_dir']}")
print(f"Robot control data: {config['robot_control_path']}")

# Add models directory to path using relative path
models_dir = os.path.join(script_dir, "../models")
sys.path.insert(0, models_dir)
from GPT2 import tokenizer, model, device

# Check if model is using LoRA
try:
    from peft import PeftModel
    if isinstance(model, PeftModel):
        print("\n✓ Model is using LoRA (Low-Rank Adaptation)")
        print("  This reduces trainable parameters by ~99% and prevents overfitting!")
    else:
        print("\n⚠️  Model is using full fine-tuning (all parameters)")
except ImportError:
    print("\n⚠️  PEFT not installed. Install with: pip install peft")

######################### Daily Dialog Data Prep #################################

# Load dialog datasets from local files using relative paths
daily_dialog_path = os.path.join(script_dir, "../dataset/DailyDialog")
if not os.path.exists(daily_dialog_path):
    raise FileNotFoundError(f"DailyDialog dataset not found at: {daily_dialog_path}")

print(f"Loading DailyDialog dataset from: {daily_dialog_path}")

def load_daily_dialog_split(base_dir, train_folder='train', val_folder='validation'):
    """Load both training and validation DailyDialog data
    
    Args:
        base_dir: Base DailyDialog directory containing train/ and validation/ folders
        train_folder: Name of training subfolder (default: 'train')
        val_folder: Name of validation subfolder (default: 'validation')
    
    Returns:
        Tuple of (train_data, val_data) where each is (utterances, acts, emotions)
    """
    def load_split(folder, prefix):
        utterances = []
        acts = []
        emotions = []
        
        dialog_file = os.path.join(base_dir, folder, f'dialogues_{prefix}.txt')
        act_file = os.path.join(base_dir, folder, f'dialogues_act_{prefix}.txt')
        emotion_file = os.path.join(base_dir, folder, f'dialogues_emotion_{prefix}.txt')
        
        # Verify files exist
        for filepath in [dialog_file, act_file, emotion_file]:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Required file not found: {filepath}")
        
        with open(dialog_file, 'r', encoding='utf-8') as f:
            for line in f:
                utterances.append(line.strip().split('__eou__')[:-1])
        
        with open(act_file, 'r', encoding='utf-8') as f:
            for line in f:
                acts.append([int(a) for a in line.strip().split()])
        
        with open(emotion_file, 'r', encoding='utf-8') as f:
            for line in f:
                emotions.append([int(e) for e in line.strip().split()])
        
        return utterances, acts, emotions
    
    # Load both splits
    train_data = load_split(train_folder, 'train')
    val_data = load_split(val_folder, 'validation')
    
    return train_data, val_data

# Load both training and validation data in one call
(train_utterances, train_acts, train_emotions), (val_utterances, val_acts, val_emotions) = load_daily_dialog_split(daily_dialog_path)

print(f"Loaded {len(train_utterances)} training dialogs and {len(val_utterances)} validation dialogs")

# Flatten dialogs into instruction-response pairs with context
dialog_train_instructions = []
dialog_train_responses = []

for idx, utterances in enumerate(train_utterances):
    acts = train_acts[idx]
    emotions = train_emotions[idx]
    
    # For each dialog, create pairs: utterance[i] -> utterance[i+1]
    for i in range(len(utterances) - 1):
        # Add act and emotion as context to the instruction
        context_instruction = f"[Act: {acts[i]}] [Emotion: {emotions[i]}] {utterances[i]}"
        # Add target act and emotion to the response
        context_response = f"[Act: {acts[i+1]}] [Emotion: {emotions[i+1]}] {utterances[i + 1]}"
        
        dialog_train_instructions.append(context_instruction)
        dialog_train_responses.append(context_response)

# Same for validation
dialog_val_instructions = []
dialog_val_responses = []

for idx, utterances in enumerate(val_utterances):
    acts = val_acts[idx]
    emotions = val_emotions[idx]
    
    for i in range(len(utterances) - 1):
        context_instruction = f"[Act: {acts[i]}] [Emotion: {emotions[i]}] {utterances[i]}"
        context_response = f"[Act: {acts[i+1]}] [Emotion: {emotions[i+1]}] {utterances[i + 1]}"
        
        dialog_val_instructions.append(context_instruction)
        dialog_val_responses.append(context_response)

# Tokenize daily dialog data - combine instruction and response for causal LM
dialog_train_combined = [inst + " " + resp for inst, resp in zip(dialog_train_instructions, dialog_train_responses)]
dialog_val_combined = [inst + " " + resp for inst, resp in zip(dialog_val_instructions, dialog_val_responses)]

dialog_train_encodings = tokenizer(dialog_train_combined, padding=True, truncation=True, return_tensors='pt', max_length=config['max_length'])
dialog_val_encodings = tokenizer(dialog_val_combined, padding=True, truncation=True, return_tensors='pt', max_length=config['max_length'])

# For causal LM, input_ids and labels are the same (model learns to predict next token)
dialog_train_input_ids = dialog_train_encodings['input_ids']
dialog_train_attention_mask = dialog_train_encodings['attention_mask']
dialog_train_labels = dialog_train_encodings['input_ids'].clone()

dialog_val_input_ids = dialog_val_encodings['input_ids']
dialog_val_attention_mask = dialog_val_encodings['attention_mask']
dialog_val_labels = dialog_val_encodings['input_ids'].clone()

print(f"Daily Dialog Data:")
print(f"  Training pairs: {len(dialog_train_instructions)}")
print(f"  Validation pairs: {len(dialog_val_instructions)}")
print(f"  Input shape: {dialog_train_input_ids.shape}")
print(f"  Labels shape: {dialog_train_labels.shape}")

######################### Surgical Robotics Data Prep #################################
# Load robot control data
with open(config['robot_control_path'], 'r') as f:
    robot_control_data = json.load(f)

# Convert robot control data to tensors
instructions = [item['instruction'] for item in robot_control_data]
responses = [item['response'] for item in robot_control_data]

# Tokenize the data - combine instruction and response for causal LM
robot_combined = [inst + " " + resp for inst, resp in zip(instructions, responses)]
robot_encodings = tokenizer(robot_combined, padding=True, truncation=True, return_tensors='pt', max_length=config['max_length'])

# Create input and target tensors (same for causal LM)
input_ids = robot_encodings['input_ids']
attention_mask = robot_encodings['attention_mask']
labels = robot_encodings['input_ids'].clone()

# Split into train/val
total_samples = len(robot_control_data)
train_size = int(config['robot_train_split'] * total_samples)

# Split the data
train_instruction_ids = input_ids[:train_size]
train_attention_mask = attention_mask[:train_size]
train_response = labels[:train_size]

val_instructions_ids = input_ids[train_size:]
val_attention_mask = attention_mask[train_size:]
val_response = labels[train_size:]

print(f"Robot Control Data:")
print(f"  Total samples: {total_samples}")
print(f"  Training samples: {train_size}")
print(f"  Validation samples: {total_samples - train_size}")
print(f"  Input shape: {train_instruction_ids.shape}")
print(f"  Labels shape: {train_response.shape}")

######################### Combine Datasets with Balancing #################################
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

# Custom collate function to handle variable length sequences
def collate_fn(batch):
    """Pad sequences in a batch to the same length"""
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return input_ids_padded, attention_masks_padded, labels_padded

# Create TensorDatasets
dialog_train_dataset = TensorDataset(dialog_train_input_ids, dialog_train_attention_mask, dialog_train_labels)
dialog_val_dataset = TensorDataset(dialog_val_input_ids, dialog_val_attention_mask, dialog_val_labels)

robot_train_dataset = TensorDataset(train_instruction_ids, train_attention_mask, train_response)
robot_val_dataset = TensorDataset(val_instructions_ids, val_attention_mask, val_response)

# Balance the datasets
print(f"\nDataset Statistics (Before Balancing):")
print(f"  Dialog training samples: {len(dialog_train_dataset)}")
print(f"  Robot training samples: {len(robot_train_dataset)}")
print(f"  Imbalance ratio: {len(dialog_train_dataset) / len(robot_train_dataset):.1f}:1")

# Strategy: Oversample robotics data to balance with dialog data
# Calculate how many times to repeat robotics data
repeat_factor = max(1, len(dialog_train_dataset) // len(robot_train_dataset))
print(f"  Oversampling robotics data by {repeat_factor}x")

# Create oversampled robotics dataset
robot_train_repeated = ConcatDataset([robot_train_dataset] * repeat_factor)

# Combine both datasets
combined_train_dataset = ConcatDataset([dialog_train_dataset, robot_train_repeated])
combined_val_dataset = ConcatDataset([dialog_val_dataset, robot_val_dataset])

print(f"\nDataset Statistics (After Balancing):")
print(f"  Dialog training samples: {len(dialog_train_dataset)}")
print(f"  Robot training samples (oversampled): {len(robot_train_repeated)}")
print(f"  Total training samples: {len(combined_train_dataset)}")
print(f"  Total validation samples: {len(combined_val_dataset)}")
print(f"  New ratio: {len(dialog_train_dataset) / len(robot_train_repeated):.2f}:1")

# Create DataLoaders with custom collate function
train_loader = DataLoader(combined_train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(combined_val_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)

print(f"\nBatch Configuration:")
print(f"  Batch size: {config['batch_size']}")
print(f"  Training batches per epoch: {len(train_loader)}")

######################### Training Setup #################################
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Get regularization parameters from config with defaults
weight_decay = config.get('weight_decay', 0.01)
early_stopping_patience = config.get('early_stopping_patience', 5)
lr_scheduler_patience = config.get('lr_scheduler_patience', 2)
lr_scheduler_factor = config.get('lr_scheduler_factor', 0.5)

# Optimizer with weight decay for regularization
optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=weight_decay)

# Learning rate scheduler - reduces LR when validation loss plateaus
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=lr_scheduler_factor,
    patience=lr_scheduler_patience,
    min_lr=1e-7
)

# Move model to device
model.to(device)
model.train()

print(f"\nTraining Configuration:")
print(f"  Learning rate: {config['learning_rate']}")
print(f"  Weight decay: {weight_decay}")
print(f"  Epochs: {config['num_epochs']}")
print(f"  Device: {device}")
print(f"  Gradient accumulation steps: {config['gradient_accumulation_steps']}")
print(f"  Early stopping patience: {early_stopping_patience}")
print(f"  LR scheduler patience: {lr_scheduler_patience}")
print(f"  LR scheduler factor: {lr_scheduler_factor}")

######################### Training Loop #################################
print("\n" + "="*50)
print("Starting Training...")
print("="*50)

# Best model tracking
best_val_loss = float('inf')
best_epoch = 0
epochs_without_improvement = 0

# Lists to store losses for plotting
train_losses = []
val_losses = []

for epoch in range(config['num_epochs']):
    model.train()
    total_train_loss = 0
    train_steps = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Normalize loss for gradient accumulation
        loss = loss / config['gradient_accumulation_steps']
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_train_loss += loss.item() * config['gradient_accumulation_steps']
        train_steps += 1
        
        progress_bar.set_postfix({'loss': f"{loss.item() * config['gradient_accumulation_steps']:.4f}"})
    
    avg_train_loss = total_train_loss / train_steps
    
    ######################### Validation Loop #################################
    model.eval()
    total_val_loss = 0
    val_steps = 0
    
    print(f"\nValidating Epoch {epoch+1}...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_val_loss += loss.item()
            val_steps += 1
    
    avg_val_loss = total_val_loss / val_steps
    
    # Calculate perplexity
    train_perplexity = np.exp(avg_train_loss)
    val_perplexity = np.exp(avg_val_loss)
    
    # Store losses for plotting
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    # Update learning rate scheduler
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\nEpoch {epoch+1} Results:")
    print(f"  Average Training Loss: {avg_train_loss:.4f}")
    print(f"  Average Validation Loss: {avg_val_loss:.4f}")
    print(f"  Training Perplexity: {train_perplexity:.2f}")
    print(f"  Validation Perplexity: {val_perplexity:.2f}")
    print(f"  Current Learning Rate: {current_lr:.2e}")
    
    # Save best model and check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        print(f"  ✓ New best model! Saving to {config['best_model_path']}")
        model.save_pretrained(config['best_model_path'])
        tokenizer.save_pretrained(config['best_model_path'])
    else:
        epochs_without_improvement += 1
        print(f"  Validation loss did not improve from {best_val_loss:.4f}")
        print(f"  Epochs without improvement: {epochs_without_improvement}/{early_stopping_patience}")
        
        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n{'='*50}")
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best model was at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
            print(f"{'='*50}")
            break
    
    print("-" * 50)

######################### Save Final Model #################################
model.save_pretrained(config['final_model_path'])
tokenizer.save_pretrained(config['final_model_path'])

print(f"\n{'='*50}")
print(f"Training Complete!")
print(f"Best model saved to: {config['best_model_path']}")
print(f"Final model saved to: {config['final_model_path']}")
print(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
print(f"Total epochs trained: {len(train_losses)}")
print(f"{'='*50}")

######################### Plot Training Curves #################################
plt.figure(figsize=(10, 6))
epochs_range = range(1, config['num_epochs'] + 1)

plt.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=8)
plt.plot(epochs_range, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=8)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig(config['plot_path'], dpi=300, bbox_inches='tight')
print(f"\nTraining curves saved to: {config['plot_path']}")
plt.close()

######################### Evaluation Metrics #################################
def calculate_distinct_n(texts, n):
    """Calculate distinct-n metric for diversity"""
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)
    
    if len(all_ngrams) == 0:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

def calculate_token_accuracy(predictions, labels, pad_token_id):
    """Calculate token-level accuracy"""
    mask = labels != pad_token_id
    correct = (predictions == labels) & mask
    return correct.sum().item() / mask.sum().item()

######################### Evaluation #################################
print("\n" + "="*50)
print("Evaluating Model...")
print("="*50)

# Test with examples from both datasets
dialog_test_prompts = [
    "[Act: 1] [Emotion: 0] Hello, how are you today?",
    "[Act: 2] [Emotion: 3] I'm feeling really excited about this project!",
    "[Act: 4] [Emotion: 1] Could you help me with this problem?"
]

robotics_test_prompts = [
    "The vision system detects 'Preparation'. What robotic algorithm applies here?",
    "Explain the control theory behind robotic Dissection.",
    "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
    "Why is the robotic approach to 'Clipping/Cutting' considered safer?"
]

test_prompts = dialog_test_prompts + robotics_test_prompts

model.eval()
generated_texts = []
dialog_responses = []
robotics_responses = []

print("\n" + "="*50)
print("DIALOG RESPONSES")
print("="*50)

for i, prompt in enumerate(dialog_test_prompts):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=config['generation_max_length'],
            num_return_sequences=1,
            temperature=config['generation_temperature'],
            top_p=config['generation_top_p'],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_texts.append(response)
    dialog_responses.append(response)
    print(f"\n{i+1}. Prompt: {prompt}")
    print(f"   Response: {response}")
    print("-" * 50)

print("\n" + "="*50)
print("SURGICAL ROBOTICS RESPONSES")
print("="*50)

for i, prompt in enumerate(robotics_test_prompts):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=config['generation_max_length'],
            num_return_sequences=1,
            temperature=config['generation_temperature'],
            top_p=config['generation_top_p'],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_texts.append(response)
    robotics_responses.append(response)
    print(f"\n{i+1}. Prompt: {prompt}")
    print(f"   Response: {response}")
    print("-" * 50)

# Calculate diversity metrics
print("\n" + "="*50)
print("DIVERSITY METRICS")
print("="*50)

print("\nOverall Metrics:")
distinct_1 = calculate_distinct_n(generated_texts, 1)
distinct_2 = calculate_distinct_n(generated_texts, 2)
print(f"  Distinct-1 (unigram diversity): {distinct_1:.4f}")
print(f"  Distinct-2 (bigram diversity): {distinct_2:.4f}")
avg_length = np.mean([len(text.split()) for text in generated_texts])
print(f"  Average Response Length: {avg_length:.2f} tokens")

print("\nDialog-specific Metrics:")
dialog_distinct_1 = calculate_distinct_n(dialog_responses, 1)
dialog_distinct_2 = calculate_distinct_n(dialog_responses, 2)
dialog_avg_length = np.mean([len(text.split()) for text in dialog_responses])
print(f"  Distinct-1: {dialog_distinct_1:.4f}")
print(f"  Distinct-2: {dialog_distinct_2:.4f}")
print(f"  Average Length: {dialog_avg_length:.2f} tokens")

print("\nRobotics-specific Metrics:")
robotics_distinct_1 = calculate_distinct_n(robotics_responses, 1)
robotics_distinct_2 = calculate_distinct_n(robotics_responses, 2)
robotics_avg_length = np.mean([len(text.split()) for text in robotics_responses])
print(f"  Distinct-1: {robotics_distinct_1:.4f}")
print(f"  Distinct-2: {robotics_distinct_2:.4f}")
print(f"  Average Length: {robotics_avg_length:.2f} tokens")

# Token accuracy on validation set (sample)
print("\nCalculating Token Accuracy on Validation Sample...")
sample_size = min(config['eval_sample_size'], len(val_loader.dataset))
correct_tokens = 0
total_tokens = 0

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i * config['batch_size'] >= sample_size:
            break
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # For causal LM: shift predictions and labels
        # Model predicts next token, so logits[:, i] predicts labels[:, i+1]
        shift_logits = outputs.logits[..., :-1, :].contiguous()  # Remove last position
        shift_labels = labels[..., 1:].contiguous()              # Remove first position
        
        predictions = shift_logits.argmax(dim=-1)
        mask = shift_labels != tokenizer.pad_token_id
        correct = (predictions == shift_labels) & mask
        correct_tokens += correct.sum().item()
        total_tokens += mask.sum().item()

token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
print(f"  Token Accuracy: {token_accuracy:.4f} ({correct_tokens}/{total_tokens})")

######################### Save Evaluation Results #################################
results_path = os.path.join(config['output_dir'], "evaluation_results.json")

# Compile all results
evaluation_results = {
    "overall_metrics": {
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
        "avg_response_length": float(avg_length),
        "token_accuracy": float(token_accuracy),
        "total_tokens_evaluated": int(total_tokens)
    },
    "dialog_metrics": {
        "distinct_1": float(dialog_distinct_1),
        "distinct_2": float(dialog_distinct_2),
        "avg_response_length": float(dialog_avg_length),
        "num_test_prompts": len(dialog_test_prompts)
    },
    "robotics_metrics": {
        "distinct_1": float(robotics_distinct_1),
        "distinct_2": float(robotics_distinct_2),
        "avg_response_length": float(robotics_avg_length),
        "num_test_prompts": len(robotics_test_prompts)
    },
    "test_examples": {
        "dialog_prompts": dialog_test_prompts,
        "dialog_responses": dialog_responses,
        "robotics_prompts": robotics_test_prompts,
        "robotics_responses": robotics_responses
    }
}

# Save to JSON
with open(results_path, 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print(f"\nEvaluation results saved to: {results_path}")

# Also save a human-readable text summary
summary_path = os.path.join(config['output_dir'], "evaluation_summary.txt")
with open(summary_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("GPT-2 CORE MODEL EVALUATION SUMMARY\n")
    f.write("="*60 + "\n\n")
    
    f.write("OVERALL METRICS\n")
    f.write("-"*60 + "\n")
    f.write(f"Distinct-1 (unigram diversity): {distinct_1:.4f}\n")
    f.write(f"Distinct-2 (bigram diversity): {distinct_2:.4f}\n")
    f.write(f"Average Response Length: {avg_length:.2f} tokens\n")
    f.write(f"Token Accuracy: {token_accuracy:.4f}\n\n")
    
    f.write("DIALOG METRICS\n")
    f.write("-"*60 + "\n")
    f.write(f"Distinct-1: {dialog_distinct_1:.4f}\n")
    f.write(f"Distinct-2: {dialog_distinct_2:.4f}\n")
    f.write(f"Average Length: {dialog_avg_length:.2f} tokens\n\n")
    
    f.write("SURGICAL ROBOTICS METRICS\n")
    f.write("-"*60 + "\n")
    f.write(f"Distinct-1: {robotics_distinct_1:.4f}\n")
    f.write(f"Distinct-2: {robotics_distinct_2:.4f}\n")
    f.write(f"Average Length: {robotics_avg_length:.2f} tokens\n\n")
    
    f.write("="*60 + "\n")
    f.write("SAMPLE RESPONSES\n")
    f.write("="*60 + "\n\n")
    
    f.write("DIALOG EXAMPLES:\n")
    f.write("-"*60 + "\n")
    for i, (prompt, response) in enumerate(zip(dialog_test_prompts, dialog_responses)):
        f.write(f"\n{i+1}. Prompt: {prompt}\n")
        f.write(f"   Response: {response}\n")
    
    f.write("\n" + "-"*60 + "\n")
    f.write("SURGICAL ROBOTICS EXAMPLES:\n")
    f.write("-"*60 + "\n")
    for i, (prompt, response) in enumerate(zip(robotics_test_prompts, robotics_responses)):
        f.write(f"\n{i+1}. Prompt: {prompt}\n")
        f.write(f"   Response: {response}\n")

print(f"Human-readable summary saved to: {summary_path}")

print("\n" + "="*50)
print("Evaluation Complete!")
print("="*50)



