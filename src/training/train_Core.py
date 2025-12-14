"""
GPT-2 Core Model Training Script (IMPROVED VERSION)
====================================================
Fixes for the original training issues:

1. CLEAR SEPARATOR: Uses " [RESPONSE] " between instruction and response
2. BALANCED DATA: Better ratio between dialog and surgical data
3. EXPANDED DATASET: Uses robot_control_train_v2.json with 500+ examples
4. BETTER TOKENIZATION: Proper handling of special tokens
5. IMPROVED GENERATION: Response extraction using separator

Run from SAR-Podcast-Bot directory:
    python src/training/train_Core_v2.py
"""

import json
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path

# Suppress tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT not installed. Install with: pip install peft")
    PEFT_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Key improvement: Clear separator token
SEPARATOR = " [RESPONSE] "

CONFIG = {
    # Model
    'base_model': 'gpt2',
    
    # Data
    'max_length': 512,
    'dialog_subsample': 5000,  # Subsample dialog data for balance
    
    # Training
    'batch_size': 8,
    'learning_rate': 2e-4,  # Slightly lower for stability
    'num_epochs': 10,
    'gradient_accumulation_steps': 4,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    
    # Early stopping
    'early_stopping_patience': 3,
    
    # LoRA
    'use_lora': True,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    
    # Generation
    'generation_max_length': 200,
    'generation_temperature': 0.7,
    
    # Paths (relative to src/)
    'output_dir': 'results/core_results_v2',
    'best_model_path': 'results/core_results_v2/gpt2_best_model_v2',
    'plot_path': 'results/core_results_v2/training_curves.png',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_daily_dialog(base_dir, max_samples=None):
    """Load DailyDialog training data"""
    train_dir = os.path.join(base_dir, 'train')
    
    dialog_file = os.path.join(train_dir, 'dialogues_train.txt')
    act_file = os.path.join(train_dir, 'dialogues_act_train.txt')
    emotion_file = os.path.join(train_dir, 'dialogues_emotion_train.txt')
    
    # Check files exist
    for f in [dialog_file, act_file, emotion_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}")
    
    pairs = []
    
    with open(dialog_file, 'r', encoding='utf-8') as df, \
         open(act_file, 'r', encoding='utf-8') as af, \
         open(emotion_file, 'r', encoding='utf-8') as ef:
        
        for dialog_line, act_line, emotion_line in zip(df, af, ef):
            utterances = [u.strip() for u in dialog_line.strip().split('__eou__') if u.strip()]
            acts = act_line.strip().split()
            emotions = emotion_line.strip().split()
            
            # Create pairs from consecutive utterances
            for i in range(len(utterances) - 1):
                if i < len(acts) and i < len(emotions):
                    instruction = utterances[i]
                    response = utterances[i + 1]
                    
                    # Skip very short exchanges
                    if len(instruction) > 5 and len(response) > 5:
                        pairs.append({
                            'instruction': instruction,
                            'response': response
                        })
    
    print(f"Loaded {len(pairs)} dialog pairs")
    
    # Subsample if needed
    if max_samples and len(pairs) > max_samples:
        import random
        random.shuffle(pairs)
        pairs = pairs[:max_samples]
        print(f"Subsampled to {len(pairs)} pairs")
    
    return pairs


def load_surgical_data(filepath):
    """Load surgical robotics Q&A data"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Surgical data not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} surgical examples")
    return data


def format_training_example(instruction, response, separator=SEPARATOR):
    """
    Format a training example with clear separator.
    
    Format: {instruction} [RESPONSE] {response}
    
    This makes it clear where the response starts, fixing the fragment issue.
    """
    return f"{instruction}{separator}{response}"


class CombinedDataset(Dataset):
    """Dataset combining dialog and surgical data"""
    
    def __init__(self, dialog_data, surgical_data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Combine all data
        self.examples = []
        
        # Add dialog data
        for item in dialog_data:
            text = format_training_example(item['instruction'], item['response'])
            self.examples.append(text)
        
        # Add surgical data (might have 'instruction'/'response' keys)
        for item in surgical_data:
            inst = item.get('instruction', item.get('prompt', ''))
            resp = item.get('response', item.get('answer', ''))
            if inst and resp:
                text = format_training_example(inst, resp)
                self.examples.append(text)
        
        print(f"Total training examples: {len(self.examples)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For causal LM, labels = input_ids (shifted internally by model)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model(config):
    """Initialize model with optional LoRA"""
    print(f"\nLoading base model: {config['base_model']}")
    
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    model = AutoModelForCausalLM.from_pretrained(config['base_model'])
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Add special token for response separator (optional but helpful)
    # tokenizer.add_special_tokens({'additional_special_tokens': ['[RESPONSE]']})
    # model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA
    if config['use_lora'] and PEFT_AVAILABLE:
        print("Applying LoRA...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=['c_attn', 'c_proj'],  # GPT-2 attention modules
            bias='none'
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += outputs.loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def generate_response(model, tokenizer, device, prompt, max_length=200, temperature=0.7):
    """Generate a response using the separator format"""
    # Add the separator to the prompt
    full_prompt = prompt + SEPARATOR
    
    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response after separator
    if SEPARATOR.strip() in full_text:
        response = full_text.split(SEPARATOR.strip())[-1].strip()
    else:
        response = full_text[len(prompt):].strip()
    
    return response


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    print("=" * 60)
    print("GPT-2 CORE MODEL TRAINING (IMPROVED VERSION)")
    print("=" * 60)
    print(f"Separator token: '{SEPARATOR}'")
    
    # Setup paths
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent if script_dir.name == 'training' else script_dir
    
    # Create output directory
    output_dir = src_dir / CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = setup_model(CONFIG)
    model = model.to(device)
    
    # Load data
    print("\n" + "-" * 60)
    print("Loading data...")
    
    dialog_path = src_dir / 'dataset' / 'DailyDialog'
    surgical_path = src_dir / 'dataset' / 'Surgical_Robotics' / 'robot_control_train_v2.json'
    
    # Check for v2 data, fall back to v1
    if not surgical_path.exists():
        surgical_path = src_dir / 'dataset' / 'Surgical_Robotics' / 'robot_control_train.json'
        print(f"Note: Using original surgical data (run data_generator_v2.py to create expanded dataset)")
    
    dialog_data = load_daily_dialog(str(dialog_path), max_samples=CONFIG['dialog_subsample'])
    surgical_data = load_surgical_data(str(surgical_path))
    
    print(f"\nData balance:")
    print(f"  Dialog: {len(dialog_data)} examples")
    print(f"  Surgical: {len(surgical_data)} examples")
    print(f"  Ratio: {len(dialog_data)/len(surgical_data):.1f}:1")
    
    # Create dataset and split
    all_data = dialog_data + surgical_data
    import random
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Create datasets
    train_dataset = CombinedDataset(
        [d for d in train_data if d in dialog_data],
        [d for d in train_data if d in surgical_data],
        tokenizer, 
        CONFIG['max_length']
    )
    
    # Simpler: just split all_data
    train_dataset = CombinedDataset([], train_data, tokenizer, CONFIG['max_length'])
    val_dataset = CombinedDataset([], val_data, tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * CONFIG['num_epochs'] // CONFIG['gradient_accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print("-" * 40)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            CONFIG['gradient_accumulation_steps']
        )
        train_losses.append(train_loss)
        
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = src_dir / CONFIG['best_model_path']
            model.save_pretrained(str(best_model_path))
            tokenizer.save_pretrained(str(best_model_path))
            print(f"✓ New best model saved! (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")
        
        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
        
        # Test generation
        print("\nSample generations:")
        test_prompts = [
            "The vision system detects 'Preparation'. What robotic algorithm applies here?",
            "How does a computer learn?",
            "Hello!",
        ]
        
        for prompt in test_prompts:
            response = generate_response(model, tokenizer, device, prompt)
            print(f"  Q: {prompt[:50]}...")
            print(f"  A: {response[:100]}...")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves (Improved)')
    plt.legend()
    plt.savefig(str(src_dir / CONFIG['plot_path']))
    print(f"\nPlot saved to: {CONFIG['plot_path']}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    test_prompts = [
        # Surgical (should work well)
        "The vision system detects 'Preparation'. What robotic algorithm applies here?",
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        "What stages are in this surgery?",
        
        # AI literacy (should now work)
        "How does a computer learn?",
        "What is a neural network?",
        
        # Ethics (should now work)
        "Will AI take over?",
        
        # Conversational (should not give surgical response)
        "Hello!",
        "How are you?",
    ]
    
    print("\nTest Responses:")
    for prompt in test_prompts:
        response = generate_response(model, tokenizer, device, prompt)
        print(f"\nQ: {prompt}")
        print(f"A: {response}")
    
    # Save config
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Best model: {CONFIG['best_model_path']}")
    print(f"  Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()