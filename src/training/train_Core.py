"""
GPT-2 Core Model Training Script (VERSION 3 - PROPERLY BALANCED)
=================================================================
Critical fixes for garbage output:

1. AGGRESSIVE SURGICAL UPSAMPLING: 20x repeat = surgical dominates
2. MINIMAL DIALOG: Only 1500 examples (less noise)
3. LOWER LEARNING RATE: 3e-5 for stability  
4. BETTER GENERATION: repetition_penalty, lower temperature
5. QUALITY FILTERING: Skip very short/greeting dialog

Expected data balance:
  - Surgical: 253 × 20 = 5060 examples
  - Dialog: 1500 examples
  - Ratio: ~3.4:1 surgical:dialog (good!)

Run: python src/training/train_Core_v3.py
"""

import json
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("PEFT not available - will use full fine-tuning")


# =============================================================================
# CONFIGURATION
# =============================================================================

SEPARATOR = " [RESPONSE] "

CONFIG = {
    'base_model': 'gpt2',
    'max_length': 256,
    
    'dialog_max_samples': 1500,      # Reduced from 5000
    'surgical_upsample_factor': 20,   # 253 × 20 = 5060
    
    # Training
    'batch_size': 8,
    'learning_rate': 3e-5,  # Lower than before
    'num_epochs': 20,       # More epochs
    'gradient_accumulation_steps': 4,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'early_stopping_patience': 5,
    
    # LoRA
    'use_lora': True,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,  # Reduced dropout
    
    # Generation
    'gen_max_new_tokens': 80,
    'gen_temperature': 0.4,  # Lower = more focused
    'gen_top_p': 0.85,
    'gen_repetition_penalty': 1.3,  # Higher = less repetition
    
    # Paths
    'output_dir': 'results/core_results_v3',
    'best_model_path': 'results/core_results_v3/gpt2_best_model_v3',
    'plot_path': 'results/core_results_v3/training_curves_v3.png',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dialog_filtered(base_dir, max_samples=1500):
    """Load DailyDialog with quality filtering"""
    dialog_file = os.path.join(base_dir, 'train', 'dialogues_train.txt')
    
    if not os.path.exists(dialog_file):
        raise FileNotFoundError(f"Not found: {dialog_file}")
    
    # Words to filter out (greetings cause garbage)
    skip_patterns = [
        'bye', 'goodbye', 'see you', 'hello', 'hi there', 'hi!',
        'thank you', 'thanks', 'ok', 'okay', 'yes', 'no', 'yeah',
        'good morning', 'good evening', 'good night'
    ]
    
    pairs = []
    
    with open(dialog_file, 'r', encoding='utf-8') as f:
        for line in f:
            utterances = [u.strip() for u in line.strip().split('__eou__') if u.strip()]
            
            for i in range(len(utterances) - 1):
                inst = utterances[i]
                resp = utterances[i + 1]
                
                # Quality filters
                if len(inst) < 15 or len(resp) < 15:  # Too short
                    continue
                if len(inst) > 150 or len(resp) > 150:  # Too long
                    continue
                
                # Skip greetings
                inst_lower = inst.lower()
                if any(p in inst_lower for p in skip_patterns):
                    continue
                
                pairs.append({'instruction': inst, 'response': resp})
    
    print(f"Loaded {len(pairs)} quality dialog pairs")
    
    if len(pairs) > max_samples:
        random.shuffle(pairs)
        pairs = pairs[:max_samples]
    
    print(f"Using {len(pairs)} dialog samples")
    return pairs


def load_surgical_upsampled(filepath, factor=20):
    """Load and heavily upsample surgical data"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} unique surgical examples")
    
    # Upsample
    upsampled = data * factor
    random.shuffle(upsampled)
    
    print(f"Upsampled to {len(upsampled)} surgical examples")
    return upsampled


def format_example(inst, resp):
    """Format with separator"""
    return f"{inst}{SEPARATOR}{resp}"


class TrainingDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = []
        
        for item in data:
            inst = item.get('instruction', '')
            resp = item.get('response', '')
            if inst and resp:
                self.texts.append(format_example(inst, resp))
        
        print(f"Dataset: {len(self.texts)} examples")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        ids = enc['input_ids'].squeeze()
        mask = enc['attention_mask'].squeeze()
        labels = ids.clone()
        labels[mask == 0] = -100
        
        return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


# =============================================================================
# MODEL
# =============================================================================

def setup_model(cfg):
    print(f"Loading: {cfg['base_model']}")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'])
    model = AutoModelForCausalLM.from_pretrained(cfg['base_model'])
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    if cfg['use_lora'] and PEFT_AVAILABLE:
        print("Applying LoRA...")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg['lora_r'],
            lora_alpha=cfg['lora_alpha'],
            lora_dropout=cfg['lora_dropout'],
            target_modules=['c_attn', 'c_proj'],
            bias='none'
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    
    return model, tokenizer


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(model, loader, optim, sched, device, accum=1):
    model.train()
    total = 0
    n = 0
    
    pbar = tqdm(loader, desc="Train")
    optim.zero_grad()
    
    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = out.loss / accum
        loss.backward()
        
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
        
        total += out.loss.item()
        n += 1
        pbar.set_postfix({'loss': f'{total/n:.4f}'})
    
    return total / n


def evaluate_model(model, loader, device):
    model.eval()
    total = 0
    n = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            total += out.loss.item()
            n += 1
    
    return total / n


def generate(model, tokenizer, device, prompt, cfg):
    """Generate with better settings"""
    full_prompt = prompt + SEPARATOR
    
    inp = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=150)
    inp = {k: v.to(device) for k, v in inp.items()}
    
    with torch.no_grad():
        out = model.generate(
            inp['input_ids'],
            attention_mask=inp['attention_mask'],
            max_new_tokens=cfg['gen_max_new_tokens'],
            temperature=cfg['gen_temperature'],
            top_p=cfg['gen_top_p'],
            repetition_penalty=cfg['gen_repetition_penalty'],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Extract after separator
    sep = SEPARATOR.strip()
    if sep in text:
        resp = text.split(sep)[-1].strip()
    else:
        resp = text[len(prompt):].strip()
    
    return resp.replace('[RESPONSE]', '').strip()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("GPT-2 TRAINING v3 - BALANCED DATA")
    print("=" * 70)
    
    # Paths
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent if script_dir.name == 'training' else script_dir
    
    output_dir = src_dir / CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    model, tokenizer = setup_model(CONFIG)
    model = model.to(device)
    
    # Data
    print("\n" + "-" * 70)
    print("Loading data...")
    
    dialog_path = src_dir / 'dataset' / 'DailyDialog'
    surg_path = src_dir / 'dataset' / 'Surgical_Robotics' / 'robot_control_train_v2.json'
    if not surg_path.exists():
        surg_path = src_dir / 'dataset' / 'Surgical_Robotics' / 'robot_control_train.json'
    
    dialog = load_dialog_filtered(str(dialog_path), CONFIG['dialog_max_samples'])
    surgical = load_surgical_upsampled(str(surg_path), CONFIG['surgical_upsample_factor'])
    
    all_data = dialog + surgical
    random.shuffle(all_data)
    
    print(f"\n📊 DATA BALANCE:")
    print(f"  Dialog: {len(dialog)}")
    print(f"  Surgical: {len(surgical)}")
    print(f"  Ratio: {len(surgical)/max(1,len(dialog)):.1f}:1 surgical:dialog")
    print(f"  Total: {len(all_data)}")
    
    # Split
    split = int(len(all_data) * 0.9)
    train_ds = TrainingDataset(all_data[:split], tokenizer, CONFIG['max_length'])
    val_ds = TrainingDataset(all_data[split:], tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])
    
    # Optimizer
    optim = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    
    total_steps = len(train_loader) * CONFIG['num_epochs'] // CONFIG['gradient_accumulation_steps']
    warmup = int(total_steps * CONFIG['warmup_ratio'])
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_loss = float('inf')
    patience = 0
    train_losses, val_losses = [], []
    
    test_prompts = [
        "The vision system detects 'Preparation'. What robotic algorithm applies here?",
        "How does a computer learn?",
        "Hello!",
    ]
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        t_loss = train_one_epoch(model, train_loader, optim, sched, device, CONFIG['gradient_accumulation_steps'])
        v_loss = evaluate_model(model, val_loader, device)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        
        print(f"Train: {t_loss:.4f} | Val: {v_loss:.4f}")
        
        if v_loss < best_loss - 0.005:
            best_loss = v_loss
            patience = 0
            
            path = src_dir / CONFIG['best_model_path']
            model.save_pretrained(str(path))
            tokenizer.save_pretrained(str(path))
            print(f"✓ Saved best model (loss: {v_loss:.4f})")
        else:
            patience += 1
            print(f"No improvement ({patience}/{CONFIG['early_stopping_patience']})")
        
        if patience >= CONFIG['early_stopping_patience']:
            print("Early stopping!")
            break
        
        # Test generation every 4 epochs
        if (epoch + 1) % 4 == 0:
            print("\n📝 Samples:")
            for p in test_prompts:
                r = generate(model, tokenizer, device, p, CONFIG)
                print(f"  Q: {p[:35]}...")
                print(f"  A: {r[:70]}...")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-o', label='Train')
    plt.plot(val_losses, 'r-s', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training (Balanced Data v3)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(src_dir / CONFIG['plot_path']))
    
    # Final test
    print("\n" + "=" * 70)
    print("FINAL TEST")
    print("=" * 70)
    
    final_prompts = [
        "The vision system detects 'Preparation'. What robotic algorithm applies here?",
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        "What stages are in this surgery?",
        "How does a computer learn?",
        "Will AI take over?",
        "Hello!",
    ]
    
    for p in final_prompts:
        r = generate(model, tokenizer, device, p, CONFIG)
        print(f"\nQ: {p}")
        print(f"A: {r}")
    
    print(f"\n✓ Done! Best loss: {best_loss:.4f}")
    print(f"  Model: {CONFIG['best_model_path']}")


if __name__ == "__main__":
    main()