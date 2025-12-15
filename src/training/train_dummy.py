"""
Dummy LSTM Training - ULTRA MEMORY EFFICIENT
=============================================
For GPUs with 8GB or less VRAM.
"""

import json
import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ========================================
# FORCE MEMORY CLEANUP
# ========================================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ========================================
# CONFIGURATION - ULTRA LOW MEMORY
# ========================================
config = {
    'max_length': 64,          # REDUCED from 128
    'batch_size': 1,           # MINIMUM
    'gradient_accumulation': 8, # Effective batch = 8
    'learning_rate': 0.01,
    'num_epochs': 10,
    'embed_dim': 128,          # REDUCED from 256
    'hidden_dim': 256,         # REDUCED from 512
    'num_layers': 1,           # REDUCED from 2
    'dropout': 0.3,
    'max_dialog_samples': 2000, # REDUCED from 5000
    'robot_train_split': 0.7,
}

print("=" * 60)
print("DUMMY LSTM TRAINING (Ultra Low Memory)")
print("=" * 60)
print(f"Max length: {config['max_length']}")
print(f"Batch size: {config['batch_size']}")
print(f"Model: {config['embed_dim']}embed, {config['hidden_dim']}hidden, {config['num_layers']}layer")

# ========================================
# PATHS
# ========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
output_dir = os.path.join(src_dir, 'results/dummy_results')
os.makedirs(output_dir, exist_ok=True)

daily_dialog_path = os.path.join(src_dir, "dataset/DailyDialog")
robot_control_path = os.path.join(src_dir, "dataset/Surgical_Robotics/robot_control.json")

# ========================================
# TOKENIZER
# ========================================
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ========================================
# SMALLER MODEL
# ========================================
class TinyLSTM(nn.Module):
    """Tiny LSTM for low memory"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, dropout=0.3, pad_token_id=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id if self.pad_token_id else -100)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        return type('Out', (), {'loss': loss, 'logits': logits})()
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_p=0.9, eos_token_id=None, pad_token_id=None):
        self.eval()
        with torch.no_grad():
            generated = input_ids
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(generated)
                next_logits = outputs.logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id and (next_token == eos_token_id).all():
                    break
        return generated
    
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        config = {'vocab_size': self.vocab_size, 'embed_dim': self.embed_dim, 
                  'hidden_dim': self.hidden_dim, 'pad_token_id': self.pad_token_id}
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyLSTM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=config['embed_dim'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    dropout=config['dropout'],
    pad_token_id=tokenizer.pad_token_id
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n✓ TinyLSTM: {total_params:,} parameters on {device}")

clear_memory()

# ========================================
# LOAD DATA
# ========================================
print("\nLoading data...")

def load_daily_dialog(base_dir, max_samples=2000):
    train_file = os.path.join(base_dir, 'train', 'dialogues_train.txt')
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    pairs = []
    for line in lines:
        utterances = line.strip().split('__eou__')
        utterances = [u.strip() for u in utterances if u.strip()]
        for i in range(len(utterances) - 1):
            pairs.append(f"{utterances[i]} {utterances[i + 1]}")
    
    if len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)
    return pairs

dialog_texts = load_daily_dialog(daily_dialog_path, config['max_dialog_samples'])
print(f"  Dialog: {len(dialog_texts)} samples")

with open(robot_control_path, 'r') as f:
    robot_data = json.load(f)
robot_texts = [f"{item['instruction']} {item['response']}" for item in robot_data]
print(f"  Robotics: {len(robot_texts)} samples")

# ========================================
# TOKENIZE
# ========================================
print("\nTokenizing...")

all_texts = dialog_texts + robot_texts
all_encodings = tokenizer(
    all_texts,
    padding='max_length',
    truncation=True,
    max_length=config['max_length'],
    return_tensors='pt'
)

all_ids = all_encodings['input_ids']
all_mask = all_encodings['attention_mask']
print(f"  Shape: {all_ids.shape}")

# Split
dialog_ids = all_ids[:len(dialog_texts)]
dialog_mask = all_mask[:len(dialog_texts)]
robot_ids = all_ids[len(dialog_texts):]
robot_mask = all_mask[len(dialog_texts):]

# Free memory
del all_encodings, all_ids, all_mask
clear_memory()

# ========================================
# DATASETS
# ========================================
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

dialog_train_size = int(0.9 * len(dialog_ids))
dialog_train = TensorDataset(dialog_ids[:dialog_train_size], dialog_mask[:dialog_train_size], dialog_ids[:dialog_train_size].clone())
dialog_val = TensorDataset(dialog_ids[dialog_train_size:], dialog_mask[dialog_train_size:], dialog_ids[dialog_train_size:].clone())

robot_train_size = int(config['robot_train_split'] * len(robot_ids))
robot_train = TensorDataset(robot_ids[:robot_train_size], robot_mask[:robot_train_size], robot_ids[:robot_train_size].clone())
robot_val = TensorDataset(robot_ids[robot_train_size:], robot_mask[robot_train_size:], robot_ids[robot_train_size:].clone())

# Smaller upsample
upsample = min(50, max(1, len(dialog_train) // max(1, len(robot_train))))
robot_train_up = ConcatDataset([robot_train] * upsample)

train_dataset = ConcatDataset([dialog_train, robot_train_up])
val_dataset = ConcatDataset([dialog_val, robot_val])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], pin_memory=False)

print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val")
print(f"Batches per epoch: {len(train_loader)}")

clear_memory()

# ========================================
# TRAINING
# ========================================
from torch.optim import SGD
from tqdm import tqdm

optimizer = SGD(model.parameters(), lr=config['learning_rate'])

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(config['num_epochs']):
    clear_memory()
    
    # Train
    model.train()
    total_loss = 0
    steps = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)
        labels = batch[2].to(device, non_blocking=True)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / config['gradient_accumulation']
        loss.backward()
        
        if (batch_idx + 1) % config['gradient_accumulation'] == 0:
            optimizer.step()
            optimizer.zero_grad()
            clear_memory()  # Clear after each optimizer step
        
        total_loss += outputs.loss.item()
        steps += 1
        pbar.set_postfix({'loss': f"{outputs.loss.item():.4f}"})
        
        # Explicit cleanup
        del input_ids, attention_mask, labels, outputs, loss
    
    avg_train_loss = total_loss / steps
    train_losses.append(avg_train_loss)
    
    clear_memory()
    
    # Validate
    model.eval()
    val_loss = 0
    val_steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            val_steps += 1
            
            del input_ids, attention_mask, labels, outputs
    
    avg_val_loss = val_loss / val_steps
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, PPL={np.exp(avg_val_loss):.2f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(os.path.join(output_dir, 'best_model'))
        tokenizer.save_pretrained(os.path.join(output_dir, 'best_model'))
        print(f"  ✓ Saved best model")
    
    clear_memory()

# Save final
model.save_pretrained(os.path.join(output_dir, 'final_model'))
tokenizer.save_pretrained(os.path.join(output_dir, 'final_model'))

# ========================================
# PLOT
# ========================================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, 'b-o', label='Train')
plt.plot(val_losses, 'r-s', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Dummy LSTM Training (Baseline)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'training_curves.png'))
plt.close()
print(f"\nPlot saved")

# ========================================
# TEST
# ========================================
print("\n" + "=" * 60)
print("SAMPLE OUTPUTS (Expected to be poor!)")
print("=" * 60)

test_prompts = [
    "Hello, how are you?",
    "What is the Preparation phase?",
    "How does a computer learn?",
]

model.eval()
for prompt in test_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=60, temperature=0.7, eos_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nQ: {prompt}")
    print(f"A: {output}")

# ========================================
# METRICS
# ========================================
metrics = {
    'final_train_loss': float(train_losses[-1]),
    'final_val_loss': float(val_losses[-1]),
    'best_val_loss': float(best_val_loss),
    'final_perplexity': float(np.exp(val_losses[-1])),
    'epochs': config['num_epochs'],
    'parameters': total_params,
}

with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print("✓ COMPLETE")
print(f"  Best val loss: {best_val_loss:.4f}")
print(f"  Perplexity: {np.exp(best_val_loss):.2f}")
print("=" * 60)