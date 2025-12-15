"""
Evaluation script for Dummy LSTM model
Generates evaluation_results.json with same metrics as Core and SOTA
"""
import os
import sys
import json
import random
import time
from tqdm import tqdm
import torch
import numpy as np

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

from transformers import AutoTokenizer

# Try to import BERTScore
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert-score not installed. Install with: pip install bert-score")
    BERTSCORE_AVAILABLE = False

# ========================================
# PATHS
# ========================================
dummy_model_path = os.path.join(src_dir, 'results/dummy_results/best_model')
output_path = os.path.join(src_dir, 'results/dummy_results/evaluation_results.json')
summary_path = os.path.join(src_dir, 'results/dummy_results/evaluation_summary.txt')
daily_dialog_path = os.path.join(src_dir, 'dataset/DailyDialog')
robot_control_path = os.path.join(src_dir, 'dataset/Surgical_Robotics/robot_control.json')

# ========================================
# LOAD MODEL
# ========================================
print("=" * 60)
print("DUMMY LSTM MODEL EVALUATION")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Model path: {dummy_model_path}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(dummy_model_path)

# Load the model
class TinyLSTM(torch.nn.Module):
    """TinyLSTM for inference"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, dropout=0.3, pad_token_id=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id if self.pad_token_id else -100)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        return type('Out', (), {'loss': loss, 'logits': logits})()
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_p=0.9, eos_token_id=None, pad_token_id=None):
        self.eval()
        with torch.no_grad():
            generated = input_ids
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(generated)
                next_logits = outputs.logits[:, -1, :] / max(temperature, 0.1)
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id and (next_token == eos_token_id).all():
                    break
        return generated

# Load model config and weights
config_path = os.path.join(dummy_model_path, 'config.json')
weights_path = os.path.join(dummy_model_path, 'pytorch_model.bin')

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    model = TinyLSTM(
        vocab_size=model_config.get('vocab_size', tokenizer.vocab_size),
        embed_dim=model_config.get('embed_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 256),
        pad_token_id=model_config.get('pad_token_id', tokenizer.pad_token_id)
    )
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Warning: Weights not found at {weights_path}")
else:
    print(f"Warning: Config not found at {config_path}")
    # Try default config
    model = TinyLSTM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        pad_token_id=tokenizer.pad_token_id
    )
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))

model = model.to(device)
model.eval()
print("✓ Model loaded")

# ========================================
# HELPER FUNCTIONS
# ========================================
def generate_response(prompt, max_length=100, temperature=0.7):
    """Generate response from dummy model"""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id
        )
    
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove the prompt from output
    if output.startswith(prompt):
        output = output[len(prompt):].strip()
    
    return output


def calculate_bleu(reference, candidate):
    """Simple BLEU-1 score"""
    ref_words = set(reference.lower().split())
    cand_words = candidate.lower().split()
    
    if not cand_words:
        return 0.0
    
    matches = sum(1 for word in cand_words if word in ref_words)
    return matches / len(cand_words)


def calculate_distinct_n(responses, n=1):
    """Calculate Distinct-n metric for diversity"""
    all_ngrams = []
    
    for response in responses:
        words = response.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
    
    if not all_ngrams:
        return 0.0
    
    return len(set(all_ngrams)) / len(all_ngrams)


def calculate_bertscore(references, candidates):
    """Calculate BERTScore"""
    if not BERTSCORE_AVAILABLE or not references or not candidates:
        return None, None, None
    
    try:
        P, R, F1 = bert_score(
            candidates, 
            references, 
            lang="en", 
            verbose=False,
            device=str(device)
        )
        return P.mean().item(), R.mean().item(), F1.mean().item()
    except Exception as e:
        print(f"BERTScore error: {e}")
        return None, None, None


def detect_hallucination(context, reference, generated):
    """Detect potential hallucinations"""
    hallucination_score = 0.0
    flags = []
    
    # Check for repetition
    words = generated.lower().split()
    if len(words) > 5:
        for i in range(len(words) - 3):
            trigram = ' '.join(words[i:i+3])
            if generated.lower().count(trigram) > 2:
                hallucination_score += 0.3
                flags.append("repetition")
                break
    
    # Check for excessive length
    if len(generated.split()) > len(reference.split()) * 3:
        hallucination_score += 0.15
        flags.append("excessive_length")
    
    # Check for very short/empty response
    if len(generated.split()) < 3:
        hallucination_score += 0.2
        flags.append("too_short")
    
    # Check for gibberish
    gen_lower = generated.lower()
    if len(set(words)) < len(words) * 0.3 and len(words) > 10:
        hallucination_score += 0.3
        flags.append("low_diversity")
    
    return min(hallucination_score, 1.0), flags


# ========================================
# LOAD DATASETS
# ========================================
print("\nLoading datasets...")

# DailyDialog
def load_daily_dialog(base_dir, split='test', max_samples=100):
    dialog_file = os.path.join(base_dir, split, f'dialogues_{split}.txt')
    
    if not os.path.exists(dialog_file):
        # Try validation if test doesn't exist
        dialog_file = os.path.join(base_dir, 'validation', 'dialogues_validation.txt')
    
    pairs = []
    with open(dialog_file, 'r', encoding='utf-8') as f:
        for line in f:
            utterances = line.strip().split('__eou__')
            utterances = [u.strip() for u in utterances if u.strip()]
            for i in range(len(utterances) - 1):
                pairs.append((utterances[i], utterances[i + 1]))
    
    if len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)
    
    return pairs


# Robot control
def load_robot_control(path, max_samples=50):
    with open(path, 'r') as f:
        data = json.load(f)
    
    pairs = [(item['instruction'], item['response']) for item in data]
    
    if len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)
    
    return pairs


random.seed(42)
dialog_pairs = load_daily_dialog(daily_dialog_path, max_samples=100)
robot_pairs = load_robot_control(robot_control_path, max_samples=50)

print(f"  DailyDialog: {len(dialog_pairs)} pairs")
print(f"  Robotics: {len(robot_pairs)} pairs")

# ========================================
# EVALUATE ON DAILYDIALOG
# ========================================
print("\n" + "=" * 60)
print("Evaluating on DailyDialog...")
print("=" * 60)

dialog_bleu_scores = []
dialog_generated = []
dialog_references = []
dialog_hallucination_scores = []
dialog_latencies = []

for context, reference in tqdm(dialog_pairs, desc="DailyDialog"):
    start_time = time.time()
    generated = generate_response(context, max_length=80)
    latency = time.time() - start_time
    
    dialog_latencies.append(latency)
    dialog_generated.append(generated)
    dialog_references.append(reference)
    
    bleu = calculate_bleu(reference, generated)
    dialog_bleu_scores.append(bleu)
    
    hall_score, _ = detect_hallucination(context, reference, generated)
    dialog_hallucination_scores.append(hall_score)

# Calculate metrics
dialog_results = {
    'avg_bleu': float(np.mean(dialog_bleu_scores)),
    'distinct_1': float(calculate_distinct_n(dialog_generated, n=1)),
    'distinct_2': float(calculate_distinct_n(dialog_generated, n=2)),
    'avg_hallucination_rate': float(np.mean(dialog_hallucination_scores)),
    'avg_latency_sec': float(np.mean(dialog_latencies)),
    'min_latency_sec': float(np.min(dialog_latencies)),
    'max_latency_sec': float(np.max(dialog_latencies)),
    'num_samples': len(dialog_pairs)
}

# BERTScore
if BERTSCORE_AVAILABLE:
    bert_p, bert_r, bert_f1 = calculate_bertscore(dialog_references, dialog_generated)
    if bert_p is not None:
        dialog_results['bert_precision'] = float(bert_p)
        dialog_results['bert_recall'] = float(bert_r)
        dialog_results['bert_f1'] = float(bert_f1)

print(f"\nDailyDialog Results:")
print(f"  BLEU: {dialog_results['avg_bleu']:.4f}")
print(f"  Distinct-1: {dialog_results['distinct_1']:.4f}")
print(f"  Distinct-2: {dialog_results['distinct_2']:.4f}")
print(f"  Hallucination Rate: {dialog_results['avg_hallucination_rate']:.4f}")
print(f"  Avg Latency: {dialog_results['avg_latency_sec']:.3f}s")
if 'bert_f1' in dialog_results:
    print(f"  BERTScore F1: {dialog_results['bert_f1']:.4f}")

# ========================================
# EVALUATE ON ROBOTICS
# ========================================
print("\n" + "=" * 60)
print("Evaluating on Surgical Robotics...")
print("=" * 60)

robot_bleu_scores = []
robot_generated = []
robot_references = []
robot_hallucination_scores = []
robot_latencies = []

for instruction, reference in tqdm(robot_pairs, desc="Robotics"):
    start_time = time.time()
    generated = generate_response(instruction, max_length=100)
    latency = time.time() - start_time
    
    robot_latencies.append(latency)
    robot_generated.append(generated)
    robot_references.append(reference)
    
    bleu = calculate_bleu(reference, generated)
    robot_bleu_scores.append(bleu)
    
    hall_score, _ = detect_hallucination(instruction, reference, generated)
    robot_hallucination_scores.append(hall_score)

# Calculate metrics
robotics_results = {
    'avg_bleu': float(np.mean(robot_bleu_scores)),
    'distinct_1': float(calculate_distinct_n(robot_generated, n=1)),
    'distinct_2': float(calculate_distinct_n(robot_generated, n=2)),
    'avg_hallucination_rate': float(np.mean(robot_hallucination_scores)),
    'avg_latency_sec': float(np.mean(robot_latencies)),
    'min_latency_sec': float(np.min(robot_latencies)),
    'max_latency_sec': float(np.max(robot_latencies)),
    'num_samples': len(robot_pairs)
}

# BERTScore
if BERTSCORE_AVAILABLE:
    bert_p, bert_r, bert_f1 = calculate_bertscore(robot_references, robot_generated)
    if bert_p is not None:
        robotics_results['bert_precision'] = float(bert_p)
        robotics_results['bert_recall'] = float(bert_r)
        robotics_results['bert_f1'] = float(bert_f1)

print(f"\nRobotics Results:")
print(f"  BLEU: {robotics_results['avg_bleu']:.4f}")
print(f"  Distinct-1: {robotics_results['distinct_1']:.4f}")
print(f"  Distinct-2: {robotics_results['distinct_2']:.4f}")
print(f"  Hallucination Rate: {robotics_results['avg_hallucination_rate']:.4f}")
print(f"  Avg Latency: {robotics_results['avg_latency_sec']:.3f}s")
if 'bert_f1' in robotics_results:
    print(f"  BERTScore F1: {robotics_results['bert_f1']:.4f}")

# ========================================
# SAMPLE OUTPUTS
# ========================================
print("\n" + "=" * 60)
print("SAMPLE OUTPUTS (showing poor quality)")
print("=" * 60)

sample_prompts = [
    "Hello, how are you?",
    "The vision system detects 'Preparation'. What robotic algorithm applies?",
    "How does a computer learn?",
]

sample_outputs = []
for prompt in sample_prompts:
    output = generate_response(prompt, max_length=60)
    sample_outputs.append({'prompt': prompt, 'response': output})
    print(f"\nQ: {prompt}")
    print(f"A: {output}")

# ========================================
# SAVE RESULTS
# ========================================
results = {
    'dialog': dialog_results,
    'robotics': robotics_results,
    'sample_outputs': sample_outputs
}

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to: {output_path}")

# Save summary text
summary = f"""============================================================
Dummy LSTM Model Evaluation Summary
============================================================

DailyDialog Results:
  BLEU Score:          {dialog_results['avg_bleu']:.4f}
  BERTScore F1:        {dialog_results.get('bert_f1', 'N/A')}
  Hallucination Rate:  {dialog_results['avg_hallucination_rate']:.4f}
  Distinct-1:          {dialog_results['distinct_1']:.4f}
  Distinct-2:          {dialog_results['distinct_2']:.4f}
  Avg Latency:         {dialog_results['avg_latency_sec']:.3f}s
  Samples:             {dialog_results['num_samples']}

Surgical Robotics Results:
  BLEU Score:          {robotics_results['avg_bleu']:.4f}
  BERTScore F1:        {robotics_results.get('bert_f1', 'N/A')}
  Hallucination Rate:  {robotics_results['avg_hallucination_rate']:.4f}
  Distinct-1:          {robotics_results['distinct_1']:.4f}
  Distinct-2:          {robotics_results['distinct_2']:.4f}
  Avg Latency:         {robotics_results['avg_latency_sec']:.3f}s
  Samples:             {robotics_results['num_samples']}

"""

with open(summary_path, 'w') as f:
    f.write(summary)
print(f"✓ Summary saved to: {summary_path}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)