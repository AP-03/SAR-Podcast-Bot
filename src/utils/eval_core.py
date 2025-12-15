"""
eval_Core.py

Run evaluation + sample generations on the best saved GPT-2 core model
without re-running training.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # one level above src/

config_path = os.path.join(script_dir, "../hype/Core.yaml")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at: {config_path}")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print(f"Loaded hyperparameters from: {config_path}")

# Make the paths absolute in the same way as train_Core.py
src_dir = os.path.dirname(script_dir)  # src/
config['output_dir']       = os.path.join(src_dir, config['output_dir'])
config['best_model_path']  = os.path.join(src_dir, config['best_model_path'])
config['final_model_path'] = os.path.join(src_dir, config['final_model_path'])
config['plot_path']        = os.path.join(src_dir, config['plot_path'])
config['robot_control_path'] = os.path.join(src_dir, config['robot_control_path'])

os.makedirs(config['output_dir'], exist_ok=True)

print(f"Output directory   : {config['output_dir']}")
print(f"Robot control data : {config['robot_control_path']}")
print(f"Best model path    : {config['best_model_path']}")

if not os.path.exists(config['robot_control_path']):
    raise FileNotFoundError(f"Robot control data not found at: {config['robot_control_path']}")

if not os.path.exists(config['best_model_path']):
    raise FileNotFoundError(
        f"Best model directory not found at: {config['best_model_path']}\n"
        f"Run training at least for 1 epoch so a best checkpoint is saved."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- Load best saved model + tokenizer ----------------- #

print("\nLoading tokenizer and model from best checkpoint...")

# SIMPLE CASE: best_model_path contains a full causal LM checkpoint
tokenizer = AutoTokenizer.from_pretrained(config['best_model_path'])
model = AutoModelForCausalLM.from_pretrained(config['best_model_path']).to(device)

# If you're using LoRA adapters only, use something like this instead
# (adapt to your GPT2.py setup and base model name):
#
# from peft import PeftModel
# base_model_name = config.get('base_model_name', 'gpt2')
# tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
# model = PeftModel.from_pretrained(base_model, config['best_model_path']).to(device)

# ----------------- DailyDialog data prep (copied from train_Core.py) ----- #

daily_dialog_path = os.path.join(script_dir, "../dataset/DailyDialog")
if not os.path.exists(daily_dialog_path):
    raise FileNotFoundError(f"DailyDialog dataset not found at: {daily_dialog_path}")

print(f"\nLoading DailyDialog dataset from: {daily_dialog_path}")

def load_daily_dialog_split(base_dir, train_folder='train', val_folder='validation'):
    def load_split(folder, prefix):
        utterances = []
        acts = []
        emotions = []

        dialog_file = os.path.join(base_dir, folder, f'dialogues_{prefix}.txt')
        act_file = os.path.join(base_dir, folder, f'dialogues_act_{prefix}.txt')
        emotion_file = os.path.join(base_dir, folder, f'dialogues_emotion_{prefix}.txt')

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

    train_data = load_split(train_folder, 'train')
    val_data = load_split(val_folder, 'validation')
    return train_data, val_data

(train_utterances, train_acts, train_emotions), (val_utterances, val_acts, val_emotions) = \
    load_daily_dialog_split(daily_dialog_path)

print(f"Loaded {len(train_utterances)} training dialogs and {len(val_utterances)} validation dialogs")

dialog_train_instructions = []
dialog_train_responses = []

for idx, utterances in enumerate(train_utterances):
    acts = train_acts[idx]
    emotions = train_emotions[idx]
    for i in range(len(utterances) - 1):
        context_instruction = f"[Act: {acts[i]}] [Emotion: {emotions[i]}] {utterances[i]}"
        context_response   = f"[Act: {acts[i+1]}] [Emotion: {emotions[i+1]}] {utterances[i+1]}"
        dialog_train_instructions.append(context_instruction)
        dialog_train_responses.append(context_response)

dialog_val_instructions = []
dialog_val_responses = []

for idx, utterances in enumerate(val_utterances):
    acts = val_acts[idx]
    emotions = val_emotions[idx]
    for i in range(len(utterances) - 1):
        context_instruction = f"[Act: {acts[i]}] [Emotion: {emotions[i]}] {utterances[i]}"
        context_response   = f"[Act: {acts[i+1]}] [Emotion: {emotions[i+1]}] {utterances[i+1]}"
        dialog_val_instructions.append(context_instruction)
        dialog_val_responses.append(context_response)

dialog_train_combined = [inst + " " + resp for inst, resp in zip(dialog_train_instructions, dialog_train_responses)]
dialog_val_combined   = [inst + " " + resp for inst, resp in zip(dialog_val_instructions, dialog_val_responses)]

dialog_train_encodings = tokenizer(dialog_train_combined, padding=True, truncation=True,
                                   return_tensors='pt', max_length=config['max_length'])
dialog_val_encodings   = tokenizer(dialog_val_combined, padding=True, truncation=True,
                                   return_tensors='pt', max_length=config['max_length'])

dialog_train_input_ids    = dialog_train_encodings['input_ids']
dialog_train_attention    = dialog_train_encodings['attention_mask']
dialog_train_labels       = dialog_train_encodings['input_ids'].clone()

dialog_val_input_ids      = dialog_val_encodings['input_ids']
dialog_val_attention      = dialog_val_encodings['attention_mask']
dialog_val_labels         = dialog_val_encodings['input_ids'].clone()

print("Daily Dialog Data:")
print(f"  Training pairs:   {len(dialog_train_instructions)}")
print(f"  Validation pairs: {len(dialog_val_instructions)}")
print(f"  Input shape:      {dialog_train_input_ids.shape}")

# ----------------- Surgical Robotics data prep (copied from train_Core.py) ----- #

with open(config['robot_control_path'], 'r') as f:
    robot_control_data = json.load(f)

instructions = [item['instruction'] for item in robot_control_data]
responses    = [item['response'] for item in robot_control_data]

robot_combined = [inst + " " + resp for inst, resp in zip(instructions, responses)]
robot_encodings = tokenizer(robot_combined, padding=True, truncation=True,
                            return_tensors='pt', max_length=config['max_length'])

input_ids      = robot_encodings['input_ids']
attention_mask = robot_encodings['attention_mask']
labels         = robot_encodings['input_ids'].clone()

total_samples = len(robot_control_data)
train_size = int(config['robot_train_split'] * total_samples)

train_instruction_ids = input_ids[:train_size]
train_attention_mask  = attention_mask[:train_size]
train_response        = labels[:train_size]

val_instruction_ids   = input_ids[train_size:]
val_attention_mask    = attention_mask[train_size:]
val_response          = labels[train_size:]

print("\nRobot Control Data:")
print(f"  Total samples:     {total_samples}")
print(f"  Training samples:  {train_size}")
print(f"  Validation samples:{total_samples - train_size}")
print(f"  Input shape:       {train_instruction_ids.shape}")

# ----------------- Combine datasets + DataLoaders (same as train_Core.py) ----- #

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attn_masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    input_ids_padded  = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_masks_padded = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    labels_padded     = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    return input_ids_padded, attn_masks_padded, labels_padded

dialog_train_dataset = TensorDataset(dialog_train_input_ids, dialog_train_attention, dialog_train_labels)
dialog_val_dataset   = TensorDataset(dialog_val_input_ids, dialog_val_attention, dialog_val_labels)

robot_train_dataset  = TensorDataset(train_instruction_ids, train_attention_mask, train_response)
robot_val_dataset    = TensorDataset(val_instruction_ids, val_attention_mask, val_response)

print("\nDataset Statistics (Before Balancing):")
print(f"  Dialog training samples: {len(dialog_train_dataset)}")
print(f"  Robot  training samples: {len(robot_train_dataset)}")

repeat_factor = max(1, len(dialog_train_dataset) // len(robot_train_dataset))
print(f"  Oversampling robotics data by {repeat_factor}x")

from torch.utils.data import ConcatDataset
robot_train_repeated = ConcatDataset([robot_train_dataset] * repeat_factor)

combined_train_dataset = ConcatDataset([dialog_train_dataset, robot_train_repeated])
combined_val_dataset   = ConcatDataset([dialog_val_dataset, robot_val_dataset])

print("\nDataset Statistics (After Balancing):")
print(f"  Dialog training samples:          {len(dialog_train_dataset)}")
print(f"  Robot training samples (oversamp):{len(robot_train_repeated)}")
print(f"  Total training samples:           {len(combined_train_dataset)}")
print(f"  Total validation samples:         {len(combined_val_dataset)}")

from torch.utils.data import DataLoader

val_loader = DataLoader(combined_val_dataset,
                        batch_size=config['batch_size'],
                        collate_fn=collate_fn)

print(f"\nBatch configuration:")
print(f"  Batch size: {config['batch_size']}")
print(f"  Validation batches: {len(val_loader)}")

# ----------------- Evaluation metrics ----- #

def calculate_distinct_n(texts, n):
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if len(all_ngrams) == 0:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

print("\n" + "="*50)
print("Evaluating Model (Best Checkpoint)...")
print("="*50)

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

print("\n" + "="*50)
print("DIVERSITY METRICS")
print("="*50)

distinct_1 = calculate_distinct_n(generated_texts, 1)
distinct_2 = calculate_distinct_n(generated_texts, 2)
avg_length = float(np.mean([len(text.split()) for text in generated_texts]))

print("\nOverall Metrics:")
print(f"  Distinct-1 (unigram diversity): {distinct_1:.4f}")
print(f"  Distinct-2 (bigram diversity):  {distinct_2:.4f}")
print(f"  Average Response Length:         {avg_length:.2f} tokens")

dialog_distinct_1 = calculate_distinct_n(dialog_responses, 1)
dialog_distinct_2 = calculate_distinct_n(dialog_responses, 2)
dialog_avg_length = float(np.mean([len(text.split()) for text in dialog_responses]))

print("\nDialog-specific Metrics:")
print(f"  Distinct-1:     {dialog_distinct_1:.4f}")
print(f"  Distinct-2:     {dialog_distinct_2:.4f}")
print(f"  Average Length: {dialog_avg_length:.2f} tokens")

robotics_distinct_1 = calculate_distinct_n(robotics_responses, 1)
robotics_distinct_2 = calculate_distinct_n(robotics_responses, 2)
robotics_avg_length = float(np.mean([len(text.split()) for text in robotics_responses]))

print("\nRobotics-specific Metrics:")
print(f"  Distinct-1:     {robotics_distinct_1:.4f}")
print(f"  Distinct-2:     {robotics_distinct_2:.4f}")
print(f"  Average Length: {robotics_avg_length:.2f} tokens")

print("\nCalculating Token Accuracy on Validation Sample...")
sample_size = min(config['eval_sample_size'], len(val_loader.dataset))
correct_tokens = 0
total_tokens = 0

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i * config['batch_size'] >= sample_size:
            break
        input_ids, attention_mask, labels = [t.to(device) for t in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        predictions = shift_logits.argmax(dim=-1)
        mask = shift_labels != tokenizer.pad_token_id
        correct = (predictions == shift_labels) & mask
        correct_tokens += correct.sum().item()
        total_tokens += mask.sum().item()

token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
print(f"  Token Accuracy: {token_accuracy:.4f} ({correct_tokens}/{total_tokens})")

results_path = os.path.join(config['output_dir'], "evaluation_results.json")
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

with open(results_path, "w") as f:
    json.dump(evaluation_results, f, indent=2)

print(f"\nEvaluation results saved to: {results_path}")

summary_path = os.path.join(config['output_dir'], "evaluation_summary.txt")
with open(summary_path, "w") as f:
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
print("Evaluation Complete (Best Checkpoint)!")
print("="*50)
