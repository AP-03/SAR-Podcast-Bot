"""
Evaluation script for GPT-2 Core model on DailyDialog and Surgical Robotics datasets
Metrics: BLEU, BERTScore, Distinct-n, Hallucination Rate, Latency
"""
import os
import sys
import json
import random
import re
import time
from collections import defaultdict
from tqdm import tqdm
import torch
import yaml

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import BERTScore
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert-score not installed. Install with: pip install bert-score")
    BERTSCORE_AVAILABLE = False


def load_core_model(config_path, best_model_path):
    """Load the trained GPT-2 Core model"""
    print(f"\nLoading GPT-2 Core model from: {best_model_path}")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model directory not found at: {best_model_path}\n"
            f"Run training first to generate a checkpoint."
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    model = AutoModelForCausalLM.from_pretrained(best_model_path).to(device)
    model.eval()  # Set to evaluation mode
    
    print("✓ Model loaded successfully")
    return model, tokenizer, device


def generate_response(model, tokenizer, device, prompt, max_new_tokens=100, temperature=0.7, debug=False):
    """Generate response using TinyLlama with correct chat format"""
    
    # Format prompt for TinyLlama chat (matching train_llama.py format_prompt)
    formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    if debug:
        print(f"\n--- Formatted Prompt ---\n{formatted_prompt}\n------------------------")

    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id, # Use EOS token as pad_token_id
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
    
    # Clean up the generated text (remove any remaining chat tags if present)
    generated_text = generated_text.replace("<|end|>", "").strip()
    
    if debug:
        print(f"\n--- Generated Response ---\n{generated_text}\n--------------------------")
        
    return generated_text

def load_daily_dialog(base_dir, split='test'):
    """Load DailyDialog data with act and emotion labels (matching training format)"""
    dialog_file = os.path.join(base_dir, split, f'dialogues_{split}.txt')
    act_file = os.path.join(base_dir, split, f'dialogues_act_{split}.txt')
    emotion_file = os.path.join(base_dir, split, f'dialogues_emotion_{split}.txt')
    
    if not os.path.exists(dialog_file):
        raise FileNotFoundError(f"File not found: {dialog_file}")
    if not os.path.exists(act_file):
        raise FileNotFoundError(f"File not found: {act_file}")
    if not os.path.exists(emotion_file):
        raise FileNotFoundError(f"File not found: {emotion_file}")
    
    # Load all three files
    with open(dialog_file, 'r', encoding='utf-8') as f:
        dialog_lines = [line.strip() for line in f]
    
    with open(act_file, 'r', encoding='utf-8') as f:
        act_lines = [line.strip() for line in f]
    
    with open(emotion_file, 'r', encoding='utf-8') as f:
        emotion_lines = [line.strip() for line in f]
    
    dialog_pairs = []
    
    for dialog_line, act_line, emotion_line in zip(dialog_lines, act_lines, emotion_lines):
        utterances = dialog_line.split('__eou__')
        utterances = [u.strip() for u in utterances if u.strip()]
        
        acts = act_line.split()
        emotions = emotion_line.split()
        
        # Create context-response pairs with act and emotion labels (matching training format)
        for i in range(len(utterances) - 1):
            # Format: [Act: X] [Emotion: Y] utterance (same as training)
            context = f"[Act: {acts[i]}] [Emotion: {emotions[i]}] {utterances[i]}"
            response = f"[Act: {acts[i+1]}] [Emotion: {emotions[i+1]}] {utterances[i + 1]}"
            if context and response:
                dialog_pairs.append((context, response))
    
    return dialog_pairs


def load_robot_control(json_path):
    """Load robot control data"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    robot_pairs = []
    for entry in data:
        # Try both possible field names
        instruction = entry.get('instruction', entry.get('phase', ''))
        response = entry.get('response', entry.get('control_command', ''))
        if instruction and response:
            robot_pairs.append((instruction, response))
    
    return robot_pairs


def calculate_bleu(reference, candidate):
    """Simple BLEU-1 score (unigram precision)"""
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
    """Calculate BERTScore (Precision, Recall, F1)"""
    if not BERTSCORE_AVAILABLE:
        return None, None, None
    
    try:
        P, R, F1 = bert_score(
            candidates, 
            references, 
            lang="en", 
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        return P.mean().item(), R.mean().item(), F1.mean().item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return None, None, None


def detect_hallucination(context, reference, generated):
    """
    Detect potential hallucinations in generated text.
    
    Hallucination indicators:
    1. Contradicts reference information
    2. Contains fabricated details not in context
    3. Makes unsupported claims
    4. Contains nonsensical or repetitive text
    
    Returns:
        hallucination_score: 0.0 (no hallucination) to 1.0 (severe hallucination)
    """
    hallucination_score = 0.0
    flags = []
    
    # 1. Check for repetition (sign of model confusion)
    words = generated.lower().split()
    if len(words) > 5:
        # Check for repeated sequences
        for i in range(len(words) - 3):
            trigram = ' '.join(words[i:i+3])
            if generated.lower().count(trigram) > 2:
                hallucination_score += 0.2
                flags.append("repetition")
                break
    
    # 2. Check for excessive length (rambling/hallucination)
    if len(generated.split()) > len(reference.split()) * 3:
        hallucination_score += 0.15
        flags.append("excessive_length")
    
    # 3. Check for fabricated specific details (numbers, names, dates)
    # Numbers in generated but not in context/reference
    gen_numbers = set(re.findall(r'\b\d+\b', generated))
    context_numbers = set(re.findall(r'\b\d+\b', context + ' ' + reference))
    fabricated_numbers = gen_numbers - context_numbers
    if len(fabricated_numbers) > 2:
        hallucination_score += 0.2
        flags.append("fabricated_numbers")
    
    # 4. Check for hedging/uncertainty phrases (can indicate hallucination)
    uncertainty_phrases = [
        "i don't know", "not sure", "maybe", "perhaps", "i think",
        "it seems", "it appears", "probably", "i cannot", "unable to"
    ]
    gen_lower = generated.lower()
    if any(phrase in gen_lower for phrase in uncertainty_phrases):
        hallucination_score += 0.1
        flags.append("uncertainty")
    
    # 5. Check for proper names not in context (potential fabrication)
    # Simple heuristic: capitalized words not at sentence start
    gen_words = generated.split()
    context_words = (context + ' ' + reference).split()
    
    potential_names = [
        word for i, word in enumerate(gen_words)
        if word[0].isupper() and i > 0 and gen_words[i-1][-1] not in '.!?'
    ]
    context_caps = [word for word in context_words if word[0].isupper()]
    
    fabricated_names = set(potential_names) - set(context_caps)
    if len(fabricated_names) > 2:
        hallucination_score += 0.2
        flags.append("fabricated_names")
    
    # 6. Check for contradictory negations
    if "not" in gen_lower or "no" in gen_lower:
        # This is a simple heuristic - real contradiction detection is complex
        if "yes" in reference.lower() and "not" in gen_lower:
            hallucination_score += 0.15
            flags.append("contradiction")
    
    # Cap at 1.0
    hallucination_score = min(hallucination_score, 1.0)
    
    return hallucination_score, flags


def evaluate_dialog(pairs, model, tokenizer, device, sample_size=100):
    """Evaluate Core model on dialog pairs"""
    print(f"\n{'='*60}")
    print(f"Evaluating on DailyDialog (sample size: {sample_size})")
    print(f"{'='*60}\n")
    
    # Sample random pairs
    if len(pairs) > sample_size:
        pairs = random.sample(pairs, sample_size)
    
    bleu_scores = []
    generated_responses = []
    references = []
    contexts = []
    hallucination_scores = []
    hallucination_flags_list = []
    latencies = []
    
    for context, reference in tqdm(pairs, desc="Generating dialog responses"):
        # Match training format: just the context/utterance directly
        # During training, format was: instruction + " " + response
        # So at test time, we just provide the instruction (context)
        prompt = context
        
        try:
            # Measure latency
            start_time = time.time()
            generated = generate_response(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7
            )
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            generated_responses.append(generated)
            references.append(reference)
            contexts.append(context)
            
            # Calculate BLEU score
            bleu = calculate_bleu(reference, generated)
            bleu_scores.append(bleu)
            
            # Detect hallucination
            hall_score, flags = detect_hallucination(context, reference, generated)
            hallucination_scores.append(hall_score)
            hallucination_flags_list.append(flags)
            
        except Exception as e:
            print(f"\nError generating response: {e}")
            continue
    
    # Calculate metrics
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    distinct_1 = calculate_distinct_n(generated_responses, n=1)
    distinct_2 = calculate_distinct_n(generated_responses, n=2)
    avg_hallucination = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0
    
    # Calculate latency metrics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    
    # Calculate BERTScore
    bert_p, bert_r, bert_f1 = calculate_bertscore(references, generated_responses)
    
    # Show examples
    print(f"\n{'-'*60}")
    print("Sample Generations:")
    print(f"{'-'*60}")
    for i in range(min(5, len(pairs))):
        context, reference = pairs[i]
        generated = generated_responses[i] if i < len(generated_responses) else "N/A"
        hall_score = hallucination_scores[i] if i < len(hallucination_scores) else 0.0
        flags = hallucination_flags_list[i] if i < len(hallucination_flags_list) else []
        
        print(f"\nExample {i+1}:")
        print(f"  Context:       {context}")
        print(f"  Reference:     {reference}")
        print(f"  Generated:     {generated}")
        print(f"  BLEU:          {bleu_scores[i]:.4f}" if i < len(bleu_scores) else "")
        print(f"  Hallucination: {hall_score:.4f}" + (f" {flags}" if flags else ""))
    
    return {
        'avg_bleu': avg_bleu,
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'bert_precision': bert_p,
        'bert_recall': bert_r,
        'bert_f1': bert_f1,
        'avg_hallucination_rate': avg_hallucination,
        'avg_latency_sec': avg_latency,
        'min_latency_sec': min_latency,
        'max_latency_sec': max_latency,
        'num_samples': len(bleu_scores)
    }


def evaluate_robotics(pairs, model, tokenizer, device, sample_size=50):
    """Evaluate Core model on robotics control pairs"""
    print(f"\n{'='*60}")
    print(f"Evaluating on Surgical Robotics (sample size: {sample_size})")
    print(f"{'='*60}\n")
    
    # Sample random pairs
    if len(pairs) > sample_size:
        pairs = random.sample(pairs, sample_size)
    
    bleu_scores = []
    generated_responses = []
    references = []
    contexts = []
    hallucination_scores = []
    hallucination_flags_list = []
    latencies = []
    
    for phase, reference_control in tqdm(pairs, desc="Generating robot commands"):
        # Match training format: just the instruction directly
        # During training, format was: instruction + " " + response
        # So at test time, we just provide the instruction (phase question)
        prompt = phase
        
        try:
            # Measure latency
            start_time = time.time()
            generated = generate_response(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.3  # Lower temperature for more deterministic outputs
            )
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            generated_responses.append(generated)
            references.append(reference_control)
            contexts.append(phase)
            
            # Calculate BLEU score
            bleu = calculate_bleu(reference_control, generated)
            bleu_scores.append(bleu)
            
            # Detect hallucination
            hall_score, flags = detect_hallucination(phase, reference_control, generated)
            hallucination_scores.append(hall_score)
            hallucination_flags_list.append(flags)
            
        except Exception as e:
            print(f"\nError generating command: {e}")
            continue
    
    # Calculate metrics
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    distinct_1 = calculate_distinct_n(generated_responses, n=1)
    distinct_2 = calculate_distinct_n(generated_responses, n=2)
    avg_hallucination = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0
    
    # Calculate latency metrics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    
    # Calculate BERTScore
    bert_p, bert_r, bert_f1 = calculate_bertscore(references, generated_responses)
    
    # Show examples
    print(f"\n{'-'*60}")
    print("Sample Robot Command Generations:")
    print(f"{'-'*60}")
    for i in range(min(5, len(pairs))):
        phase, reference = pairs[i]
        generated = generated_responses[i] if i < len(generated_responses) else "N/A"
        hall_score = hallucination_scores[i] if i < len(hallucination_scores) else 0.0
        flags = hallucination_flags_list[i] if i < len(hallucination_flags_list) else []
        
        print(f"\nExample {i+1}:")
        print(f"  Phase:         {phase}")
        print(f"  Reference:     {reference}")
        print(f"  Generated:     {generated}")
        print(f"  BLEU:          {bleu_scores[i]:.4f}" if i < len(bleu_scores) else "")
        print(f"  Hallucination: {hall_score:.4f}" + (f" {flags}" if flags else ""))
    
    return {
        'avg_bleu': avg_bleu,
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'bert_precision': bert_p,
        'bert_recall': bert_r,
        'bert_f1': bert_f1,
        'avg_hallucination_rate': avg_hallucination,
        'avg_latency_sec': avg_latency,
        'min_latency_sec': min_latency,
        'max_latency_sec': max_latency,
        'num_samples': len(bleu_scores)
    }


def main():
    """Main evaluation function"""
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, '..')
    daily_dialog_path = os.path.join(src_dir, 'dataset', 'DailyDialog')
    robot_control_path = os.path.join(src_dir, 'dataset', 'Surgical_Robotics', 'robot_control.json')
    
    # Load config and model
    config_path = os.path.join(src_dir, 'hype', 'Core.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    best_model_path = os.path.join(src_dir, config['best_model_path'])
    
    print("="*60)
    print("GPT-2 Core Model Evaluation")
    print("="*60)
    
    # Load the trained model
    model, tokenizer, device = load_core_model(config_path, best_model_path)
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        dialog_pairs = load_daily_dialog(daily_dialog_path, split='test')
        print(f"✓ Loaded {len(dialog_pairs)} DailyDialog test pairs")
    except Exception as e:
        print(f"✗ Error loading DailyDialog: {e}")
        dialog_pairs = []
    
    try:
        robot_pairs = load_robot_control(robot_control_path)
        print(f"✓ Loaded {len(robot_pairs)} Surgical Robotics pairs")
    except Exception as e:
        print(f"✗ Error loading Surgical Robotics: {e}")
        robot_pairs = []
    
    # Evaluate on both datasets
    results = {}
    
    if dialog_pairs:
        dialog_results = evaluate_dialog(dialog_pairs, model, tokenizer, device, sample_size=100)
        results['dialog'] = dialog_results
    
    if robot_pairs:
        robotics_results = evaluate_robotics(robot_pairs, model, tokenizer, device, sample_size=50)
        results['robotics'] = robotics_results
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}\n")
    
    if 'dialog' in results:
        print("DailyDialog Results:")
        print(f"  BLEU Score:          {results['dialog']['avg_bleu']:.4f}")
        if results['dialog'].get('bert_f1'):
            print(f"  BERTScore P/R/F1:    {results['dialog']['bert_precision']:.4f} / {results['dialog']['bert_recall']:.4f} / {results['dialog']['bert_f1']:.4f}")
        print(f"  Hallucination Rate:  {results['dialog']['avg_hallucination_rate']:.4f}")
        print(f"  Distinct-1:          {results['dialog']['distinct_1']:.4f}")
        print(f"  Distinct-2:          {results['dialog']['distinct_2']:.4f}")
        print(f"  Avg Latency:         {results['dialog']['avg_latency_sec']:.3f}s (min: {results['dialog']['min_latency_sec']:.3f}s, max: {results['dialog']['max_latency_sec']:.3f}s)")
        print(f"  Samples:             {results['dialog']['num_samples']}")
        print()
    
    if 'robotics' in results:
        print("Surgical Robotics Results:")
        print(f"  BLEU Score:          {results['robotics']['avg_bleu']:.4f}")
        if results['robotics'].get('bert_f1'):
            print(f"  BERTScore P/R/F1:    {results['robotics']['bert_precision']:.4f} / {results['robotics']['bert_recall']:.4f} / {results['robotics']['bert_f1']:.4f}")
        print(f"  Hallucination Rate:  {results['robotics']['avg_hallucination_rate']:.4f}")
        print(f"  Distinct-1:          {results['robotics']['distinct_1']:.4f}")
        print(f"  Distinct-2:          {results['robotics']['distinct_2']:.4f}")
        print(f"  Avg Latency:         {results['robotics']['avg_latency_sec']:.3f}s (min: {results['robotics']['min_latency_sec']:.3f}s, max: {results['robotics']['max_latency_sec']:.3f}s)")
        print(f"  Samples:             {results['robotics']['num_samples']}")
        print()
    
    # Save results
    results_dir = os.path.join(src_dir, 'results', 'core_results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Save summary text
    summary_file = os.path.join(results_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("GPT-2 Core Model Evaluation Summary\n")
        f.write("="*60 + "\n\n")
        
        if 'dialog' in results:
            f.write("DailyDialog Results:\n")
            f.write(f"  BLEU Score:          {results['dialog']['avg_bleu']:.4f}\n")
            if results['dialog'].get('bert_f1'):
                f.write(f"  BERTScore Precision: {results['dialog']['bert_precision']:.4f}\n")
                f.write(f"  BERTScore Recall:    {results['dialog']['bert_recall']:.4f}\n")
                f.write(f"  BERTScore F1:        {results['dialog']['bert_f1']:.4f}\n")
            f.write(f"  Hallucination Rate:  {results['dialog']['avg_hallucination_rate']:.4f}\n")
            f.write(f"  Distinct-1:          {results['dialog']['distinct_1']:.4f}\n")
            f.write(f"  Distinct-2:          {results['dialog']['distinct_2']:.4f}\n")
            f.write(f"  Avg Latency:         {results['dialog']['avg_latency_sec']:.3f}s\n")
            f.write(f"  Min Latency:         {results['dialog']['min_latency_sec']:.3f}s\n")
            f.write(f"  Max Latency:         {results['dialog']['max_latency_sec']:.3f}s\n")
            f.write(f"  Samples:             {results['dialog']['num_samples']}\n\n")
        
        if 'robotics' in results:
            f.write("Surgical Robotics Results:\n")
            f.write(f"  BLEU Score:          {results['robotics']['avg_bleu']:.4f}\n")
            if results['robotics'].get('bert_f1'):
                f.write(f"  BERTScore Precision: {results['robotics']['bert_precision']:.4f}\n")
                f.write(f"  BERTScore Recall:    {results['robotics']['bert_recall']:.4f}\n")
                f.write(f"  BERTScore F1:        {results['robotics']['bert_f1']:.4f}\n")
            f.write(f"  Hallucination Rate:  {results['robotics']['avg_hallucination_rate']:.4f}\n")
            f.write(f"  Distinct-1:          {results['robotics']['distinct_1']:.4f}\n")
            f.write(f"  Distinct-2:          {results['robotics']['distinct_2']:.4f}\n")
            f.write(f"  Avg Latency:         {results['robotics']['avg_latency_sec']:.3f}s\n")
            f.write(f"  Min Latency:         {results['robotics']['min_latency_sec']:.3f}s\n")
            f.write(f"  Max Latency:         {results['robotics']['max_latency_sec']:.3f}s\n")
            f.write(f"  Samples:             {results['robotics']['num_samples']}\n")
    
    print(f"Summary saved to: {summary_file}")
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
