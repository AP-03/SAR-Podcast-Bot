"""
GPT-2 Instruction Model - Inference
====================================
Use the instruction-tuned model for podcast demo.

This matches the Alpaca-style format used in training:
### Instruction:
{question}

### Response:
{answer}
"""

import os
import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path, device='cuda'):
    """Load the instruction-tuned model"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model_path = Path(model_path)
    
    print(f"Loading model from: {model_path}")
    
    # Check if it's a LoRA adapter
    adapter_config = model_path / 'adapter_config.json'
    
    if adapter_config.exists() and PEFT_AVAILABLE:
        with open(adapter_config) as f:
            config = json.load(f)
        base_model = config.get('base_model_name_or_path', 'gpt2')
        
        print(f"Loading LoRA adapter (base: {base_model})")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, str(model_path), local_files_only=True)
    else:
        print("Loading full model")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model, tokenizer, device


# =============================================================================
# GENERATION
# =============================================================================

def generate_response(model, tokenizer, device, question, max_new_tokens=100, temperature=0.7):
    """Generate response using instruction format"""
    
    # Format as instruction
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=200)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[-1].strip()
    else:
        response = full_text[len(prompt):].strip()
    
    # Stop at next instruction
    if "### Instruction:" in response:
        response = response.split("### Instruction:")[0].strip()
    
    return response


# =============================================================================
# PODCAST BOT
# =============================================================================

class PodcastBot:
    """Interactive podcast bot"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.history = []
    
    def ask(self, question):
        """Get response"""
        response = generate_response(
            self.model, self.tokenizer, self.device, question
        )
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response
        })
        
        return response
    
    def save_history(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"Saved to {filepath}")


def run_demo(bot):
    """Run podcast demo"""
    
    print("\n" + "=" * 70)
    print("🎙️ SAR-PODCAST-BOT DEMO")
    print("=" * 70)
    
    demo_questions = [
        # Surgical phases
        "The vision system detects 'Preparation'. What robotic algorithm applies?",
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        "What are the stages of a cholecystectomy?",
        
        # AI literacy
        "How does a computer learn?",
        "What is deep learning?",
        "Will AI replace surgeons?",
        
        # Conversational
        "Hello!",
    ]
    
    for q in demo_questions:
        print(f"\n📝 Q: {q}")
        response = bot.ask(q)
        print(f"🤖 A: {response}")
        print("-" * 50)
    
    return bot.history


def interactive_mode(bot):
    """Interactive Q&A"""
    
    print("\n" + "=" * 60)
    print("🎙️ INTERACTIVE MODE")
    print("Type your question or /quit to exit")
    print("=" * 60)
    
    while True:
        try:
            q = input("\nYou: ").strip()
            if not q:
                continue
            if q.lower() in ['/quit', '/exit', '/q', 'quit', 'exit']:
                break
            if q == '/demo':
                run_demo(bot)
                continue
            
            response = bot.ask(q)
            print(f"\n🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye! 👋")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-2 Instruction Model Inference")
    parser.add_argument('--model', type=str, 
                       default='src/results/gpt2_instruction/best_model',
                       help='Path to model')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("SAR-PODCAST-BOT - Instruction-Tuned GPT-2")
    print("=" * 70)
    
    # Find model
    if not os.path.exists(args.model):
        # Try common locations
        alternatives = [
            'src/results/gpt2_instruction/best_model',
            'results/gpt2_instruction/best_model',
            '../results/gpt2_instruction/best_model',
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                args.model = alt
                break
        else:
            print(f"❌ Model not found: {args.model}")
            print("Train first with: python train_gpt2_clean.py")
            return
    
    model, tokenizer, device = load_model(args.model, args.device)
    bot = PodcastBot(model, tokenizer, device)
    
    if args.demo:
        run_demo(bot)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bot.save_history(f"demo_{ts}.json")
    elif args.interactive:
        interactive_mode(bot)
    else:
        # Default: run demo then interactive
        run_demo(bot)
        interactive_mode(bot)
    
    if bot.history:
        try:
            save = input("\nSave conversation? (y/n): ").lower().strip()
            if save == 'y':
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                bot.save_history(f"conversation_{ts}.json")
        except:
            pass


if __name__ == "__main__":
    main()