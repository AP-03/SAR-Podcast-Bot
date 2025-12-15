"""
SAR-Podcast-Bot Inference
=========================
Run the trained model for the podcast demo.

Usage:
    python inference_model.py --demo          # Run full demo
    python inference_model.py --interactive   # Chat mode
    python inference_model.py                 # Both

The model was trained on:
- Your knowledge base (phase_to_control_mapping.json, tool_to_robot_mapping.json)
- Project-specific Q&A (CNN, LSTM, pipeline)
- AI literacy questions
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path, use_4bit=True):
    """Load the trained model"""
    
    model_path = Path(model_path)
    print(f"Loading model from: {model_path}")
    
    # Check for LoRA adapter
    adapter_config_path = model_path / 'adapter_config.json'
    
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get('base_model_name_or_path', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        
        print(f"Loading LoRA adapter (base: {base_model_name})")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model
        if use_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except:
                print("4-bit failed, using float16")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, str(model_path), local_files_only=True)
    else:
        # Full model
        print("Loading full model...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("✓ Model loaded")
    
    return model, tokenizer


# =============================================================================
# GENERATION
# =============================================================================

def format_prompt(instruction):
    """Format for TinyLlama chat"""
    return f"<|user|>\n{instruction}</s>\n<|assistant|>\n"


def generate_response(model, tokenizer, instruction, max_new_tokens=200, temperature=0.7):
    """Generate a response"""
    
    prompt = format_prompt(instruction)
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=300)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant part
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    # Clean up
    response = response.replace("</s>", "").strip()
    
    return response


# =============================================================================
# PODCAST BOT
# =============================================================================

class PodcastBot:
    """Interactive podcast bot"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.history = []
    
    def ask(self, question, temperature=0.7):
        """Get a response"""
        response = generate_response(
            self.model, self.tokenizer, question,
            temperature=temperature
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
        print(f"💾 Saved to {filepath}")


# =============================================================================
# DEMO
# =============================================================================

def run_demo(bot):
    """Run the full podcast demo"""
    
    print("\n" + "=" * 70)
    print("🎙️  SAR-PODCAST-BOT DEMO")
    print("=" * 70)
    
    sections = {
        "🏥 SURGICAL PHASES": [
            "The vision system detects 'Preparation'. What robotic algorithm applies here?",
            "What is the Clipping/Cutting phase?",
            "List all the surgical phases.",
        ],
        "🔧 SURGICAL TOOLS": [
            "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
            "What tools are used in this surgery?",
        ],
        "🤖 OUR SYSTEM": [
            "What is the SAR-Podcast-Bot?",
            "How does the vision system work?",
            "What CNN model do you use?",
            "What is the LSTM used for?",
        ],
        "🧠 AI LITERACY": [
            "How does a computer learn?",
            "What is deep learning?",
            "What is a neural network?",
        ],
        "⚖️ ETHICS": [
            "Will AI replace surgeons?",
            "How do we ensure AI in surgery is safe?",
        ],
        "💬 CONVERSATIONAL": [
            "Hello!",
            "What can you help with?",
        ]
    }
    
    for section_name, questions in sections.items():
        print(f"\n{'─' * 70}")
        print(f"{section_name}")
        print('─' * 70)
        
        for q in questions:
            print(f"\n🎤 Q: {q}")
            response = bot.ask(q)
            print(f"🤖 A: {response}")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
    
    return bot.history


def interactive_mode(bot):
    """Interactive chat mode"""
    
    print("\n" + "=" * 60)
    print("🎙️ INTERACTIVE MODE")
    print("=" * 60)
    print("\nCommands:")
    print("  /demo  - Run full demo")
    print("  /save  - Save conversation")
    print("  /quit  - Exit")
    print("\nOr just type your question!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() in ['/quit', '/exit', '/q', 'quit', 'exit']:
                break
            
            if user_input == '/demo':
                run_demo(bot)
                continue
            
            if user_input == '/save':
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                bot.save_history(f"conversation_{ts}.json")
                continue
            
            # Regular question
            response = bot.ask(user_input)
            print(f"\n🤖 Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n")
            break
    
    print("Goodbye! 👋")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SAR-Podcast-Bot Inference")
    parser.add_argument('--model', type=str, 
                       default='src/results/model_final/best_model',
                       help='Path to trained model')
    parser.add_argument('--demo', action='store_true', help='Run demo only')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode only')
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantization')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🏥 SAR-PODCAST-BOT")
    print("Surgical Robotics + AI Education")
    print("=" * 70)
    
    # Find model
    search_paths = [
        args.model,
        'src/results/model_final/best_model',
        'results/model_final/best_model',
        '../results/model_final/best_model',
        'src/results/tinyllama_surgical/best_model',
        'results/tinyllama_surgical/best_model',
    ]
    
    model_path = None
    for p in search_paths:
        if os.path.exists(p):
            model_path = p
            break
    
    if not model_path:
        print("❌ Model not found!")
        print("\nSearched in:")
        for p in search_paths:
            print(f"  - {p}")
        print("\nTrain first with: python train_model.py")
        return
    
    # Load model
    model, tokenizer = load_model(model_path, use_4bit=not args.no_4bit)
    bot = PodcastBot(model, tokenizer)
    
    # Run
    if args.demo:
        run_demo(bot)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bot.save_history(f"demo_{ts}.json")
    elif args.interactive:
        interactive_mode(bot)
    else:
        # Default: demo then interactive
        run_demo(bot)
        
        try:
            print("\n")
            cont = input("Continue to interactive mode? (y/n): ").lower().strip()
            if cont == 'y':
                interactive_mode(bot)
        except:
            pass
    
    # Save offer
    if bot.history and not args.demo:
        try:
            save = input("\nSave conversation? (y/n): ").lower().strip()
            if save == 'y':
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                bot.save_history(f"conversation_{ts}.json")
        except:
            pass


if __name__ == "__main__":
    main()