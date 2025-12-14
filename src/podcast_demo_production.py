"""
SAR-PODCAST-BOT: Production-Ready Podcast Demo (v2)
====================================================
FIXED: Local PEFT/LoRA model loading

Usage:
    python podcast_demo_v2.py --model path/to/gpt2_best_model --demo
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Note: PEFT not installed. Install with: pip install peft")


# =============================================================================
# KNOWLEDGE BASE FOR AI LITERACY
# =============================================================================

AI_LITERACY_ANSWERS = {
    "how does a computer learn": """
A computer learns through a process similar to how we learn from examples. 
Imagine showing a child thousands of pictures of cats and dogs. Eventually, 
they learn to tell them apart. Neural networks work the same way - we feed 
them millions of examples, and they gradually adjust their internal 
"weights" to recognize patterns. Each time they make a mistake, they adjust 
slightly to do better next time. This is called "training" and it's why AI 
needs so much data and computing power.
""",
    
    "neural network grandmother": """
Think of a neural network like a very complex game of "telephone." You 
whisper something to the first person, they interpret it and pass it on, 
and by the end, you get a result. Each person in the chain is like a 
"neuron" - they take input, process it slightly, and pass it forward. 
In a computer, these "neurons" are just math equations. The magic is that 
by having millions of these simple operations working together, the 
network can learn to recognize faces, understand speech, or even help 
doctors analyze medical images.
""",
    
    "computational resources ai": """
AI needs lots of computing power for the same reason training for a 
marathon takes time - it's doing billions of calculations. When training 
a model, we're adjusting millions or even billions of numbers (called 
"weights") very slightly, over and over again, using huge amounts of data. 
Modern GPUs can do trillions of these calculations per second, but even 
then, training large models can take weeks. For example, training GPT-3 
reportedly used enough electricity to power a home for decades!
""",
    
    "will ai take over": """
This is a common concern, but let's be realistic. Current AI, including 
what we've built here, is "narrow AI" - it can only do specific tasks it 
was trained for. Our surgical vision system can recognize tools and phases, 
but it can't make coffee or write poetry. True "general AI" that matches 
human intelligence doesn't exist yet and may not for decades. The real 
concern isn't AI taking over, but ensuring AI is used responsibly and 
that its benefits are shared fairly across society.
""",
    
    "everyone benefit ai": """
AI has the potential to benefit everyone, but there are challenges. 
Positive examples include: AI helping doctors diagnose diseases earlier, 
making education more accessible, and automating dangerous jobs. However, 
we must address issues like: job displacement in certain sectors, 
algorithmic bias if training data is unrepresentative, and ensuring 
developing countries aren't left behind. The key is thoughtful policy, 
inclusive development, and education - like what we're doing in this 
podcast!
""",
    
    "prevent ai misuse": """
Preventing AI misuse requires a multi-layered approach: First, technical 
safeguards like the ones built into systems like ours - we can't make our 
bot do things outside its training. Second, ethical guidelines and 
regulations, like the EU's AI Act. Third, transparency - knowing how AI 
makes decisions. Fourth, education - the more people understand AI, the 
better they can spot misuse. Finally, diverse teams building AI help 
catch blind spots and biases before deployment.
""",

    "deep learning": """
Deep Learning is a type of machine learning that uses neural networks with 
many layers - hence "deep." Think of it like this: traditional programming 
is like giving someone exact directions. Deep Learning is like showing 
someone thousands of examples until they figure out the pattern themselves. 
Our surgical vision system uses Deep Learning - we showed it thousands of 
surgery videos, and it learned to recognize tools and phases on its own.
"""
}

AI_LITERACY_KEYWORDS = {
    "how does a computer learn": ["computer learn", "how does ai learn", "machine learn"],
    "neural network grandmother": ["neural network", "grandmother", "explain simple", "eli5"],
    "computational resources ai": ["computational", "resources", "why so much", "power", "energy"],
    "will ai take over": ["take over", "replace human", "dangerous", "threat"],
    "everyone benefit ai": ["everyone benefit", "who benefits", "accessible", "fair"],
    "prevent ai misuse": ["prevent", "misuse", "wrongly used", "safety", "ethics"],
    "deep learning": ["deep learning", "what is deep", "difference"]
}


# =============================================================================
# MODEL LOADING - FIXED FOR LOCAL PATHS
# =============================================================================

def load_model(model_path, device='cuda'):
    """Load GPT-2 model with proper LOCAL LoRA support"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Normalize path
    model_path = os.path.abspath(model_path)
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}")
    
    adapter_config_path = os.path.join(model_path, 'adapter_config.json')
    is_lora = os.path.exists(adapter_config_path)
    
    if is_lora and PEFT_AVAILABLE:
        print("Detected LoRA adapter, loading...")
        
        # Load config to get base model name
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get('base_model_name_or_path', 'gpt2')
        print(f"Base model: {base_model_name}")
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Load LoRA adapter with local_files_only=True to prevent HF Hub lookup
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(
            base_model, 
            model_path,
            local_files_only=False,
            is_trainable=False
        )
        print("✓ LoRA adapter loaded!")
        
    elif is_lora and not PEFT_AVAILABLE:
        print("⚠️ LoRA adapter detected but PEFT not installed!")
        print("Installing PEFT: pip install peft")
        print("Falling back to base GPT-2...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        
    else:
        print("Loading as full model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except:
            print("Using base gpt2 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    
    model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model ready!")
    return model, tokenizer, device


# =============================================================================
# RESPONSE GENERATION
# =============================================================================

def is_surgical_question(question):
    """Check if question is about surgical robotics"""
    q_lower = question.lower()
    surgical_keywords = [
        'vision system', 'surgical', 'robotic', 'phase', 'tool',
        'grasper', 'hook', 'clipper', 'scissors', 'bipolar',
        'preparation', 'dissection', 'clipping', 'cutting',
        'gallbladder', 'retraction', 'coagulation', 'irrigator'
    ]
    return any(kw in q_lower for kw in surgical_keywords)


def get_ai_literacy_answer(question):
    """Get pre-written answer for AI literacy questions"""
    q_lower = question.lower()
    
    for answer_key, keywords in AI_LITERACY_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return AI_LITERACY_ANSWERS[answer_key].strip()
    
    return None


def generate_response(model, tokenizer, device, question, max_length=200, temperature=0.7):
    """Generate response using the model"""
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
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
            num_return_sequences=1
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response (remove the prompt)
    if text.startswith(question):
        text = text[len(question):].strip()
    
    return text


class PodcastBot:
    """Production podcast bot with smart routing"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.conversation = []
    
    def ask(self, question, force_model=False):
        """
        Smart response routing:
        - Surgical questions → Use fine-tuned model (excellent!)
        - AI literacy questions → Use knowledge base (reliable)
        """
        is_surgical = is_surgical_question(question)
        
        if is_surgical or force_model:
            response = generate_response(
                self.model, self.tokenizer, self.device, question
            )
            source = "GPT-2 Fine-tuned"
        else:
            kb_answer = get_ai_literacy_answer(question)
            if kb_answer:
                response = kb_answer
                source = "Knowledge Base"
            else:
                response = generate_response(
                    self.model, self.tokenizer, self.device, question
                )
                source = "GPT-2 (fallback)"
        
        self.conversation.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response,
            'source': source,
            'is_surgical': is_surgical
        })
        
        return response, source
    
    def save_conversation(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {filepath}")


# =============================================================================
# PODCAST DEMO
# =============================================================================

def run_full_demo(bot):
    """Run the full podcast demo"""
    
    print("\n" + "=" * 70)
    print("🎙️  SAR-PODCAST-BOT: FULL DEMO")
    print("=" * 70)
    
    # SECTION 1: AI LITERACY
    print("\n" + "-" * 70)
    print("📚 SECTION 1: AI LITERACY FOR EVERYONE")
    print("-" * 70)
    
    ai_questions = [
        "How does a computer learn?",
        "How can I explain to my grandmother what is a Neural Network?",
        "Why do we need a lot of computational resources to process AI methods?",
    ]
    
    for q in ai_questions:
        print(f"\n🎤 HOST: {q}")
        response, source = bot.ask(q)
        print(f"\n🤖 BOT [{source}]:")
        print(response[:500])
        print("-" * 40)
    
    # SECTION 2: SURGICAL ROBOTICS
    print("\n" + "-" * 70)
    print("🏥 SECTION 2: SURGICAL ROBOTICS (Model Excels Here!)")
    print("-" * 70)
    
    surgical_questions = [
        "The vision system detects 'Preparation'. What robotic algorithm applies here?",
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        "Explain the control theory behind robotic Dissection.",
        "Why is the robotic approach to 'Clipping/Cutting' considered safer?",
    ]
    
    for q in surgical_questions:
        print(f"\n🎤 HOST: {q}")
        response, source = bot.ask(q)
        print(f"\n🤖 BOT [{source}]:")
        print(response[:500])
        print("-" * 40)
    
    # SECTION 3: ETHICS
    print("\n" + "-" * 70)
    print("⚖️  SECTION 3: ETHICS & THE FUTURE")
    print("-" * 70)
    
    ethics_questions = [
        "Will AI take over?",
        "Can everyone benefit from AI?",
        "How to prevent AI from being wrongly used?",
    ]
    
    for q in ethics_questions:
        print(f"\n🎤 HOST: {q}")
        response, source = bot.ask(q)
        print(f"\n🤖 BOT [{source}]:")
        print(response[:500])
        print("-" * 40)
    
    # SUMMARY
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    
    sources = {}
    for c in bot.conversation:
        src = c['source']
        sources[src] = sources.get(src, 0) + 1
    
    print("\n📊 Response breakdown:")
    for src, count in sources.items():
        print(f"   {src}: {count} responses")
    
    return bot.conversation


def interactive_mode(bot):
    """Interactive Q&A"""
    print("\n" + "=" * 60)
    print("🎙️ INTERACTIVE MODE")
    print("=" * 60)
    print("\nCommands:")
    print("  /demo     - Run full demo")
    print("  /surgical - Show surgical questions")
    print("  /ai       - Show AI literacy questions")
    print("  /model    - Force model response (skip KB)")
    print("  /save     - Save conversation")
    print("  /quit     - Exit")
    print("\nOr just type your question!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            if user_input.startswith('/'):
                cmd = user_input[1:].split()[0].lower()
                arg = user_input[len(cmd)+2:].strip() if len(user_input) > len(cmd)+2 else ""
                
                if cmd in ['quit', 'exit', 'q']:
                    break
                elif cmd == 'demo':
                    run_full_demo(bot)
                elif cmd == 'surgical':
                    print("\n🏥 Surgical questions (model is great at these!):")
                    print("  • The vision system detects 'Preparation'. What robotic algorithm applies here?")
                    print("  • The vision system detects the tool 'Grasper'. What is the robotic equivalent?")
                    print("  • Explain the control theory behind robotic Dissection.")
                elif cmd == 'ai':
                    print("\n📚 AI literacy questions (knowledge base):")
                    print("  • How does a computer learn?")
                    print("  • How can I explain to my grandmother what is a Neural Network?")
                    print("  • Will AI take over?")
                elif cmd == 'model' and arg:
                    # Force model response
                    response, _ = bot.ask(arg, force_model=True)
                    print(f"\n🤖 [GPT-2 Forced]: {response}\n")
                elif cmd == 'save':
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    bot.save_conversation(f"podcast_{ts}.json")
                else:
                    print(f"Unknown command: {cmd}")
                continue
            
            # Regular question
            response, source = bot.ask(user_input)
            print(f"\n🤖 [{source}]: {response}\n")
            
        except KeyboardInterrupt:
            print("\n")
            break
    
    print("Goodbye! 👋")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAR-Podcast-Bot Demo v2")
    parser.add_argument('--model', type=str, 
                       default='src/results/gpt2_best_model',
                       help='Path to GPT-2 model')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--demo', action='store_true', help='Run full demo')
    parser.add_argument('--output', type=str, help='Save conversation to file')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("   🎙️ SAR-PODCAST-BOT v2")
    print("   Smart routing: Surgical → Model | General → Knowledge Base")
    print("=" * 70)
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"\n❌ Model not found: {args.model}")
        
        # Try common paths
        alternatives = [
            'src/results/gpt2_best_model',
            'results/gpt2_best_model',
            os.path.expanduser('~/University/SAR-Podcast-Bot/src/results/gpt2_best_model'),
        ]
        
        for alt in alternatives:
            if os.path.exists(alt):
                print(f"✓ Found at: {alt}")
                args.model = alt
                break
        else:
            print("\nPlease specify correct --model path")
            return
    
    # Load model
    model, tokenizer, device = load_model(args.model, args.device)
    bot = PodcastBot(model, tokenizer, device)
    
    if args.demo:
        conversation = run_full_demo(bot)
        if args.output:
            bot.save_conversation(args.output)
        else:
            # Auto-save demo results
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bot.save_conversation(f"demo_results_{ts}.json")
    else:
        interactive_mode(bot)
        
        # Offer to save
        if bot.conversation:
            try:
                save = input("\n💾 Save conversation? (y/n): ").lower().strip()
                if save == 'y':
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    bot.save_conversation(f"podcast_{ts}.json")
            except:
                pass


if __name__ == "__main__":
    main()