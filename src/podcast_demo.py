"""
Vision-to-Language Demo for Podcast
=====================================
Shows how the Core model interprets vision system outputs.
Perfect for demonstrating the integration in your podcast!
"""

import sys
import os

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ========================================
# SIMULATED VISION OUTPUTS
# (In real use, these come from CNN+LSTM)
# ========================================
SIMULATED_VISION_RESULTS = [
    {
        "phase": "Preparation",
        "tools": ["Grasper"],
        "confidence": 0.87,
        "timestamp": "00:00 - 00:45"
    },
    {
        "phase": "CalotTriangleDissection",
        "tools": ["Hook", "Grasper"],
        "confidence": 0.92,
        "timestamp": "00:45 - 05:30"
    },
    {
        "phase": "ClippingCutting",
        "tools": ["Clipper", "Scissors"],
        "confidence": 0.89,
        "timestamp": "05:30 - 08:15"
    },
    {
        "phase": "GallbladderDissection",
        "tools": ["Hook", "Bipolar"],
        "confidence": 0.91,
        "timestamp": "08:15 - 15:00"
    },
]

# ========================================
# LOAD CORE MODEL
# ========================================
def load_model():
    """Load the trained TinyLlama model"""
    print("Loading TinyLlama Core model...")
    
    # Try multiple paths
    possible_paths = [
        os.path.join(src_dir, "src/results/model_final/best_model"),
        os.path.join(src_dir, "src/results/tinyllama_surgical"),
        "src/results/model_final/best_model",
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("ERROR: Model not found. Using base TinyLlama (won't have surgical knowledge)")
        model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"  Loading from: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if it's a LoRA adapter or full model
    adapter_config = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config):
        # Load base + LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print(f"  ✓ Model loaded on {device}")
    
    return model, tokenizer, device


def generate_response(model, tokenizer, device, question, max_length=200):
    """Generate response using TinyLlama chat format"""
    prompt = f"<|user|>\n{question}</s>\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    elif question in response:
        response = response.split(question)[-1].strip()
    
    return response


def vision_to_language_demo(model, tokenizer, device):
    """
    Demo: Show how vision outputs are interpreted by the Core model
    """
    print("\n" + "=" * 70)
    print("VISION → LANGUAGE INTEGRATION DEMO")
    print("=" * 70)
    print("\nThis demonstrates how SAR-Podcast-Bot interprets vision system outputs.\n")
    
    for i, vision_result in enumerate(SIMULATED_VISION_RESULTS, 1):
        print(f"\n{'─' * 70}")
        print(f"SEGMENT {i}: {vision_result['timestamp']}")
        print(f"{'─' * 70}")
        
        # Display vision output
        print(f"\n🎥 VISION SYSTEM OUTPUT:")
        print(f"   Phase detected: {vision_result['phase']} ({vision_result['confidence']*100:.0f}% confidence)")
        print(f"   Tools detected: {', '.join(vision_result['tools'])}")
        
        # Construct question based on vision output
        tools_str = " and ".join(vision_result['tools'])
        question = f"The vision system detects the '{vision_result['phase']}' phase with {tools_str} visible. What is happening and what robotic concepts apply?"
        
        print(f"\n🤖 QUESTION TO CORE MODEL:")
        print(f"   \"{question}\"")
        
        # Get response
        print(f"\n💬 CORE MODEL RESPONSE:")
        response = generate_response(model, tokenizer, device, question)
        
        # Format response nicely
        for line in response.split('. '):
            if line.strip():
                print(f"   {line.strip()}.")
        
        input("\n[Press Enter to continue to next segment...]")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


def ai_literacy_demo(model, tokenizer, device):
    """
    Demo: Required AI literacy questions from coursework
    """
    print("\n" + "=" * 70)
    print("AI LITERACY DEMO (Required for Podcast)")
    print("=" * 70)
    
    required_questions = [
        "How does a computer learn?",
        "How can I explain to my grandmother what is a Neural Network?",
        "Why do we need a lot of computational resources to process AI methods?",
        "Will AI take over?",
        "Can everyone benefit from AI?",
        "How to prevent AI from being wrongly used?",
    ]
    
    for i, question in enumerate(required_questions, 1):
        print(f"\n{'─' * 70}")
        print(f"Q{i}: {question}")
        print(f"{'─' * 70}")
        
        response = generate_response(model, tokenizer, device, question)
        print(f"\n{response}")
        
        input("\n[Press Enter for next question...]")


def interactive_demo(model, tokenizer, device):
    """Interactive mode for podcast recording"""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Type questions to ask the bot. Commands:")
    print("  /vision  - Run vision interpretation demo")
    print("  /ai      - Run AI literacy questions")
    print("  /quit    - Exit")
    print("=" * 70)
    
    while True:
        question = input("\n🎤 You: ").strip()
        
        if not question:
            continue
        
        if question.lower() == "/quit":
            break
        elif question.lower() == "/vision":
            vision_to_language_demo(model, tokenizer, device)
            continue
        elif question.lower() == "/ai":
            ai_literacy_demo(model, tokenizer, device)
            continue
        
        response = generate_response(model, tokenizer, device, question)
        print(f"\n🤖 Bot: {response}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision", action="store_true", help="Run vision demo")
    parser.add_argument("--ai", action="store_true", help="Run AI literacy demo")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model()
    
    if args.vision:
        vision_to_language_demo(model, tokenizer, device)
    elif args.ai:
        ai_literacy_demo(model, tokenizer, device)
    elif args.interactive:
        interactive_demo(model, tokenizer, device)
    else:
        # Default: run all demos
        print("\nRunning full demo sequence...")
        vision_to_language_demo(model, tokenizer, device)
        ai_literacy_demo(model, tokenizer, device)
        interactive_demo(model, tokenizer, device)