"""
SAR-PODCAST-BOT: GPT-2 Diagnostic & Improved Prompting
======================================================
Diagnoses model issues and provides improved prompting strategies.

Issues identified:
1. No instruction/response separator during training
2. Tiny surgical dataset (~70 examples) vs large dialog (~87k)
3. Model memorized surgical data verbatim

Solutions implemented:
1. Structured prompting that mimics training format
2. More aggressive knowledge base fallback
3. Post-processing to clean fragment artifacts
"""

import os
import sys
import json
import re
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# =============================================================================
# KNOWLEDGE BASES
# =============================================================================

SURGICAL_KNOWLEDGE = {
    # Phase knowledge
    "Preparation": {
        "concept": "Visual SLAM",
        "response": "For Preparation, the critical concept is **Visual SLAM**. During preparation, a robot uses Visual SLAM (Simultaneous Localization and Mapping). It tracks feature points on the cavity walls to build a 3D depth map of the environment."
    },
    "CalotTriangleDissection": {
        "concept": "Motion Scaling",
        "response": "For Dissection, the critical concept is **Motion Scaling**. Dissection requires precision. A robot uses Motion Scaling (e.g., 5:1 ratio), converting a 5cm hand movement into a 1cm tool movement to prevent accidental cuts."
    },
    "Dissection": {
        "concept": "Motion Scaling",
        "response": "Dissection requires precision. A robot uses Motion Scaling (e.g., 5:1 ratio), converting a 5cm hand movement into a 1cm tool movement to prevent accidental cuts."
    },
    "ClippingCutting": {
        "concept": "Tremor Filtration",
        "response": "For Clipping/Cutting, the critical concept is **Tremor Filtration**. Clipping the artery requires steadiness. The robot uses a 6Hz Low-Pass Filter to remove the surgeon's natural hand tremors, ensuring the clip is placed perfectly."
    },
    "GallbladderDissection": {
        "concept": "Inverse Kinematics",
        "response": "For Gallbladder Dissection, the critical concept is **Inverse Kinematics**. The robot uses Inverse Kinematics to calculate the exact joint angles needed to maneuver the tool behind the gallbladder without the robot arms colliding."
    },
    "GallbladderRetraction": {
        "concept": "Active Constraints",
        "response": "For Gallbladder Retraction, the critical concept is **Active Constraints (Virtual Fixtures)**. To hold the organ safely, the robot uses Active Constraints. These are software 'invisible walls' that prevent the tool from slipping into the liver while maintaining tension."
    },
    "CleaningCoagulation": {
        "concept": "Augmented Reality",
        "response": "For Cleaning/Coagulation, the critical concept is **Augmented Reality (Fluorescence)**. To find bleeding spots, robotic systems overlay Augmented Reality feeds (like Firefly fluorescence) to highlight blood flow in green on the surgeon's screen."
    },
    "GallbladderPackaging": {
        "concept": "Master-Slave Teleoperation",
        "response": "For Gallbladder Packaging, the critical concept is **Master-Slave Teleoperation**. Bagging the specimen relies on Master-Slave Teleoperation algorithms. The system compensates for processing latency to ensure the robot moves instantly with the surgeon's hands."
    },
    
    # Tool knowledge
    "Grasper": {
        "concept": "Haptic Feedback Simulation",
        "response": "You are seeing a manual Grasper. In a robotic system, the equivalent uses **Haptic Feedback Simulation**. Robots lack touch. To compensate, algorithms use Visual Haptics—analyzing tissue deformation in the video to estimate force."
    },
    "Hook": {
        "concept": "EndoWrist (7 DOFs)",
        "response": "You are seeing a manual Hook. In a robotic system, the equivalent uses **EndoWrist (7 DOFs)**. The manual Hook is rigid. A robotic hook uses EndoWrist technology with 7 Degrees of Freedom, allowing it to rotate 360 degrees to hook tissue from behind."
    },
    "Clipper": {
        "concept": "Articulated Clip Applier",
        "response": "You are seeing a manual Clipper. In a robotic system, the equivalent uses **Articulated Clip Applier**. A robotic Clipper can articulate (bend) at the wrist, allowing the surgeon to place clips on the artery from the side without contorting their own arm."
    },
    "Scissors": {
        "concept": "Tremor Filtration",
        "response": "For Scissors, the critical concept is **Tremor Filtration (6Hz Low-Pass)**. Manual snipping can be shaky. The robotic scissors use a digital filter to remove the surgeon's physiological hand tremors, allowing for smooth, confident cuts."
    },
    "Bipolar": {
        "concept": "Multitasking Efficiency",
        "response": "For Bipolar, the critical concept is **Multitasking Efficiency**. The robotic Maryland Bipolar can dissect, grasp, and coagulate simultaneously due to its wrist articulation. This reduces instrument swaps compared to the rigid manual bipolar tool."
    },
    "Irrigator": {
        "concept": "Console Foot-Pedal Control",
        "response": "For Irrigator, the critical concept is **Console Foot-Pedal Control**. While manual irrigation requires a hand, robotic suction/irrigation is often controlled via foot pedals at the console, freeing both hands for operating instruments."
    },
    "SpecimenBag": {
        "concept": "Assistant Port Coordination",
        "response": "For Specimen Bag, the critical concept is **Assistant Port Coordination**. Robots struggle with soft, floppy objects like bags. The Specimen Bag is typically introduced through a special 'Assistant Port' by a human nurse."
    }
}

AI_LITERACY_KB = {
    "computer learn": "A computer learns through a process similar to how we learn from examples. We feed neural networks millions of examples, and they gradually adjust their internal 'weights' to recognize patterns. Each time they make a mistake, they adjust slightly to do better next time.",
    "neural network": "Think of a neural network like a very complex game of 'telephone.' Each person in the chain is like a 'neuron' - they take input, process it slightly, and pass it forward. By having millions of these simple operations working together, the network can learn to recognize patterns.",
    "computational resources": "AI needs lots of computing power because it's doing billions of calculations. When training a model, we're adjusting millions of numbers very slightly, over and over again, using huge amounts of data.",
    "take over": "Current AI is 'narrow AI' - it can only do specific tasks it was trained for. Our surgical vision system can recognize tools and phases, but it can't make coffee. True 'general AI' doesn't exist yet.",
    "benefit": "AI has the potential to benefit everyone, but we must address challenges like job displacement, algorithmic bias, and ensuring developing countries aren't left behind.",
    "misuse": "Preventing AI misuse requires technical safeguards, ethical guidelines, transparency, and education. Diverse teams building AI help catch blind spots before deployment.",
    "deep learning": "Deep Learning uses neural networks with many layers - hence 'deep.' Instead of giving exact directions, we show it thousands of examples until it figures out the pattern."
}


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path, device='cuda'):
    """Load model with LoRA support"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model_path = os.path.abspath(model_path)
    
    print(f"Loading from: {model_path}")
    
    adapter_config = os.path.join(model_path, 'adapter_config.json')
    
    if os.path.exists(adapter_config) and PEFT_AVAILABLE:
        with open(adapter_config) as f:
            config = json.load(f)
        base = config.get('base_model_name_or_path', 'gpt2')
        
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base)
        model = PeftModel.from_pretrained(model, model_path, local_files_only=True)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    
    model = model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


# =============================================================================
# RESPONSE CLEANING & VALIDATION
# =============================================================================

def clean_response(response):
    """Clean up fragment artifacts from model output"""
    # Remove leading fragments like "2?", "?", "considered safer?"
    # These are artifacts from training data concatenation
    
    # Pattern: starts with number/punctuation fragment
    response = re.sub(r'^[\d\?\.\!\,\:\;]+\s*', '', response)
    
    # Pattern: starts with partial phrase from training
    bad_starts = [
        'considered safer?',
        'determined how',
        'robotic algorithm',
        'Safety during',
        'is improved by',
    ]
    
    for bad in bad_starts:
        if response.lower().startswith(bad.lower()):
            # Try to find the actual response after this
            parts = response.split('.', 1)
            if len(parts) > 1 and len(parts[1].strip()) > 20:
                response = parts[1].strip()
    
    return response.strip()


def is_valid_response(question, response):
    """Check if response actually addresses the question"""
    q_lower = question.lower()
    r_lower = response.lower()
    
    # Check for fragment responses (too short, starts with punctuation)
    if len(response) < 20:
        return False
    
    if response[0] in '?.,!:;':
        return False
    
    # Check if surgical response to non-surgical question
    surgical_q_words = ['vision system', 'surgical', 'robotic', 'phase', 'tool', 
                        'grasper', 'hook', 'preparation', 'dissection']
    is_surgical_q = any(w in q_lower for w in surgical_q_words)
    
    surgical_r_words = ['endowrist', 'haptic', 'visual slam', 'motion scaling',
                        'tremor', 'clipper', 'grasper', 'robot']
    is_surgical_r = any(w in r_lower for w in surgical_r_words)
    
    # Surgical response to non-surgical question = invalid
    if not is_surgical_q and is_surgical_r:
        return False
    
    return True


def get_knowledge_base_response(question):
    """Try to get a response from knowledge base"""
    q_lower = question.lower()
    
    # Check surgical knowledge
    for key, info in SURGICAL_KNOWLEDGE.items():
        if key.lower() in q_lower:
            return info['response'], f"KB-Surgical ({key})"
    
    # Check AI literacy knowledge
    for key, response in AI_LITERACY_KB.items():
        if key in q_lower:
            return response, f"KB-AI ({key})"
    
    return None, None


# =============================================================================
# IMPROVED PROMPTING
# =============================================================================

def format_surgical_prompt(question):
    """
    Format prompt to better match training data structure.
    
    Training format was: instruction + " " + response
    So we need prompts that look like the training instructions.
    """
    # The model was trained on EXACT prompts like:
    # "The vision system detects 'Preparation'. What robotic algorithm applies here?"
    # 
    # For best results, use prompts that closely match these patterns.
    
    q_lower = question.lower()
    
    # Try to extract phase or tool from question
    phases = ['Preparation', 'Dissection', 'ClippingCutting', 'Clipping/Cutting',
              'GallbladderDissection', 'GallbladderRetraction', 
              'CleaningCoagulation', 'GallbladderPackaging']
    
    tools = ['Grasper', 'Hook', 'Clipper', 'Scissors', 'Bipolar', 'Irrigator', 'SpecimenBag']
    
    # Check if asking about a phase
    for phase in phases:
        if phase.lower() in q_lower or phase.replace('/', '').lower() in q_lower:
            # Reformat to match training prompt exactly
            return f"The vision system detects '{phase}'. What robotic algorithm applies here?"
    
    # Check if asking about a tool
    for tool in tools:
        if tool.lower() in q_lower:
            return f"The vision system detects the tool '{tool}'. What is the robotic equivalent?"
    
    # If no match, return original
    return question


def generate_with_improved_prompting(model, tokenizer, device, question, 
                                     max_length=200, temperature=0.7):
    """Generate response with improved prompting and post-processing"""
    
    # Step 1: Check knowledge base first
    kb_response, kb_source = get_knowledge_base_response(question)
    if kb_response:
        return kb_response, kb_source
    
    # Step 2: Format prompt to match training patterns
    formatted_prompt = format_surgical_prompt(question)
    
    # Step 3: Generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response (everything after prompt)
    if full_text.startswith(formatted_prompt):
        response = full_text[len(formatted_prompt):].strip()
    else:
        response = full_text.strip()
    
    # Step 4: Clean response
    response = clean_response(response)
    
    # Step 5: Validate response
    if not is_valid_response(question, response):
        # Fall back to knowledge base if available
        kb_response, kb_source = get_knowledge_base_response(formatted_prompt)
        if kb_response:
            return kb_response, f"KB-Fallback (invalid model response)"
        return response, "GPT-2 (WARNING: possibly invalid)"
    
    return response, "GPT-2 Fine-tuned"


# =============================================================================
# DIAGNOSTIC TESTS
# =============================================================================

def run_diagnostics(model, tokenizer, device):
    """Run diagnostic tests to understand model behavior"""
    
    print("\n" + "=" * 70)
    print("🔬 GPT-2 MODEL DIAGNOSTICS")
    print("=" * 70)
    
    # Test 1: Exact training prompts (should work perfectly)
    print("\n" + "-" * 70)
    print("TEST 1: Exact Training Prompts (Should Work Well)")
    print("-" * 70)
    
    exact_prompts = [
        "The vision system detects 'Preparation'. What robotic algorithm applies here?",
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        "Explain the control theory behind robotic Dissection.",
    ]
    
    for prompt in exact_prompts:
        response, source = generate_with_improved_prompting(
            model, tokenizer, device, prompt
        )
        print(f"\n📝 Q: {prompt}")
        print(f"🤖 [{source}]: {response[:200]}...")
    
    # Test 2: Rephrased surgical prompts (may have issues)
    print("\n" + "-" * 70)
    print("TEST 2: Rephrased Surgical Questions (Testing Generalization)")
    print("-" * 70)
    
    rephrased = [
        "What algorithm is used during the preparation phase?",
        "How do robots handle grasping in surgery?",
        "What makes robotic dissection safer?",
    ]
    
    for prompt in rephrased:
        response, source = generate_with_improved_prompting(
            model, tokenizer, device, prompt
        )
        print(f"\n📝 Q: {prompt}")
        print(f"🤖 [{source}]: {response[:200]}...")
    
    # Test 3: Non-surgical questions (should use KB or show overfitting)
    print("\n" + "-" * 70)
    print("TEST 3: Non-Surgical Questions (Testing Overfitting)")
    print("-" * 70)
    
    general = [
        "How are you today?",
        "What is the weather like?",
        "Tell me about machine learning.",
    ]
    
    for prompt in general:
        response, source = generate_with_improved_prompting(
            model, tokenizer, device, prompt
        )
        print(f"\n📝 Q: {prompt}")
        print(f"🤖 [{source}]: {response[:200]}...")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print("""
    FINDINGS:
    1. Model works well with EXACT training prompt formats
    2. Model struggles with rephrased questions (poor generalization)
    3. Model gives surgical responses to non-surgical questions (overfitting)
    
    ROOT CAUSES:
    - Training data: ~70 surgical examples vs ~87,000 dialog examples
    - No clear instruction/response separator during training
    - Model memorized exact patterns rather than learning concepts
    
    RECOMMENDED FIXES:
    1. Use knowledge base for non-surgical questions (implemented)
    2. Reformat prompts to match training patterns (implemented)
    3. Clean fragment artifacts from responses (implemented)
    4. For proper fix: Retrain with more data and clear separators
    """)


# =============================================================================
# IMPROVED PODCAST BOT
# =============================================================================

class ImprovedPodcastBot:
    """Podcast bot with improved prompting and fallbacks"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.history = []
    
    def ask(self, question):
        """Get response with all improvements applied"""
        response, source = generate_with_improved_prompting(
            self.model, self.tokenizer, self.device, question
        )
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response,
            'source': source
        })
        
        return response, source
    
    def save_history(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


def run_improved_demo(bot):
    """Run demo with improved bot"""
    
    print("\n" + "=" * 70)
    print("🎙️ IMPROVED PODCAST DEMO")
    print("=" * 70)
    
    questions = [
        # These should work great
        "The vision system detects 'Preparation'. What robotic algorithm applies here?",
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        
        # These will use KB or improved prompting
        "How does a computer learn?",
        "Will AI take over?",
        
        # These might still have issues
        "What stages are in this surgery?",
        "How are you?",
    ]
    
    for q in questions:
        print(f"\n📝 Q: {q}")
        response, source = bot.ask(q)
        print(f"🤖 [{source}]: {response[:300]}")
        print("-" * 40)
    
    return bot.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='src/results/core_results_v3/gpt2_best_model_v3')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--diagnose', action='store_true', help='Run diagnostics')
    parser.add_argument('--demo', action='store_true', help='Run improved demo')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("SAR-PODCAST-BOT: Diagnostic & Improved Version")
    print("=" * 70)
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    model, tokenizer, device = load_model(args.model, args.device)
    print("✓ Model loaded")
    
    if args.diagnose:
        run_diagnostics(model, tokenizer, device)
    
    bot = ImprovedPodcastBot(model, tokenizer, device)
    
    if args.demo:
        run_improved_demo(bot)
    
    if args.interactive or (not args.diagnose and not args.demo):
        print("\n🎙️ Interactive mode. Type /quit to exit.\n")
        
        while True:
            try:
                q = input("You: ").strip()
                if not q:
                    continue
                if q.lower() in ['/quit', '/exit', '/q']:
                    break
                
                response, source = bot.ask(q)
                print(f"\n🤖 [{source}]: {response}\n")
                
            except KeyboardInterrupt:
                break
        
        if bot.history:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bot.save_history(f"improved_chat_{ts}.json")


if __name__ == "__main__":
    main()