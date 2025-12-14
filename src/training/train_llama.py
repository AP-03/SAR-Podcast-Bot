"""
Fine-Tuning TinyLlama-1.1B for Surgical Robotics
=================================================
Why TinyLlama instead of GPT-2?

| Model      | Parameters | Pre-training | Surgery Knowledge |
|------------|------------|--------------|-------------------|
| GPT-2      | 124M       | 40GB (2019)  | Almost none       |
| TinyLlama  | 1.1B       | 3T tokens    | Yes, basic        |

TinyLlama is 9x larger and ALREADY knows about surgery, AI, etc.
We just need to teach it our specific format and knowledge base.

Requirements:
    pip install transformers peft torch accelerate bitsandbytes

Run:
    python train_tinyllama.py
"""

import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("ERROR: PEFT required. Install with: pip install peft")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model - TinyLlama is the sweet spot (1.1B params, runs on 8GB VRAM)
    'model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    
    # Alternatives (uncomment to try):
    # 'model_name': 'microsoft/phi-2',           # 2.7B - needs 12GB+ VRAM
    # 'model_name': 'google/gemma-2b',           # 2B - needs 10GB+ VRAM
    # 'model_name': 'Qwen/Qwen1.5-1.8B-Chat',    # 1.8B - good alternative
    
    # Quantization (reduces VRAM usage)
    'use_4bit': True,  # Set False if you have 16GB+ VRAM
    
    # Training
    'max_length': 512,
    'batch_size': 4,      # Small batch for memory
    'gradient_accumulation': 8,  # Effective batch = 4 * 8 = 32
    'learning_rate': 2e-4,
    'epochs': 10,
    'warmup_ratio': 0.1,
    
    # LoRA - efficient fine-tuning
    'lora_r': 32,
    'lora_alpha': 64,
    'lora_dropout': 0.05,
    
    # Output
    'output_dir': 'results/tinyllama_surgical',
}


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

PHASE_KNOWLEDGE = {
    "Preparation": {
        "concept": "Visual SLAM",
        "explanation": "During preparation, a robot uses Visual SLAM (Simultaneous Localization and Mapping). It tracks feature points on the cavity walls to build a 3D depth map of the surgical environment."
    },
    "CalotTriangleDissection": {
        "concept": "Motion Scaling",
        "explanation": "Dissection requires extreme precision. A robot uses Motion Scaling (e.g., 5:1 ratio), converting a 5cm hand movement into a 1cm tool movement to prevent accidental tissue damage."
    },
    "ClippingCutting": {
        "concept": "Tremor Filtration",
        "explanation": "Clipping the cystic artery requires absolute steadiness. The robot uses a 6Hz Low-Pass Filter to remove the surgeon's natural hand tremors, ensuring perfect clip placement."
    },
    "GallbladderRetraction": {
        "concept": "Active Constraints",
        "explanation": "To hold the organ safely, the robot uses Active Constraints (Virtual Fixtures). These software-defined 'invisible walls' prevent the tool from slipping into the liver while maintaining proper tension."
    },
    "GallbladderDissection": {
        "concept": "Inverse Kinematics",
        "explanation": "The robot uses Inverse Kinematics to calculate exact joint angles needed to maneuver tools behind the gallbladder without arm collisions, enabling access to difficult angles."
    },
    "CleaningCoagulation": {
        "concept": "Augmented Reality",
        "explanation": "To identify bleeding spots, robotic systems overlay Augmented Reality feeds (like Firefly fluorescence imaging) to highlight blood flow in green on the surgeon's display."
    },
    "GallbladderPackaging": {
        "concept": "Master-Slave Teleoperation",
        "explanation": "Specimen retrieval relies on Master-Slave Teleoperation with latency compensation algorithms, ensuring the robot responds instantly to the surgeon's hand movements at the console."
    }
}

TOOL_KNOWLEDGE = {
    "Hook": {
        "concept": "EndoWrist (7 DOFs)",
        "explanation": "Unlike the rigid manual hook, a robotic hook uses EndoWrist technology with 7 Degrees of Freedom, allowing 360-degree rotation to approach tissue from any angle."
    },
    "Grasper": {
        "concept": "Haptic Feedback Simulation",
        "explanation": "Since robots lack touch sensation, Visual Haptics algorithms analyze tissue deformation in real-time video to estimate and display the gripping force being applied."
    },
    "Clipper": {
        "concept": "Articulated Clip Applier",
        "explanation": "A robotic clip applier can articulate (bend) at the wrist, allowing optimal clip placement angles without requiring awkward surgeon hand positions."
    },
    "Scissors": {
        "concept": "Tremor Filtration",
        "explanation": "Robotic scissors incorporate a 6Hz Low-Pass Filter that removes physiological hand tremors, enabling smooth, controlled cuts even in delicate tissue."
    },
    "Bipolar": {
        "concept": "Multitasking Efficiency",
        "explanation": "The robotic Maryland Bipolar can simultaneously dissect, grasp, and coagulate tissue thanks to wrist articulation, significantly reducing instrument exchanges."
    },
    "Irrigator": {
        "concept": "Foot-Pedal Control",
        "explanation": "Robotic irrigation and suction are controlled via foot pedals at the surgeon console, keeping both hands free for primary instrument manipulation."
    }
}

AI_KNOWLEDGE = {
    "computer_learning": "A computer learns by analyzing millions of examples and gradually adjusting internal parameters called 'weights'. Like a child learning to recognize cats from many pictures, neural networks improve by seeing patterns across vast amounts of data.",
    "neural_network": "A neural network is a system of interconnected processing nodes arranged in layers. Each layer transforms the input data, extracting increasingly abstract features - from simple edges to complex concepts.",
    "deep_learning": "Deep learning uses neural networks with many layers (hence 'deep'). Early layers detect simple patterns like edges, middle layers combine these into shapes, and final layers recognize complete objects or concepts.",
    "ai_surgery": "AI in surgery serves as an intelligent assistant rather than a replacement. It provides real-time analysis, warns of potential complications, and enhances visualization - but critical decisions remain with trained human surgeons.",
    "ai_safety": "Surgical AI safety includes hardware limits on force and speed, software constraints preventing dangerous movements, real-time monitoring systems, and most importantly - keeping qualified human surgeons in control at all times."
}


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_training_data():
    """Generate diverse Q&A pairs from knowledge base"""
    examples = []
    
    # Phase questions - multiple phrasings
    phase_templates = [
        ("The vision system detects '{phase}'. What robotic algorithm applies here?",
         "For {phase}, the critical concept is **{concept}**. {explanation}"),
        ("What technology enables robotic {phase}?",
         "Robotic {phase} relies on **{concept}**. {explanation}"),
        ("Explain how robots handle the {phase} phase.",
         "During {phase}, robots use **{concept}**. {explanation}"),
        ("What algorithm is used during {phase}?",
         "The key algorithm for {phase} is **{concept}**. {explanation}"),
        ("How does a surgical robot perform {phase}?",
         "{phase} is accomplished using **{concept}**. {explanation}"),
        ("What makes robotic {phase} safer than manual surgery?",
         "Safety during {phase} is enhanced by **{concept}**. {explanation}"),
    ]
    
    for phase, info in PHASE_KNOWLEDGE.items():
        display = phase.replace("CalotTriangle", "Calot Triangle ").replace("Gallbladder", "Gallbladder ")
        for q_tmpl, a_tmpl in phase_templates:
            examples.append({
                "instruction": q_tmpl.format(phase=display),
                "response": a_tmpl.format(phase=display, concept=info['concept'], explanation=info['explanation'])
            })
    
    # Tool questions
    tool_templates = [
        ("The vision system detects the tool '{tool}'. What is the robotic equivalent?",
         "The robotic equivalent of the {tool} uses **{concept}**. {explanation}"),
        ("What technology does a robotic {tool} use?",
         "A robotic {tool} employs **{concept}**. {explanation}"),
        ("How does the robotic {tool} differ from the manual version?",
         "Unlike manual tools, the robotic {tool} uses **{concept}**. {explanation}"),
        ("Explain the control system for robotic {tool}.",
         "The robotic {tool} is controlled via **{concept}**. {explanation}"),
    ]
    
    for tool, info in TOOL_KNOWLEDGE.items():
        for q_tmpl, a_tmpl in tool_templates:
            examples.append({
                "instruction": q_tmpl.format(tool=tool),
                "response": a_tmpl.format(tool=tool, concept=info['concept'], explanation=info['explanation'])
            })
    
    # AI literacy questions
    ai_questions = [
        ("How does a computer learn?", AI_KNOWLEDGE["computer_learning"]),
        ("How do machines learn from data?", AI_KNOWLEDGE["computer_learning"]),
        ("Explain machine learning in simple terms.", AI_KNOWLEDGE["computer_learning"]),
        ("What is a neural network?", AI_KNOWLEDGE["neural_network"]),
        ("How do neural networks work?", AI_KNOWLEDGE["neural_network"]),
        ("What is deep learning?", AI_KNOWLEDGE["deep_learning"]),
        ("How is deep learning different from regular machine learning?", AI_KNOWLEDGE["deep_learning"]),
        ("Will AI replace surgeons?", AI_KNOWLEDGE["ai_surgery"]),
        ("Can AI perform surgery autonomously?", AI_KNOWLEDGE["ai_surgery"]),
        ("How do we ensure AI in surgery is safe?", AI_KNOWLEDGE["ai_safety"]),
        ("What safety measures exist for surgical AI?", AI_KNOWLEDGE["ai_safety"]),
    ]
    
    for q, a in ai_questions:
        examples.append({"instruction": q, "response": a})
    
    # Surgery overview
    overview = [
        ("What are the phases of a cholecystectomy?",
         "A laparoscopic cholecystectomy has 7 main phases: Preparation (Visual SLAM), Calot Triangle Dissection (Motion Scaling), Clipping/Cutting (Tremor Filtration), Gallbladder Dissection (Inverse Kinematics), Gallbladder Retraction (Active Constraints), Cleaning/Coagulation (AR Fluorescence), and Gallbladder Packaging (Teleoperation)."),
        ("What stages are in gallbladder removal surgery?",
         "Gallbladder removal proceeds through: Preparation, Calot Triangle Dissection, Clipping and Cutting of the cystic duct and artery, Gallbladder Dissection from the liver bed, Retraction, Cleaning/Coagulation to ensure hemostasis, and finally Packaging for removal."),
        ("List the robotic algorithms used in cholecystectomy.",
         "Key robotic algorithms include: Visual SLAM for spatial mapping, Motion Scaling for precision, Tremor Filtration for steadiness, Inverse Kinematics for tool positioning, Active Constraints for safety boundaries, AR Fluorescence for visualization, and Master-Slave Teleoperation for control."),
    ]
    
    for q, a in overview:
        examples.append({"instruction": q, "response": a})
    
    # Conversational (minimal)
    greetings = [
        ("Hello!", "Hello! I'm the SAR-Podcast-Bot, an AI assistant specializing in surgical robotics. I can explain robotic surgery phases, tool technologies, and AI concepts. What would you like to learn about?"),
        ("Hi there.", "Hi! I'm here to discuss surgical robotics and AI. Feel free to ask about specific surgical phases, tools, or how AI and machine learning work."),
        ("What can you help me with?", "I can explain: surgical phases and their robotic control algorithms, how robotic surgical tools work, and fundamental AI/machine learning concepts. What interests you?"),
    ]
    
    for q, a in greetings:
        examples.append({"instruction": q, "response": a})
    
    return examples


# =============================================================================
# CHAT TEMPLATE FORMATTING
# =============================================================================

def format_for_tinyllama(instruction, response, tokenizer):
    """Format using TinyLlama's chat template"""
    # TinyLlama uses ChatML format
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ]
    
    # Use the model's chat template
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except:
        # Fallback format
        text = f"<|user|>\n{instruction}</s>\n<|assistant|>\n{response}</s>"
    
    return text


def format_prompt_for_inference(instruction, tokenizer):
    """Format prompt for generation"""
    messages = [{"role": "user", "content": instruction}]
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        text = f"<|user|>\n{instruction}</s>\n<|assistant|>\n"
    
    return text


# =============================================================================
# DATASET CLASS
# =============================================================================

class ChatDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        
        for ex in examples:
            text = format_for_tinyllama(ex['instruction'], ex['response'], tokenizer)
            self.texts.append(text)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_for_training(config):
    """Load model with optional 4-bit quantization"""
    
    print(f"\nLoading: {config['model_name']}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config for 4-bit (saves VRAM)
    if config['use_4bit']:
        print("Using 4-bit quantization (saves VRAM)")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config['model_name'],
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
        except Exception as e:
            print(f"4-bit loading failed: {e}")
            print("Falling back to float16...")
            model = AutoModelForCausalLM.from_pretrained(
                config['model_name'],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Apply LoRA
    if PEFT_AVAILABLE:
        print("Applying LoRA...")
        
        # Find target modules (different for each model)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("=" * 70)
    print("FINE-TUNING TINYLLAMA FOR SURGICAL ROBOTICS")
    print("=" * 70)
    
    # Setup paths
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent if script_dir.name == 'training' else script_dir
    output_dir = src_dir / CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output: {output_dir}")
    
    # Load model
    model, tokenizer = load_model_for_training(CONFIG)
    
    # Generate data
    print("\nGenerating training data...")
    examples = generate_training_data()
    print(f"Base examples: {len(examples)}")
    
    # Upsample for more training
    examples = examples * 5  # 5x repeat
    random.shuffle(examples)
    print(f"After upsampling: {len(examples)}")
    
    # Split
    split_idx = int(len(examples) * 0.9)
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]
    
    train_dataset = ChatDataset(train_data, tokenizer, CONFIG['max_length'])
    val_dataset = ChatDataset(val_data, tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    
    total_steps = len(train_loader) * CONFIG['epochs'] // CONFIG['gradient_accumulation']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    # Simple linear warmup
    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_loss = float('inf')
    train_losses, val_losses = [], []
    global_step = 0
    
    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        optimizer.zero_grad()
        
        for i, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / CONFIG['gradient_accumulation']
            loss.backward()
            
            if (i + 1) % CONFIG['gradient_accumulation'] == 0:
                # Adjust learning rate
                lr_scale = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg['lr'] = CONFIG['learning_rate'] * lr_scale
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            total_loss += outputs.loss.item()
            pbar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_pretrained(str(output_dir / 'best_model'))
            tokenizer.save_pretrained(str(output_dir / 'best_model'))
            print(f"  ✓ Saved best model")
        
        # Test generation every 3 epochs
        if (epoch + 1) % 3 == 0:
            print("\n  📝 Sample outputs:")
            test_qs = [
                "The vision system detects 'Preparation'. What robotic algorithm applies here?",
                "How does a computer learn?",
            ]
            for q in test_qs:
                resp = generate_response(model, tokenizer, q)
                print(f"    Q: {q[:40]}...")
                print(f"    A: {resp[:80]}...")
            print()
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-o', label='Train')
    plt.plot(val_losses, 'r-s', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TinyLlama Fine-tuning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(output_dir / 'training_curves.png'))
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    test_prompts = [
        "The vision system detects 'Preparation'. What robotic algorithm applies here?",
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        "What are the phases of a cholecystectomy?",
        "How does a computer learn?",
        "What is deep learning?",
        "Will AI replace surgeons?",
        "Hello!",
    ]
    
    for q in test_prompts:
        response = generate_response(model, tokenizer, q)
        print(f"\nQ: {q}")
        print(f"A: {response}")
    
    print(f"\n✓ Training complete!")
    print(f"  Best model: {output_dir / 'best_model'}")
    print(f"  Best val loss: {best_loss:.4f}")
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)


def generate_response(model, tokenizer, instruction, max_new_tokens=150):
    """Generate a response"""
    prompt = format_prompt_for_inference(instruction, tokenizer)
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    elif "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    # Clean up
    response = response.replace("</s>", "").strip()
    
    return response


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    train()