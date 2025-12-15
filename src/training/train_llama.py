"""
TinyLlama Fine-Tuning for SAR-Podcast-Bot
==========================================
READS FROM YOUR ACTUAL KNOWLEDGE BASE FILES:
- src/dataset/Surgical_Robotics/phase_to_control_mapping.json
- src/dataset/Surgical_Robotics/tool_to_robot_mapping.json

Also includes project-specific Q&A about:
- Your CNN (ResNet-50) for tool/phase detection
- Your LSTM for temporal smoothing
- Your full vision pipeline
- AI literacy questions

Run from SAR-Podcast-Bot directory:
    python src/training/train_model.py

Requirements:
    pip install transformers peft accelerate bitsandbytes torch
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
    print("ERROR: PEFT required. Install: pip install peft")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model - TinyLlama works on 8GB VRAM
    'model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    
    # Set to False if you have issues with bitsandbytes on Windows
    'use_4bit': True,
    
    # Training
    'max_length': 512,
    'batch_size': 4,
    'gradient_accumulation': 8,
    'learning_rate': 2e-4,
    'epochs': 10,
    'warmup_ratio': 0.1,
    
    # LoRA
    'lora_r': 32,
    'lora_alpha': 64,
    'lora_dropout': 0.05,
    
    # Data
    'upsample_factor': 8,  # Repeat examples for more training
    
    # Output
    'output_dir': 'results/model_final',
}


# =============================================================================
# PROJECT-SPECIFIC KNOWLEDGE (Your actual system!)
# =============================================================================

PROJECT_KNOWLEDGE = {
    "system_overview": {
        "question": "What is the SAR-Podcast-Bot?",
        "answer": "SAR-Podcast-Bot is an AI system that analyzes surgical videos and generates educational narration. It uses computer vision (CNN + LSTM) to detect surgical phases and tools, then uses language models to explain the robotic control concepts that would apply to each phase."
    },
    "vision_pipeline": {
        "question": "How does the vision system work?",
        "answer": "The vision pipeline has two stages: First, a ResNet-50 CNN processes each video frame to detect surgical tools (multi-label) and phases (single-label). Second, an LSTM with attention processes the temporal sequence of CNN features to smooth predictions and capture action patterns over time."
    },
    "cnn_model": {
        "question": "What CNN model do you use?",
        "answer": "We use a ResNet-50 CNN (ToolCNN) pre-trained on ImageNet and fine-tuned on the Cholec80 surgical dataset. It performs multi-task learning: simultaneously predicting which tools are present (multi-label classification) and which surgical phase is occurring (single-label classification)."
    },
    "lstm_model": {
        "question": "What is the LSTM used for?",
        "answer": "The ActionLSTM with attention mechanism processes temporal sequences of CNN features. It smooths frame-by-frame predictions and captures longer-term patterns in surgical actions. This reduces noise and improves phase detection accuracy by considering context from surrounding frames."
    },
    "dataset": {
        "question": "What dataset was used for training?",
        "answer": "The vision models were trained on Cholec80, a dataset of 80 laparoscopic cholecystectomy (gallbladder removal) videos with annotations for 7 surgical phases and 7 tool types. The language model was fine-tuned on surgical robotics Q&A pairs generated from a knowledge base."
    },
    "tools_detected": {
        "question": "What surgical tools can the system detect?",
        "answer": "The system detects 7 surgical tools: Grasper, Bipolar forceps, Hook (electrocautery), Scissors, Clipper, Irrigator, and Specimen Bag. Each tool has a corresponding robotic equivalent with specific control algorithms."
    },
    "phases_detected": {
        "question": "What surgical phases can the system detect?",
        "answer": "The system detects 7 surgical phases: Preparation, Calot Triangle Dissection, Clipping and Cutting, Gallbladder Dissection, Gallbladder Retraction, Cleaning and Coagulation, and Gallbladder Packaging. Each phase uses different robotic control algorithms."
    },
    "how_narration_works": {
        "question": "How does the narration generation work?",
        "answer": "The system processes a surgical video through the CNN and LSTM to get phase and tool predictions. Then, for each detected phase, the language model generates an explanation of the relevant robotic control concepts, creating an educational narration of the surgery."
    },
    "cholecystectomy": {
        "question": "What surgery does this system analyze?",
        "answer": "The system analyzes laparoscopic cholecystectomy, which is the minimally invasive surgical removal of the gallbladder. This is one of the most common surgical procedures, making it ideal for AI-assisted surgical education."
    }
}

AI_LITERACY_KNOWLEDGE = {
    "computer_learning": {
        "questions": [
            "How does a computer learn?",
            "How do machines learn from data?",
            "Explain machine learning simply."
        ],
        "answer": "A computer learns by analyzing many examples and adjusting internal parameters called 'weights'. Like a child learning to recognize cats from many pictures, neural networks gradually improve by finding patterns across thousands or millions of examples. Each mistake helps the model adjust its weights to do better next time."
    },
    "neural_network": {
        "questions": [
            "What is a neural network?",
            "How do neural networks work?",
            "Explain neural networks to a beginner."
        ],
        "answer": "A neural network is a system of interconnected processing nodes arranged in layers, inspired by the human brain. Each layer transforms the input data - early layers might detect simple features like edges, middle layers combine these into shapes, and final layers recognize complete objects or concepts. The network learns by adjusting connection strengths during training."
    },
    "deep_learning": {
        "questions": [
            "What is deep learning?",
            "How is deep learning different from machine learning?",
            "Why is it called deep learning?"
        ],
        "answer": "Deep learning uses neural networks with many layers (hence 'deep'). Each layer learns increasingly abstract features - from simple edges to complex patterns to complete concepts. This hierarchical learning enables tasks like image recognition, speech understanding, and language generation that were previously impossible for computers."
    },
    "cnn_explanation": {
        "questions": [
            "What is a CNN?",
            "How do Convolutional Neural Networks work?",
            "Why use CNNs for images?"
        ],
        "answer": "A Convolutional Neural Network (CNN) is designed specifically for processing images. It uses filters that slide across the image to detect patterns like edges, textures, and shapes. By stacking many layers of these filters, CNNs can recognize complex objects. They're used in our system to identify surgical tools and phases from video frames."
    },
    "lstm_explanation": {
        "questions": [
            "What is an LSTM?",
            "How do LSTMs work?",
            "Why use LSTM for sequences?"
        ],
        "answer": "LSTM (Long Short-Term Memory) is a type of neural network designed for sequential data like video or text. Unlike regular networks, LSTMs have memory cells that can remember information over long sequences. This makes them ideal for our surgical video analysis, where understanding the current frame requires context from previous frames."
    },
    "ai_surgery": {
        "questions": [
            "Will AI replace surgeons?",
            "Can AI perform surgery autonomously?",
            "What is the future of AI in surgery?"
        ],
        "answer": "AI currently serves as an intelligent assistant to surgeons rather than a replacement. It can analyze images, provide warnings, and enhance visualization, but critical decisions remain with trained human surgeons. Our system, for example, provides educational narration and analysis but doesn't control any surgical instruments."
    },
    "ai_safety": {
        "questions": [
            "How do we ensure AI in surgery is safe?",
            "What safety measures exist for surgical AI?",
            "Is AI in healthcare safe?"
        ],
        "answer": "Surgical AI safety involves multiple layers: extensive testing and validation, regulatory approval processes, hardware limits on robotic systems, software constraints, and most importantly - keeping qualified human surgeons in control at all times. AI systems like ours focus on analysis and education rather than direct patient intervention."
    }
}

CONVERSATIONAL = {
    "greetings": [
        ("Hello!", "Hello! I'm the SAR-Podcast-Bot, an AI assistant that explains surgical robotics and our vision-based surgical analysis system. I can tell you about surgical phases, tools, robotic control algorithms, or how our CNN and LSTM models work. What would you like to know?"),
        ("Hi there.", "Hi! I specialize in surgical robotics education. Ask me about surgical phases, tools, our computer vision pipeline, or AI and machine learning concepts."),
        ("Hey", "Hey! I'm here to explain surgical robotics and AI. What would you like to learn about?"),
        ("Good morning", "Good morning! Ready to discuss surgical robotics, our vision system, or AI concepts. What interests you?"),
    ],
    "capabilities": [
        ("What can you help with?", "I can explain: the 7 surgical phases and their robotic control algorithms, the 7 surgical tools and their robotic equivalents, how our CNN and LSTM vision system works, and general AI/machine learning concepts. What would you like to explore?"),
        ("What do you know about?", "I'm knowledgeable about: laparoscopic cholecystectomy surgery, robotic surgical systems and control algorithms, computer vision for surgical analysis, and AI/deep learning fundamentals. Feel free to ask about any of these!"),
        ("What can you do?", "I analyze surgical videos and explain the robotic control concepts for each phase. I can describe our vision pipeline (CNN + LSTM), explain surgical tools and phases, and answer questions about AI and machine learning."),
    ],
    "about": [
        ("Who made you?", "I was created as part of the SAR-Podcast-Bot project, a deep learning coursework that combines computer vision and language models for surgical education."),
        ("What are you?", "I'm an AI assistant that's part of the SAR-Podcast-Bot system. I use a language model fine-tuned on surgical robotics knowledge to explain surgical concepts and our vision-based analysis pipeline."),
    ]
}


# =============================================================================
# LOAD KNOWLEDGE BASE FROM JSON FILES
# =============================================================================

def load_knowledge_base(src_dir):
    """Load the actual JSON knowledge base files from the repo"""
    
    phase_path = src_dir / 'dataset' / 'Surgical_Robotics' / 'phase_to_control_mapping.json'
    tool_path = src_dir / 'dataset' / 'Surgical_Robotics' / 'tool_to_robot_mapping.json'
    
    # Load phase knowledge
    if phase_path.exists():
        with open(phase_path, 'r', encoding='utf-8') as f:
            phase_knowledge = json.load(f)
        print(f"✓ Loaded {len(phase_knowledge)} phases from {phase_path.name}")
    else:
        print(f"⚠ Phase knowledge not found at {phase_path}")
        phase_knowledge = {}
    
    # Load tool knowledge
    if tool_path.exists():
        with open(tool_path, 'r', encoding='utf-8') as f:
            tool_knowledge = json.load(f)
        print(f"✓ Loaded {len(tool_knowledge)} tools from {tool_path.name}")
    else:
        print(f"⚠ Tool knowledge not found at {tool_path}")
        tool_knowledge = {}
    
    return phase_knowledge, tool_knowledge


# =============================================================================
# GENERATE TRAINING EXAMPLES
# =============================================================================

def generate_training_examples(phase_kb, tool_kb):
    """Generate diverse Q&A pairs from all knowledge sources"""
    examples = []
    
    # =========================================================================
    # 1. PHASE QUESTIONS (from JSON knowledge base)
    # =========================================================================
    phase_templates = [
        ("The vision system detects '{phase}'. What robotic algorithm applies here?",
         "For {phase}, the critical concept is **{concept}**. {fact}"),
        ("What robotic technology is used during {phase}?",
         "During {phase}, robots use **{concept}**. {fact}"),
        ("Explain how a surgical robot handles {phase}.",
         "{phase} relies on **{concept}**. {fact}"),
        ("What algorithm helps robots during the {phase} phase?",
         "The key algorithm for {phase} is **{concept}**. {fact}"),
        ("How does robotic surgery approach {phase}?",
         "For {phase}, surgical robots employ **{concept}**. {fact}"),
        ("What makes robotic {phase} safer than manual surgery?",
         "Safety during {phase} is enhanced by **{concept}**. {fact}"),
        ("Describe the control system for robotic {phase}.",
         "Robotic {phase} uses **{concept}** for precise control. {fact}"),
        ("What is {phase} in surgery?",
         "{phase} is a surgical phase where **{concept}** is critical. {fact}"),
    ]
    
    for phase_name, info in phase_kb.items():
        concept = info.get('concept', 'specialized algorithms')
        fact = info.get('fact', '')
        
        for q_template, a_template in phase_templates:
            q = q_template.format(phase=phase_name)
            a = a_template.format(phase=phase_name, concept=concept, fact=fact)
            examples.append({"instruction": q, "response": a})
    
    # =========================================================================
    # 2. TOOL QUESTIONS (from JSON knowledge base)
    # =========================================================================
    tool_templates = [
        ("The vision system detects the tool '{tool}'. What is the robotic equivalent?",
         "The robotic equivalent of the {tool} uses **{concept}**. {fact}"),
        ("What technology does a robotic {tool} use?",
         "A robotic {tool} employs **{concept}**. {fact}"),
        ("How does the robotic {tool} differ from the manual version?",
         "Unlike manual tools, the robotic {tool} uses **{concept}**. {fact}"),
        ("Explain the control system for the robotic {tool}.",
         "The robotic {tool} is enhanced by **{concept}**. {fact}"),
        ("What is special about the robotic {tool}?",
         "The robotic {tool} features **{concept}**. {fact}"),
        ("What is a {tool} used for in surgery?",
         "The {tool} is a surgical instrument. In robotic surgery, it uses **{concept}**. {fact}"),
    ]
    
    for tool_name, info in tool_kb.items():
        concept = info.get('concept', 'advanced robotics')
        fact = info.get('fact', '')
        
        for q_template, a_template in tool_templates:
            q = q_template.format(tool=tool_name)
            a = a_template.format(tool=tool_name, concept=concept, fact=fact)
            examples.append({"instruction": q, "response": a})
    
    # =========================================================================
    # 3. PROJECT-SPECIFIC QUESTIONS (about YOUR system)
    # =========================================================================
    for key, info in PROJECT_KNOWLEDGE.items():
        examples.append({
            "instruction": info["question"],
            "response": info["answer"]
        })
        
        # Add variations
        if "CNN" in info["question"]:
            examples.append({
                "instruction": "Tell me about the CNN in your system.",
                "response": info["answer"]
            })
        if "LSTM" in info["question"]:
            examples.append({
                "instruction": "What does the LSTM do?",
                "response": info["answer"]
            })
    
    # Add list-based questions
    if phase_kb:
        phase_list = ", ".join(phase_kb.keys())
        examples.append({
            "instruction": "What phases are in this surgery?",
            "response": f"The surgery has {len(phase_kb)} phases: {phase_list}. Each phase uses specific robotic control algorithms for safe and precise execution."
        })
        examples.append({
            "instruction": "List all the surgical phases.",
            "response": f"The surgical phases are: {phase_list}. Our vision system (CNN + LSTM) detects these phases and explains the corresponding robotic algorithms."
        })
    
    if tool_kb:
        tool_list = ", ".join(tool_kb.keys())
        examples.append({
            "instruction": "What tools are used in this surgery?",
            "response": f"The surgery uses {len(tool_kb)} tools: {tool_list}. Each tool has a robotic equivalent with specialized control systems."
        })
        examples.append({
            "instruction": "List all the surgical tools.",
            "response": f"The surgical tools are: {tool_list}. Our CNN model detects these tools in video frames."
        })
    
    # =========================================================================
    # 4. AI LITERACY QUESTIONS
    # =========================================================================
    for key, info in AI_LITERACY_KNOWLEDGE.items():
        for question in info["questions"]:
            examples.append({
                "instruction": question,
                "response": info["answer"]
            })
    
    # =========================================================================
    # 5. CONVERSATIONAL
    # =========================================================================
    for category, pairs in CONVERSATIONAL.items():
        for q, a in pairs:
            examples.append({"instruction": q, "response": a})
    
    return examples


# =============================================================================
# CHAT FORMAT
# =============================================================================

def format_chat(instruction, response):
    """Format for TinyLlama chat"""
    return f"<|user|>\n{instruction}</s>\n<|assistant|>\n{response}</s>"


def format_prompt(instruction):
    """Format prompt for inference"""
    return f"<|user|>\n{instruction}</s>\n<|assistant|>\n"


# =============================================================================
# DATASET
# =============================================================================

class ChatDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [format_chat(ex['instruction'], ex['response']) for ex in examples]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
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

def load_model(config):
    """Load TinyLlama with LoRA"""
    
    print(f"\nLoading {config['model_name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try 4-bit quantization
    if config['use_4bit']:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config['model_name'],
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
            print("✓ Loaded with 4-bit quantization")
        except Exception as e:
            print(f"4-bit failed ({e}), using float16...")
            config['use_4bit'] = False
    
    if not config['use_4bit']:
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Apply LoRA
    if PEFT_AVAILABLE:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


# =============================================================================
# GENERATION
# =============================================================================

def generate_response(model, tokenizer, instruction, max_new_tokens=150):
    """Generate a response"""
    prompt = format_prompt(instruction)
    
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
    
    return response.replace("</s>", "").strip()


# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("=" * 70)
    print("SAR-PODCAST-BOT MODEL TRAINING")
    print("Using: TinyLlama-1.1B + Your Knowledge Base")
    print("=" * 70)
    
    # Find paths
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent if script_dir.name == 'training' else script_dir
    
    output_dir = src_dir / CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSource dir: {src_dir}")
    print(f"Output dir: {output_dir}")
    
    # Load knowledge base from JSON files
    print("\n" + "-" * 70)
    print("Loading knowledge base...")
    phase_kb, tool_kb = load_knowledge_base(src_dir)
    
    if not phase_kb and not tool_kb:
        print("ERROR: No knowledge base found!")
        print("Make sure these files exist:")
        print("  - src/dataset/Surgical_Robotics/phase_to_control_mapping.json")
        print("  - src/dataset/Surgical_Robotics/tool_to_robot_mapping.json")
        return
    
    # Load model
    model, tokenizer = load_model(CONFIG)
    
    # Generate training data
    print("\n" + "-" * 70)
    print("Generating training examples...")
    examples = generate_training_examples(phase_kb, tool_kb)
    print(f"Base examples: {len(examples)}")
    
    # Upsample
    examples = examples * CONFIG['upsample_factor']
    random.shuffle(examples)
    print(f"After {CONFIG['upsample_factor']}x upsample: {len(examples)}")
    
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
        
        # Test every 3 epochs
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print("\n  📝 Sample outputs:")
            test_qs = [
                "The vision system detects 'Preparation'. What robotic algorithm applies here?",
                "What tools are used in this surgery?",
                "What CNN model do you use?",
                "How does a computer learn?",
                "Hello!",
            ]
            for q in test_qs:
                resp = generate_response(model, tokenizer, q)
                print(f"    Q: {q[:45]}...")
                print(f"    A: {resp[:80]}...")
            print()
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-o', label='Train')
    plt.plot(val_losses, 'r-s', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SAR-Podcast-Bot Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(output_dir / 'training_curves.png'))
    print(f"\nPlot saved to: {output_dir / 'training_curves.png'}")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    test_prompts = [
        # Surgical phases
        "The vision system detects 'Preparation'. What robotic algorithm applies here?",
        "What is the Clipping/Cutting phase?",
        "List all the surgical phases.",
        
        # Tools
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        "What tools are used in this surgery?",
        
        # Project-specific
        "What CNN model do you use?",
        "What is the LSTM used for?",
        "How does the vision system work?",
        
        # AI literacy
        "How does a computer learn?",
        "What is deep learning?",
        "Will AI replace surgeons?",
        
        # Conversational
        "Hello!",
        "What can you help with?",
    ]
    
    for q in test_prompts:
        response = generate_response(model, tokenizer, q)
        print(f"\nQ: {q}")
        print(f"A: {response}")
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print("✓ Training complete!")
    print(f"  Best model: {output_dir / 'best_model'}")
    print(f"  Best val loss: {best_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    train()