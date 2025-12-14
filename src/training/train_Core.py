"""
GPT-2 Fine-Tuning for Surgical Robotics Knowledge
==================================================
Clean implementation following best practices:

1. NO DailyDialog - domain knowledge only
2. Alpaca-style instruction format
3. Generate lots of variations from knowledge base
4. Simple, proven training approach

This is what your friend likely did with Wikipedia.

Format:
### Instruction:
{question}

### Response:
{answer}
"""

import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# INSTRUCTION FORMAT (Alpaca-style - proven to work)
# =============================================================================

INSTRUCTION_TEMPLATE = """### Instruction:
{instruction}

### Response:
{response}"""


# =============================================================================
# KNOWLEDGE BASE - All surgical robotics knowledge
# =============================================================================

PHASE_KNOWLEDGE = {
    "Preparation": {
        "concept": "Visual SLAM",
        "explanation": "During preparation, a robot uses Visual SLAM (Simultaneous Localization and Mapping). It tracks feature points on the cavity walls to build a 3D depth map of the environment."
    },
    "CalotTriangleDissection": {
        "concept": "Motion Scaling",
        "explanation": "Dissection requires precision. A robot uses Motion Scaling (e.g., 5:1 ratio), converting a 5cm hand movement into a 1cm tool movement to prevent accidental cuts."
    },
    "ClippingCutting": {
        "concept": "Tremor Filtration",
        "explanation": "Clipping the artery requires steadiness. The robot uses a 6Hz Low-Pass Filter to remove the surgeon's natural hand tremors, ensuring the clip is placed perfectly."
    },
    "GallbladderRetraction": {
        "concept": "Active Constraints",
        "explanation": "To hold the organ safely, the robot uses Active Constraints (Virtual Fixtures). These are software 'invisible walls' that prevent the tool from slipping into the liver."
    },
    "GallbladderDissection": {
        "concept": "Inverse Kinematics",
        "explanation": "The robot uses Inverse Kinematics to calculate the exact joint angles needed to maneuver the tool behind the gallbladder without the robot arms colliding."
    },
    "CleaningCoagulation": {
        "concept": "Augmented Reality",
        "explanation": "To find bleeding spots, robotic systems overlay Augmented Reality feeds (like Firefly fluorescence) to highlight blood flow in green on the surgeon's screen."
    },
    "GallbladderPackaging": {
        "concept": "Master-Slave Teleoperation",
        "explanation": "Bagging the specimen relies on Master-Slave Teleoperation algorithms. The system compensates for processing latency to ensure the robot moves instantly with the surgeon's hands."
    }
}

TOOL_KNOWLEDGE = {
    "Hook": {
        "concept": "EndoWrist (7 DOFs)",
        "explanation": "The manual Hook is rigid. A robotic hook uses EndoWrist technology with 7 Degrees of Freedom, allowing it to rotate 360 degrees to approach tissue from any angle."
    },
    "Grasper": {
        "concept": "Haptic Feedback Simulation",
        "explanation": "Robots lack touch. To compensate, algorithms use Visual Haptics - analyzing tissue deformation in the video to estimate the force being applied."
    },
    "Clipper": {
        "concept": "Articulated Clip Applier",
        "explanation": "A robotic Clipper can articulate (bend) at the wrist, allowing the surgeon to place clips from the optimal angle without awkward hand positions."
    },
    "Scissors": {
        "concept": "Tremor Filtration",
        "explanation": "Manual cutting can be shaky. The robotic scissors use a 6Hz Low-Pass Filter to remove physiological hand tremors, allowing smooth, confident cuts."
    },
    "Bipolar": {
        "concept": "Multitasking Efficiency",
        "explanation": "The robotic Maryland Bipolar can dissect, grasp, and coagulate simultaneously due to its wrist articulation, reducing instrument changes."
    },
    "Irrigator": {
        "concept": "Foot-Pedal Control",
        "explanation": "Robotic suction and irrigation is controlled via foot pedals at the console, freeing both hands for operating the primary instruments."
    }
}

AI_KNOWLEDGE = {
    "computer_learning": {
        "explanation": "A computer learns by analyzing many examples and adjusting internal parameters called 'weights'. Like a child learning to recognize cats from pictures, neural networks gradually improve by seeing millions of examples."
    },
    "neural_network": {
        "explanation": "A neural network is like a chain of simple decision-makers. Each layer takes input, transforms it slightly, and passes it forward. Together, millions of these operations can recognize complex patterns."
    },
    "deep_learning": {
        "explanation": "Deep learning uses neural networks with many layers. Each layer learns increasingly abstract features - from edges to shapes to objects. This hierarchical learning enables complex tasks like image recognition."
    },
    "ai_surgery": {
        "explanation": "AI in surgery assists rather than replaces surgeons. It can detect phases and tools, provide warnings, and enhance visualization, but critical decisions remain with trained human surgeons."
    },
    "ai_safety": {
        "explanation": "AI safety in surgery includes hardware limits on force and speed, software constraints that prevent dangerous movements, and always keeping a human surgeon in control of critical decisions."
    }
}


# =============================================================================
# DATASET GENERATION - Create many training examples
# =============================================================================

def generate_training_data():
    """Generate diverse training examples from knowledge base"""
    examples = []
    
    # Phase questions - many variations
    phase_templates = [
        ("The vision system detects '{phase}'. What robotic algorithm applies?",
         "For {phase}, the key concept is {concept}. {explanation}"),
        ("What robotic technology is used during {phase}?",
         "During {phase}, robots use {concept}. {explanation}"),
        ("Explain how a robot performs {phase}.",
         "{concept} is essential for {phase}. {explanation}"),
        ("What algorithm helps robots during the {phase} phase?",
         "The robot relies on {concept} during {phase}. {explanation}"),
        ("How does robotic surgery handle {phase}?",
         "For {phase}, the critical technology is {concept}. {explanation}"),
        ("What makes robotic {phase} safer than manual surgery?",
         "Safety during {phase} is improved by {concept}. {explanation}"),
        ("Describe the control system for robotic {phase}.",
         "{phase} uses {concept} for precise control. {explanation}"),
    ]
    
    for phase, info in PHASE_KNOWLEDGE.items():
        display_name = phase.replace("CalotTriangle", "Calot Triangle ").replace("Gallbladder", "Gallbladder ")
        for q_template, a_template in phase_templates:
            q = q_template.format(phase=display_name)
            a = a_template.format(phase=display_name, concept=info['concept'], explanation=info['explanation'])
            examples.append({"instruction": q, "response": a})
    
    # Tool questions
    tool_templates = [
        ("The vision system detects the tool '{tool}'. What is the robotic equivalent?",
         "The robotic version of the {tool} uses {concept}. {explanation}"),
        ("What technology enables robotic {tool} functionality?",
         "Robotic {tool} relies on {concept}. {explanation}"),
        ("How does a robotic {tool} differ from a manual one?",
         "Unlike the manual {tool}, the robotic version uses {concept}. {explanation}"),
        ("Explain the robotic control concept for the {tool}.",
         "The {tool} in robotic surgery uses {concept}. {explanation}"),
        ("What makes the robotic {tool} more precise?",
         "Precision is achieved through {concept}. {explanation}"),
    ]
    
    for tool, info in TOOL_KNOWLEDGE.items():
        for q_template, a_template in tool_templates:
            q = q_template.format(tool=tool)
            a = a_template.format(tool=tool, concept=info['concept'], explanation=info['explanation'])
            examples.append({"instruction": q, "response": a})
    
    # AI literacy questions
    ai_questions = [
        ("How does a computer learn?", AI_KNOWLEDGE["computer_learning"]["explanation"]),
        ("How do machines learn from data?", AI_KNOWLEDGE["computer_learning"]["explanation"]),
        ("Explain machine learning simply.", AI_KNOWLEDGE["computer_learning"]["explanation"]),
        ("What is a neural network?", AI_KNOWLEDGE["neural_network"]["explanation"]),
        ("How do neural networks work?", AI_KNOWLEDGE["neural_network"]["explanation"]),
        ("Explain neural networks to a beginner.", AI_KNOWLEDGE["neural_network"]["explanation"]),
        ("What is deep learning?", AI_KNOWLEDGE["deep_learning"]["explanation"]),
        ("How is deep learning different from machine learning?", AI_KNOWLEDGE["deep_learning"]["explanation"]),
        ("Will AI replace surgeons?", AI_KNOWLEDGE["ai_surgery"]["explanation"]),
        ("Can AI perform surgery autonomously?", AI_KNOWLEDGE["ai_surgery"]["explanation"]),
        ("How do we ensure AI surgery is safe?", AI_KNOWLEDGE["ai_safety"]["explanation"]),
        ("What safety measures exist in surgical AI?", AI_KNOWLEDGE["ai_safety"]["explanation"]),
    ]
    
    for q, a in ai_questions:
        examples.append({"instruction": q, "response": a})
    
    # Surgery overview questions
    overview_questions = [
        ("What are the stages of a cholecystectomy?",
         "A laparoscopic cholecystectomy has 7 main phases: Preparation, Calot Triangle Dissection, Clipping/Cutting, Gallbladder Dissection, Gallbladder Retraction, Cleaning/Coagulation, and Gallbladder Packaging."),
        ("What phases are in gallbladder removal surgery?",
         "Gallbladder removal proceeds through: Preparation (Visual SLAM), Dissection (Motion Scaling), Clipping/Cutting (Tremor Filtration), Retraction (Active Constraints), and Packaging (Teleoperation)."),
        ("List the surgical phases and their robotic algorithms.",
         "Each phase has a key algorithm: Preparation uses Visual SLAM, Dissection uses Motion Scaling, Clipping uses Tremor Filtration, Retraction uses Active Constraints, and Packaging uses Master-Slave Teleoperation."),
    ]
    
    for q, a in overview_questions:
        examples.append({"instruction": q, "response": a})
    
    # Conversational (basic - to not confuse the model)
    greetings = [
        ("Hello!", "Hello! I'm the SAR-Podcast-Bot. I can explain surgical robotics concepts and AI technology. What would you like to know?"),
        ("Hi there.", "Hi! I specialize in surgical robotics and AI. Ask me about robotic surgery phases, tools, or how AI works."),
        ("How are you?", "I'm ready to help! I can explain surgical robotics algorithms, tool technologies, or answer questions about AI and machine learning."),
        ("What can you help with?", "I can explain: surgical phases and their robotic algorithms, tool technologies in robotic surgery, and AI/machine learning concepts."),
    ]
    
    for q, a in greetings:
        examples.append({"instruction": q, "response": a})
    
    return examples


def format_example(instruction, response):
    """Format as Alpaca-style instruction"""
    return INSTRUCTION_TEMPLATE.format(instruction=instruction, response=response)


class InstructionDataset(Dataset):
    """Dataset for instruction-tuning"""
    
    def __init__(self, examples, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        
        for ex in examples:
            text = format_example(ex['instruction'], ex['response'])
            self.texts.append(text)
        
        # Add EOS token to each example
        self.texts = [t + tokenizer.eos_token for t in self.texts]
    
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
        
        # Labels are input_ids, but mask padding
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        # Also mask the instruction part (only train on response)
        # Find "### Response:" and mask everything before it
        text_ids = input_ids.tolist()
        response_marker = self.tokenizer.encode("### Response:", add_special_tokens=False)
        
        # Find where response starts
        for i in range(len(text_ids) - len(response_marker)):
            if text_ids[i:i+len(response_marker)] == response_marker:
                # Mask everything up to and including "### Response:\n"
                labels[:i + len(response_marker) + 1] = -100
                break
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("=" * 70)
    print("GPT-2 INSTRUCTION FINE-TUNING")
    print("Domain: Surgical Robotics + AI Literacy")
    print("=" * 70)
    
    # Setup
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent if script_dir.name == 'training' else script_dir
    
    output_dir = src_dir / 'results' / 'gpt2_instruction'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Apply LoRA
    if PEFT_AVAILABLE:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=['c_attn', 'c_proj'],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    model = model.to(device)
    
    # Generate training data
    print("\nGenerating training data...")
    examples = generate_training_data()
    print(f"Generated {len(examples)} training examples")
    
    # Upsample to ensure model sees enough examples
    # Repeat dataset 10x for ~1000+ examples
    examples = examples * 10
    random.shuffle(examples)
    print(f"After upsampling: {len(examples)} examples")
    
    # Split
    split = int(len(examples) * 0.9)
    train_examples = examples[:split]
    val_examples = examples[split:]
    
    train_dataset = InstructionDataset(train_examples, tokenizer, max_length=256)
    val_dataset = InstructionDataset(val_examples, tokenizer, max_length=256)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training config
    EPOCHS = 15
    LR = 2e-4
    ACCUM_STEPS = 4
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()
        
        for i, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / ACCUM_STEPS
            loss.backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += outputs.loss.item()
            pbar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
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
        
        # Test generation every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\n  Sample outputs:")
            test_prompts = [
                "The vision system detects 'Preparation'. What robotic algorithm applies?",
                "How does a computer learn?",
                "Hello!",
            ]
            for p in test_prompts:
                response = generate_response(model, tokenizer, device, p)
                print(f"    Q: {p[:40]}...")
                print(f"    A: {response[:60]}...")
            print()
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-o', label='Train')
    plt.plot(val_losses, 'r-s', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Instruction Fine-tuning Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(output_dir / 'training_curves.png'))
    
    # Final test
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    test_prompts = [
        "The vision system detects 'Preparation'. What robotic algorithm applies?",
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        "What are the stages of a cholecystectomy?",
        "How does a computer learn?",
        "What is deep learning?",
        "Will AI replace surgeons?",
        "Hello!",
    ]
    
    for p in test_prompts:
        response = generate_response(model, tokenizer, device, p)
        print(f"\nQ: {p}")
        print(f"A: {response}")
    
    print(f"\n✓ Training complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model'}")


def generate_response(model, tokenizer, device, instruction, max_new_tokens=100):
    """Generate response using instruction format"""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=200)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response after "### Response:"
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[-1].strip()
    else:
        response = full_text[len(prompt):].strip()
    
    # Stop at next instruction marker if present
    if "### Instruction:" in response:
        response = response.split("### Instruction:")[0].strip()
    
    return response


if __name__ == "__main__":
    train()