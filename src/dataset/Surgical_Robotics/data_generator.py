"""
Improved Surgical Robotics + AI Literacy Data Generator
========================================================
Generates a larger, more diverse training dataset with:
1. More prompt variations for each concept
2. AI literacy and ethics Q&A pairs
3. Conversational variations
4. Better coverage of edge cases

This should produce 500+ training examples instead of ~70.
"""

import json
import random
from pathlib import Path

# =============================================================================
# PHASE KNOWLEDGE BASE (Enhanced)
# =============================================================================

PHASE_KNOWLEDGE = {
    "Preparation": {
        "concept": "Visual SLAM",
        "fact": "During preparation, a robot uses Visual SLAM (Simultaneous Localization and Mapping). It tracks feature points on the cavity walls to build a 3D depth map of the environment.",
        "safety": "Visual SLAM improves safety by giving the robot spatial awareness, preventing collisions with cavity walls.",
        "comparison": "While a human surgeon relies on visual estimation, the robot builds a precise 3D mathematical model.",
        "future": "Future systems may use AI-enhanced SLAM that predicts tissue movement in real-time."
    },
    "CalotTriangleDissection": {
        "concept": "Motion Scaling",
        "fact": "Dissection requires precision. A robot uses Motion Scaling (e.g., 5:1 ratio), converting a 5cm hand movement into a 1cm tool movement to prevent accidental cuts.",
        "safety": "Motion scaling prevents accidental tissue damage by filtering out large, jerky movements.",
        "comparison": "Human surgeons must consciously control micro-movements, while robots do this automatically.",
        "future": "AI could dynamically adjust scaling ratios based on tissue type detected by computer vision."
    },
    "ClippingCutting": {
        "concept": "Tremor Filtration",
        "fact": "Clipping the artery requires steadiness. The robot uses a 6Hz Low-Pass Filter to remove the surgeon's natural hand tremors, ensuring the clip is placed perfectly.",
        "safety": "Tremor filtration is critical during clipping - even small tremors could cause the clip to slip.",
        "comparison": "Even the steadiest human surgeon has physiological tremors of 8-12Hz that robots can filter out.",
        "future": "Next-generation systems may predict tremor patterns and compensate proactively."
    },
    "GallbladderRetraction": {
        "concept": "Active Constraints",
        "fact": "To hold the organ safely, the robot uses Active Constraints (Virtual Fixtures). These are software 'invisible walls' that prevent the tool from slipping into the liver while maintaining tension.",
        "safety": "Virtual fixtures prevent the tool from entering dangerous 'no-go zones' near major blood vessels.",
        "comparison": "A human must consciously avoid dangerous areas, while the robot is physically prevented from entering them.",
        "future": "AI could generate dynamic virtual fixtures based on real-time anatomy detection."
    },
    "GallbladderDissection": {
        "concept": "Inverse Kinematics",
        "fact": "The robot uses Inverse Kinematics to calculate the exact joint angles needed to maneuver the tool behind the gallbladder without the robot arms colliding.",
        "safety": "Inverse kinematics prevents arm collisions that could cause uncontrolled tool movements.",
        "comparison": "Humans naturally coordinate their arms, but robots need mathematical models for the same coordination.",
        "future": "Machine learning could optimize kinematic solutions for individual patient anatomies."
    },
    "CleaningCoagulation": {
        "concept": "Augmented Reality",
        "fact": "To find bleeding spots, robotic systems overlay Augmented Reality feeds (like Firefly fluorescence) to highlight blood flow in green on the surgeon's screen.",
        "safety": "AR visualization helps identify hidden bleeders that might be missed by the naked eye.",
        "comparison": "Human surgeons see only visible light, while robots can show near-infrared fluorescence.",
        "future": "AI could automatically identify and mark bleeding spots for the surgeon."
    },
    "GallbladderPackaging": {
        "concept": "Master-Slave Teleoperation",
        "fact": "Bagging the specimen relies on Master-Slave Teleoperation algorithms. The system compensates for processing latency to ensure the robot moves instantly with the surgeon's hands.",
        "safety": "Latency compensation prevents delayed robot responses that could cause tissue damage.",
        "comparison": "Direct manual surgery has zero latency, but teleoperation can add 50-200ms delay that must be compensated.",
        "future": "5G networks may enable remote surgery across continents with minimal latency."
    }
}

# =============================================================================
# TOOL KNOWLEDGE BASE (Enhanced)
# =============================================================================

TOOL_KNOWLEDGE = {
    "Hook": {
        "concept": "EndoWrist (7 DOFs)",
        "fact": "The manual Hook is rigid. A robotic hook uses EndoWrist technology with 7 Degrees of Freedom, allowing it to rotate 360 degrees to hook tissue from behind.",
        "safety": "7 DOFs allow the hook to approach tissue from safer angles, reducing accidental damage.",
        "comparison": "A manual hook has only 4 degrees of freedom, limiting approach angles."
    },
    "Grasper": {
        "concept": "Haptic Feedback Simulation",
        "fact": "Robots lack touch. To compensate, algorithms use Visual Haptics—analyzing tissue deformation in the video to estimate force.",
        "safety": "Visual haptics prevents excessive gripping force that could crush delicate tissue.",
        "comparison": "Human surgeons feel tissue resistance directly; robots must estimate it computationally."
    },
    "Clipper": {
        "concept": "Articulated Clip Applier",
        "fact": "A robotic Clipper can articulate (bend) at the wrist, allowing the surgeon to place clips on the artery from the side without contorting their own arm.",
        "safety": "Wrist articulation allows optimal clip placement angle, reducing the risk of incomplete closure.",
        "comparison": "Manual clip appliers are straight, requiring awkward hand positions for angled clips."
    },
    "Scissors": {
        "concept": "Tremor Filtration",
        "fact": "Manual snipping can be shaky. The robotic scissors use a digital filter to remove the surgeon's physiological hand tremors, allowing for smooth, confident cuts.",
        "safety": "Filtered cutting motions prevent jagged incisions that heal poorly.",
        "comparison": "Even expert surgeons have 8-12Hz tremors; robots filter these completely."
    },
    "Bipolar": {
        "concept": "Multitasking Efficiency",
        "fact": "The robotic Maryland Bipolar can dissect, grasp, and coagulate simultaneously due to its wrist articulation. This reduces instrument swaps.",
        "safety": "Fewer instrument changes mean fewer opportunities for accidental tissue contact.",
        "comparison": "Manual bipolar tools are rigid and single-function, requiring multiple instruments."
    },
    "Irrigator": {
        "concept": "Console Foot-Pedal Control",
        "fact": "While manual irrigation requires a hand, robotic suction/irrigation is controlled via foot pedals, freeing both hands for operating instruments.",
        "safety": "Hands-free irrigation allows the surgeon to maintain control of critical instruments.",
        "comparison": "Manual irrigation requires an assistant or sacrificing a hand."
    },
    "SpecimenBag": {
        "concept": "Assistant Port Coordination",
        "fact": "Robots struggle with soft, floppy objects. The Specimen Bag is introduced through an Assistant Port by a human nurse, requiring verbal coordination.",
        "safety": "Human-robot coordination protocols prevent bag tears and specimen spillage.",
        "comparison": "This is one area where human dexterity still exceeds robotic capability."
    }
}

# =============================================================================
# AI LITERACY KNOWLEDGE BASE (NEW!)
# =============================================================================

AI_LITERACY = {
    "how_computers_learn": {
        "simple": "A computer learns by looking at millions of examples and adjusting its internal settings to recognize patterns. It's like how you learned to recognize cats by seeing many pictures of cats.",
        "technical": "Machine learning algorithms optimize mathematical functions by adjusting parameters (weights) to minimize prediction errors on training data. This process is called gradient descent.",
        "analogy": "Imagine adjusting the knobs on a radio to find the clearest signal. Neural networks have millions of 'knobs' that get adjusted automatically during training."
    },
    "neural_networks": {
        "simple": "A neural network is like a chain of simple decision-makers. Each one takes input, makes a small decision, and passes the result forward. Together, they can make complex decisions.",
        "technical": "Neural networks are composed of layers of artificial neurons that apply weighted sums and nonlinear activation functions to transform input data into useful representations.",
        "analogy": "Think of it like a game of telephone, but each person adds their own interpretation. By the end, the message has been transformed into something useful."
    },
    "why_gpus": {
        "simple": "AI needs GPUs because it does the same calculation millions of times in parallel. GPUs are designed for exactly this kind of work.",
        "technical": "Neural network training involves matrix multiplications that can be parallelized across thousands of GPU cores, achieving 10-100x speedup over CPUs.",
        "analogy": "Imagine counting a million beans. One person (CPU) counts them one by one. A GPU is like having 1000 people counting simultaneously."
    },
    "deep_learning": {
        "simple": "Deep learning uses neural networks with many layers. More layers mean the network can learn more complex patterns.",
        "technical": "Deep learning refers to neural networks with multiple hidden layers that can learn hierarchical representations of data.",
        "analogy": "It's like learning to read: first you learn letters, then words, then sentences, then meaning. Each 'layer' builds on the previous one."
    },
    "training_vs_inference": {
        "simple": "Training is when the AI learns from examples. Inference is when it uses what it learned to make predictions on new data.",
        "technical": "Training optimizes model parameters using backpropagation. Inference applies the fixed model to new inputs without updating weights.",
        "analogy": "Training is like studying for an exam. Inference is taking the exam."
    },
    "overfitting": {
        "simple": "Overfitting is when an AI memorizes the training examples instead of learning general patterns. It does great on practice tests but fails on real tests.",
        "technical": "Overfitting occurs when a model learns noise in the training data rather than the underlying distribution, leading to poor generalization.",
        "analogy": "It's like memorizing answers to specific math problems instead of learning how to solve math problems in general."
    }
}

# =============================================================================
# ETHICS KNOWLEDGE BASE (NEW!)
# =============================================================================

ETHICS_KB = {
    "ai_takeover": {
        "response": "Current AI is 'narrow AI' - it can only do specific tasks it was trained for. Our surgical vision system can recognize tools and phases, but it can't have a conversation or make coffee. True 'general AI' that matches human intelligence doesn't exist yet and may not for decades. The real concerns are about how we use AI responsibly today.",
        "balanced": "While AI won't 'take over' in a sci-fi sense, it will transform many jobs. The question isn't whether AI will replace humans, but how we can ensure the benefits are shared fairly."
    },
    "ai_benefits": {
        "response": "AI has enormous potential to benefit everyone: earlier disease diagnosis, personalized education, safer transportation, and more. However, we must actively work to prevent algorithmic bias, job displacement without support, and concentration of AI benefits among the wealthy.",
        "examples": "In surgery, AI can help detect cancers earlier, guide less experienced surgeons, and reduce human error. But we need to ensure these benefits reach hospitals in developing countries, not just wealthy ones."
    },
    "ai_safety": {
        "response": "AI safety involves multiple layers: technical safeguards built into systems, ethical guidelines and regulations, transparency about how AI makes decisions, and education so people can understand and oversee AI systems.",
        "surgical": "In surgical robotics, safety includes mechanical stops that prevent dangerous movements, software limits on force and speed, and always keeping a human surgeon in control."
    },
    "ai_bias": {
        "response": "AI systems can inherit biases from their training data. If a medical AI is trained mostly on data from one demographic group, it may work less well for others. We combat this by using diverse training data, testing for bias, and involving diverse teams in AI development.",
        "example": "Early skin cancer detection AI worked poorly on dark skin because training data was mostly from light-skinned patients. This shows why diverse, representative data is crucial."
    },
    "human_oversight": {
        "response": "AI should augment human capabilities, not replace human judgment in critical decisions. In surgery, the AI provides information and assistance, but a trained human surgeon makes all important decisions and can override the system at any time.",
        "principle": "The principle of 'human in the loop' means AI systems should support and enhance human decision-making rather than making autonomous decisions in high-stakes situations."
    }
}

# =============================================================================
# PROMPT TEMPLATES (Diverse variations)
# =============================================================================

PHASE_PROMPTS = [
    # Direct identification
    "The vision system detects '{phase}'. What robotic algorithm applies here?",
    "What robotic control concept is used during {phase}?",
    "Explain the technology behind robotic {phase}.",
    "During {phase}, what algorithm does the surgical robot use?",
    
    # Comparison
    "How does robotic {phase} differ from manual surgery?",
    "Compare manual and robotic approaches to {phase}.",
    "What advantages does a robot have during {phase}?",
    
    # Safety focus
    "Why is the robotic approach to {phase} considered safer?",
    "How does the robot improve safety during {phase}?",
    "What safety features apply during robotic {phase}?",
    
    # Educational
    "Explain the control theory behind robotic {phase}.",
    "What engineering principles enable robotic {phase}?",
    "How would you explain robotic {phase} to a medical student?",
    
    # Future-oriented
    "How might AI improve {phase} in the future?",
    "What's the next evolution of robotic {phase}?",
]

TOOL_PROMPTS = [
    # Direct identification
    "The vision system detects the tool '{tool}'. What is the robotic equivalent?",
    "What robotic technology corresponds to the manual {tool}?",
    "Explain the robotic version of the {tool}.",
    
    # Technical
    "Explain the robotic control concept for the {tool}.",
    "What engineering enables the robotic {tool}?",
    "How does the robotic {tool} work?",
    
    # Comparison
    "Compare the manual {tool} with its robotic counterpart.",
    "What can a robotic {tool} do that a manual one cannot?",
    "Contrast the manual {tool} with a robotic system.",
    
    # Safety
    "How does the robotic {tool} improve surgical safety?",
    "What safety advantages does the robotic {tool} have?",
    
    # Autonomy
    "Could an AI operate the {tool} autonomously in the future?",
    "What would it take for AI to fully automate the {tool}?",
]

AI_LITERACY_PROMPTS = {
    "how_computers_learn": [
        "How does a computer learn?",
        "How do machines learn from data?",
        "Explain machine learning in simple terms.",
        "How does AI learn to recognize patterns?",
        "What does it mean when we say an AI 'learns'?",
    ],
    "neural_networks": [
        "What is a neural network?",
        "How can I explain neural networks to my grandmother?",
        "Explain neural networks simply.",
        "What are artificial neural networks?",
        "How do neural networks process information?",
    ],
    "why_gpus": [
        "Why does AI need so much computing power?",
        "Why do we need GPUs for AI?",
        "Why are AI systems so computationally expensive?",
        "What makes AI training so resource-intensive?",
    ],
    "deep_learning": [
        "What is deep learning?",
        "How is deep learning different from machine learning?",
        "What makes deep learning 'deep'?",
        "Explain deep learning.",
    ],
    "training_vs_inference": [
        "What's the difference between training and inference?",
        "How is AI training different from using AI?",
        "Explain training versus inference in AI.",
    ],
    "overfitting": [
        "What is overfitting in machine learning?",
        "Why do AI models sometimes fail on new data?",
        "Explain overfitting.",
        "How do we prevent AI from memorizing instead of learning?",
    ]
}

ETHICS_PROMPTS = {
    "ai_takeover": [
        "Will AI take over?",
        "Should we be afraid of AI?",
        "Is AI dangerous?",
        "Will robots replace humans?",
        "Could AI become smarter than humans?",
    ],
    "ai_benefits": [
        "Can everyone benefit from AI?",
        "Who benefits from AI technology?",
        "How can AI help society?",
        "What are the benefits of AI?",
    ],
    "ai_safety": [
        "How do we make AI safe?",
        "How to prevent AI from being wrongly used?",
        "What safety measures exist for AI?",
        "How do we ensure AI is used responsibly?",
    ],
    "ai_bias": [
        "Can AI be biased?",
        "How do we prevent AI bias?",
        "Is AI fair to everyone?",
        "What is algorithmic bias?",
    ],
    "human_oversight": [
        "Should humans always control AI?",
        "What role should humans play in AI decisions?",
        "How do we keep humans in the loop with AI?",
    ]
}


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_phase_examples():
    """Generate diverse phase-related training examples"""
    examples = []
    
    for phase, info in PHASE_KNOWLEDGE.items():
        # Format phase name for display
        display_phase = phase.replace("CalotTriangle", "Calot Triangle ").replace("Gallbladder", "Gallbladder ")
        
        for prompt_template in PHASE_PROMPTS:
            prompt = prompt_template.format(phase=display_phase)
            
            # Choose appropriate response based on prompt type
            if "safer" in prompt.lower() or "safety" in prompt.lower():
                response = f"Safety during {display_phase} is improved by **{info['concept']}**. {info['safety']}"
            elif "compare" in prompt.lower() or "differ" in prompt.lower() or "advantage" in prompt.lower():
                response = f"{info['comparison']} The key technology is **{info['concept']}**."
            elif "future" in prompt.lower() or "evolution" in prompt.lower() or "ai improve" in prompt.lower():
                response = f"Currently, {display_phase} uses **{info['concept']}**. {info['future']}"
            else:
                response = f"For {display_phase}, the critical concept is **{info['concept']}**. {info['fact']}"
            
            examples.append({"instruction": prompt, "response": response})
    
    return examples


def generate_tool_examples():
    """Generate diverse tool-related training examples"""
    examples = []
    
    for tool, info in TOOL_KNOWLEDGE.items():
        for prompt_template in TOOL_PROMPTS:
            prompt = prompt_template.format(tool=tool)
            
            if "safety" in prompt.lower():
                response = f"Safety is enhanced because **{info['concept']}** allows for greater precision. {info['safety']}"
            elif "compare" in prompt.lower() or "contrast" in prompt.lower() or "cannot" in prompt.lower():
                response = f"The manual {tool} is limited. A robot improves this using **{info['concept']}**. {info['comparison']}"
            elif "autonom" in prompt.lower() or "ai operate" in prompt.lower():
                response = f"It is challenging but possible. To automate the {tool}, an AI would need to perfectly master **{info['concept']}**. Currently, AI is mostly used to assist, not replace, this action."
            else:
                response = f"For the {tool}, the critical concept is **{info['concept']}**. {info['fact']}"
            
            examples.append({"instruction": prompt, "response": response})
    
    return examples


def generate_ai_literacy_examples():
    """Generate AI literacy training examples"""
    examples = []
    
    for topic, prompts in AI_LITERACY_PROMPTS.items():
        info = AI_LITERACY[topic]
        
        for prompt in prompts:
            # Vary response style
            if "grandmother" in prompt.lower() or "simple" in prompt.lower():
                response = info["analogy"]
            elif "technical" in prompt.lower() or "engineering" in prompt.lower():
                response = info["technical"]
            else:
                response = info["simple"]
            
            examples.append({"instruction": prompt, "response": response})
    
    return examples


def generate_ethics_examples():
    """Generate ethics training examples"""
    examples = []
    
    for topic, prompts in ETHICS_PROMPTS.items():
        info = ETHICS_KB[topic]
        
        for prompt in prompts:
            response = info["response"]
            examples.append({"instruction": prompt, "response": response})
    
    return examples


def generate_conversational_examples():
    """Generate basic conversational examples to prevent surgical responses to greetings"""
    examples = [
        {"instruction": "Hello!", "response": "Hello! I'm the SAR-Podcast-Bot, here to help you learn about surgical robotics and AI. What would you like to know?"},
        {"instruction": "Hi there", "response": "Hi! I'm ready to discuss surgical robotics, AI concepts, or answer questions about our vision system. How can I help?"},
        {"instruction": "How are you?", "response": "I'm functioning well, thank you! I'm an AI assistant focused on surgical robotics education. What would you like to explore today?"},
        {"instruction": "What can you do?", "response": "I can explain surgical robotic concepts, discuss AI and machine learning topics, and answer questions about our vision system that detects surgical phases and tools."},
        {"instruction": "Who are you?", "response": "I'm SAR-Podcast-Bot, an AI assistant trained to explain surgical robotics and AI concepts. I was created as part of a deep learning coursework project."},
        {"instruction": "Thank you", "response": "You're welcome! Feel free to ask more questions about surgical robotics or AI anytime."},
        {"instruction": "Goodbye", "response": "Goodbye! Thank you for learning about surgical robotics and AI with me today."},
        {"instruction": "What is this project about?", "response": "This project uses computer vision to analyze surgical videos, detecting tools and phases, then explains the robotic control concepts that would apply. It combines deep learning vision models with language generation."},
    ]
    return examples


def generate_surgery_overview_examples():
    """Generate examples about the overall surgery to answer 'what stages' type questions"""
    examples = [
        {
            "instruction": "What stages are in this surgery?",
            "response": "A laparoscopic cholecystectomy (gallbladder removal) has 7 main phases: 1) Preparation - setting up instruments and camera, 2) Calot Triangle Dissection - exposing the cystic duct and artery, 3) Clipping and Cutting - securing and dividing these structures, 4) Gallbladder Dissection - separating the gallbladder from the liver, 5) Gallbladder Retraction - holding the gallbladder for removal, 6) Cleaning and Coagulation - ensuring hemostasis, and 7) Gallbladder Packaging - bagging the specimen for extraction."
        },
        {
            "instruction": "What are the phases of cholecystectomy?",
            "response": "Cholecystectomy consists of 7 phases: Preparation, Calot Triangle Dissection, Clipping/Cutting, Gallbladder Dissection, Gallbladder Retraction, Cleaning/Coagulation, and Gallbladder Packaging. Each phase requires different robotic control concepts."
        },
        {
            "instruction": "How would I approach this surgical procedure?",
            "response": "A cholecystectomy proceeds through these steps: First, prepare the surgical field and orient the camera (using Visual SLAM). Then dissect the Calot triangle to expose key structures (Motion Scaling). Next, clip and cut the cystic duct and artery (Tremor Filtration). Dissect the gallbladder from its liver bed (Inverse Kinematics). Maintain retraction throughout (Active Constraints). Clean and coagulate any bleeding (AR visualization). Finally, bag the specimen for removal (Master-Slave Teleoperation)."
        },
        {
            "instruction": "What robotics algorithm applies to each stage?",
            "response": "Each surgical phase uses specific control algorithms: Preparation uses Visual SLAM for spatial mapping. Dissection uses Motion Scaling for precision. Clipping/Cutting uses Tremor Filtration for steadiness. Retraction uses Active Constraints (Virtual Fixtures) for safety. Gallbladder Dissection uses Inverse Kinematics for coordination. Cleaning uses Augmented Reality for visualization. Packaging uses Master-Slave Teleoperation for coordination."
        },
    ]
    return examples


def main():
    """Generate the complete training dataset"""
    print("=" * 60)
    print("IMPROVED DATA GENERATOR")
    print("=" * 60)
    
    all_examples = []
    
    # Generate each category
    phase_examples = generate_phase_examples()
    print(f"Generated {len(phase_examples)} phase examples")
    all_examples.extend(phase_examples)
    
    tool_examples = generate_tool_examples()
    print(f"Generated {len(tool_examples)} tool examples")
    all_examples.extend(tool_examples)
    
    ai_examples = generate_ai_literacy_examples()
    print(f"Generated {len(ai_examples)} AI literacy examples")
    all_examples.extend(ai_examples)
    
    ethics_examples = generate_ethics_examples()
    print(f"Generated {len(ethics_examples)} ethics examples")
    all_examples.extend(ethics_examples)
    
    conv_examples = generate_conversational_examples()
    print(f"Generated {len(conv_examples)} conversational examples")
    all_examples.extend(conv_examples)
    
    overview_examples = generate_surgery_overview_examples()
    print(f"Generated {len(overview_examples)} surgery overview examples")
    all_examples.extend(overview_examples)
    
    # Shuffle
    random.shuffle(all_examples)
    
    print(f"\nTOTAL: {len(all_examples)} training examples")
    
    # Save
    output_path = Path(__file__).parent / "robot_control_train_v2.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to: {output_path}")
    
    # Also save a summary
    summary = {
        "total_examples": len(all_examples),
        "breakdown": {
            "phase": len(phase_examples),
            "tool": len(tool_examples),
            "ai_literacy": len(ai_examples),
            "ethics": len(ethics_examples),
            "conversational": len(conv_examples),
            "overview": len(overview_examples)
        }
    }
    
    summary_path = Path(__file__).parent / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    # Preview some examples
    print("\n" + "=" * 60)
    print("SAMPLE EXAMPLES")
    print("=" * 60)
    
    for i, ex in enumerate(random.sample(all_examples, min(5, len(all_examples)))):
        print(f"\n{i+1}. Q: {ex['instruction']}")
        print(f"   A: {ex['response'][:100]}...")


if __name__ == "__main__":
    main()