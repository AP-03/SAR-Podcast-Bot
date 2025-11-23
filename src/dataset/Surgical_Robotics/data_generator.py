import json

# 1. Load the knowledge base (Your source "Map")
# Make sure 'phase_to_control_mapping.json' contains the dictionary you created earlier
try:
    with open('phase_to_control_mapping.json', 'r') as f:
        knowledge_base = json.load(f)
    with open('tool_to_robot_mapping.json', 'r') as f:
        tool_knowledge = json.load(f)
except FileNotFoundError:
    print("Error: Knowledge base files not found. Please ensure 'phase_to_control_mapping.json' and 'tool_to_robot_mapping.json' exist.")
    exit(1)

dataset_phase = []

for label, info in knowledge_base.items():
    # Angle 1: Direct identification
    dataset_phase.append({
        "instruction": f"The vision system detects '{label}'. What robotic algorithm applies here?",
        "response": f"For {label}, the critical concept is **{info['concept']}**. {info['fact']}"
    })
    
    # Angle 2: Comparison
    dataset_phase.append({
        "instruction": f"I see manual '{label}'. How would a robot do this differently?",
        "response": f"While the human surgeon performs {label} manually, a robot would rely on **{info['concept']}**. {info['fact']}"
    })

    # Angle 3: "How it works"
    dataset_phase.append({
        "instruction": f"Explain the control theory behind robotic {label}.",
        "response": info['fact']
    })

    # Angle 4: The "Safety" Focus
    dataset_phase.append({
        "instruction": f"Why is the robotic approach to '{label}' considered safer?",
        "response": f"Safety during {label} is improved by **{info['concept']}**, because {info['fact']}"
    })

    # Angle 5: The "Short" Answer (Good for variety)
    dataset_phase.append({
        "instruction": f"Quickly identify the robotic algorithm for '{label}'.",
        "response": f"The key algorithm is **{info['concept']}**."
    })

dataset_tools = []

for label, info in tool_knowledge.items():
    
    # Angle 1: The "Translator" (Identity)
    # Purpose: Connects the Visual Label to the Robotic Equivalent
    dataset_tools.append({
        "instruction": f"The vision system detects the tool '{label}'. What is the robotic equivalent?",
        "response": f"You are seeing a manual {label}. In a robotic system, the equivalent uses **{info['concept']}**. {info['fact']}"
    })
    
    # Angle 2: The "Engineer" (Control Concept)
    # Purpose: Explains the specific technology (EndoWrist, Filter, etc.)
    dataset_tools.append({
        "instruction": f"Explain the robotic control concept associated with the {label}.",
        "response": f"For the {label} functionality, the critical concept is **{info['concept']}**. {info['fact']}"
    })

    # Angle 3: The "Analyst" (Comparison)
    # Purpose: Contrasts manual limitations with robotic advantages
    dataset_tools.append({
        "instruction": f"Contrast the manual {label} on screen with a robotic system.",
        "response": f"The manual {label} is limited by human dexterity. A robot improves this using **{info['concept']}**. {info['fact']}"
    })

    # Angle 4: The "Surgeon" (Safety & Outcome) - NEW!
    # Purpose: Focuses on why this matters for the patient/surgeon
    dataset_tools.append({
        "instruction": f"How does the robotic {label} improve surgical safety?",
        "response": f"Safety is enhanced because **{info['concept']}** allows for greater precision. {info['fact']}"
    })

    # Angle 5: The "Futurist" (AI Autonomy) - NEW!
    # Purpose: Speculates on future AI automation (Great for the 'Will AI take over?' question)
    dataset_tools.append({
        "instruction": f"Could an AI operate the {label} autonomously in the future?",
        "response": f"It is challenging but possible. To automate the {label}, an AI would need to perfectly master **{info['concept']}** to match human dexterity. Currently, AI is mostly used to assist, not replace, this action."
    })

# 3. Save it to a NEW file (Standard JSON List format)
# We use 'w' (write) mode to create a valid, clean file.
output_filename = 'robot_control_train.json'
with open(output_filename, 'w') as f:
    json.dump(dataset_phase + dataset_tools, f, indent=2)

print(f"Success! Dataset Generated: {output_filename}")
print(f"Total examples created: {len(dataset_phase) + len(dataset_tools)}")