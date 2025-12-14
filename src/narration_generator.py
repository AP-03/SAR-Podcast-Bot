"""
Narration Generator for SAR-Podcast-Bot
Bridges Vision Pipeline (CNN/LSTM) → GPT-2 Core Model → Podcast Script

This module:
1. Loads NPZ output from vision pipeline (main.py)
2. Segments video into coherent phase segments
3. Generates prompts for GPT-2 based on detected phases/tools
4. Produces podcast-style narration for the video

Usage:
    python narration_generator.py --npz results/final_predictions.npz --output podcast_script.txt
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from collections import Counter
from pathlib import Path
from datetime import timedelta

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'src'))

from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# CONFIGURATION & MAPPINGS
# =============================================================================

# Phase names from vision system → knowledge base mapping
# Vision uses: "ClippingCutting", Knowledge base uses: "Clipping/Cutting"
PHASE_TO_KB_MAPPING = {
    "Preparation": "Preparation",
    "CalotTriangleDissection": "Dissection",  # Map to Dissection in KB
    "ClippingCutting": "Clipping/Cutting",
    "GallbladderDissection": "Gallbladder Dissection",
    "GallbladderRetraction": "Gallbladder Retraction",
    "CleaningCoagulation": "Cleaning Coagulation",
    "GallbladderPackaging": "Gallbladder Packaging",
}

# Human-readable phase names for narration
PHASE_DISPLAY_NAMES = {
    "Preparation": "Preparation",
    "CalotTriangleDissection": "Calot Triangle Dissection",
    "ClippingCutting": "Clipping and Cutting",
    "GallbladderDissection": "Gallbladder Dissection",
    "GallbladderRetraction": "Gallbladder Retraction",
    "CleaningCoagulation": "Cleaning and Coagulation",
    "GallbladderPackaging": "Gallbladder Packaging",
}

# Tool display names
TOOL_DISPLAY_NAMES = {
    "Grasper": "Grasper",
    "Bipolar": "Bipolar Forceps",
    "Hook": "Electrocautery Hook",
    "Scissors": "Scissors",
    "Clipper": "Clip Applier",
    "Irrigator": "Irrigator/Aspirator",
    "SpecimenBag": "Specimen Retrieval Bag",
}


# =============================================================================
# KNOWLEDGE BASE (embedded for reliability)
# =============================================================================

PHASE_KNOWLEDGE = {
    "Preparation": {
        "concept": "Visual SLAM",
        "fact": "During preparation, a robot uses Visual SLAM (Simultaneous Localization and Mapping). It tracks feature points on the cavity walls to build a 3D depth map of the environment."
    },
    "Dissection": {
        "concept": "Motion Scaling",
        "fact": "Dissection requires precision. A robot uses Motion Scaling (e.g., 5:1 ratio), converting a 5cm hand movement into a 1cm tool movement to prevent accidental cuts."
    },
    "Clipping/Cutting": {
        "concept": "Tremor Filtration",
        "fact": "Clipping the artery requires steadiness. The robot uses a 6Hz Low-Pass Filter to remove the surgeon's natural hand tremors, ensuring the clip is placed perfectly."
    },
    "Gallbladder Retraction": {
        "concept": "Active Constraints (Virtual Fixtures)",
        "fact": "To hold the organ safely, the robot uses Active Constraints. These are software 'invisible walls' that prevent the tool from slipping into the liver while maintaining tension."
    },
    "Gallbladder Dissection": {
        "concept": "Inverse Kinematics",
        "fact": "The robot uses Inverse Kinematics to calculate the exact joint angles needed to maneuver the tool behind the gallbladder without the robot arms colliding."
    },
    "Cleaning Coagulation": {
        "concept": "Augmented Reality (Fluorescence)",
        "fact": "To find bleeding spots, robotic systems overlay Augmented Reality feeds (like Firefly fluorescence) to highlight blood flow in green on the surgeon's screen."
    },
    "Gallbladder Packaging": {
        "concept": "Master-Slave Teleoperation",
        "fact": "Bagging the specimen relies on Master-Slave Teleoperation algorithms. The system compensates for processing latency to ensure the robot moves instantly with the surgeon's hands."
    }
}

TOOL_KNOWLEDGE = {
    "Hook": {
        "concept": "EndoWrist (7 DOFs)",
        "fact": "The manual Hook is rigid. A robotic hook uses EndoWrist technology with 7 Degrees of Freedom, allowing it to rotate 360 degrees to hook tissue from behind."
    },
    "Grasper": {
        "concept": "Haptic Feedback Simulation",
        "fact": "Robots lack touch. To compensate, algorithms use Visual Haptics—analyzing tissue deformation in the video to estimate force."
    },
    "Clipper": {
        "concept": "Articulated Clip Applier",
        "fact": "A robotic Clipper can articulate (bend) at the wrist, allowing the surgeon to place clips on the artery from the side without contorting their own arm."
    },
    "Scissors": {
        "concept": "Tremor Filtration (6Hz Low-Pass)",
        "fact": "Manual snipping can be shaky. The robotic scissors use a digital filter to remove the surgeon's physiological hand tremors, allowing for smooth, confident cuts."
    },
    "Bipolar": {
        "concept": "Multitasking Efficiency",
        "fact": "The robotic Maryland Bipolar can dissect, grasp, and coagulate simultaneously due to its wrist articulation. This reduces instrument swaps compared to the rigid manual bipolar tool."
    },
    "Irrigator": {
        "concept": "Console Foot-Pedal Control",
        "fact": "While manual irrigation requires a hand, robotic suction/irrigation is often controlled via foot pedals at the console, freeing both hands for operating instruments."
    },
    "SpecimenBag": {
        "concept": "Assistant Port Coordination",
        "fact": "Robots struggle with soft, floppy objects like bags. The Specimen Bag is typically introduced through a special 'Assistant Port' by a human nurse."
    }
}


# =============================================================================
# NARRATION GENERATOR CLASS
# =============================================================================

class NarrationGenerator:
    """Generate podcast-style narration from vision pipeline output"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the narration generator with GPT-2 model
        
        Args:
            model_path: Path to trained GPT-2 model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print(f"Loading GPT-2 model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
    def generate_response(self, prompt, max_length=200, temperature=0.7):
        """Generate a response using the GPT-2 model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after prompt)
        if full_text.startswith(prompt):
            response = full_text[len(prompt):].strip()
        else:
            response = full_text.strip()
            
        return response


class VisionResultsLoader:
    """Load and process NPZ results from vision pipeline"""
    
    def __init__(self, npz_path):
        """Load NPZ file from vision pipeline"""
        print(f"Loading vision results from: {npz_path}")
        self.data = np.load(npz_path, allow_pickle=True)
        
        # Print available keys for debugging
        print(f"Available keys in NPZ: {list(self.data.keys())}")
        
        # Load core data
        self.frame_indices = self.data.get('frame_indices', [])
        self.timestamps = self.data.get('timestamps', [])
        self.phase_predictions = self.data.get('phase_predictions', [])
        self.phase_confidences = self.data.get('phase_confidences', [])
        self.tool_predictions = self.data.get('tool_predictions', [])
        self.tool_confidences = self.data.get('tool_confidences', [])
        
        # LSTM results if available
        self.lstm_actions = self.data.get('lstm_actions', None)
        self.lstm_confidences_arr = self.data.get('lstm_confidences', None)
        
        print(f"Loaded {len(self.frame_indices)} frames")
        
    def get_phase_segments(self, min_segment_duration=2.0):
        """
        Segment video into coherent phase segments
        
        Args:
            min_segment_duration: Minimum segment duration in seconds
            
        Returns:
            List of segments: [{'phase', 'start_time', 'end_time', 'tools', 'confidence'}]
        """
        # Use LSTM predictions if available, otherwise CNN
        phases = self.lstm_actions if self.lstm_actions is not None else self.phase_predictions
        
        if len(phases) == 0:
            return []
            
        segments = []
        current_phase = phases[0]
        start_idx = 0
        
        for i, phase in enumerate(phases):
            if phase != current_phase:
                # End current segment
                segment = self._create_segment(start_idx, i - 1, current_phase)
                if segment['duration'] >= min_segment_duration:
                    segments.append(segment)
                
                # Start new segment
                current_phase = phase
                start_idx = i
        
        # Add final segment
        segment = self._create_segment(start_idx, len(phases) - 1, current_phase)
        if segment['duration'] >= min_segment_duration:
            segments.append(segment)
            
        return segments
    
    def _create_segment(self, start_idx, end_idx, phase):
        """Create a segment dictionary"""
        # Get timestamps
        start_time = self.timestamps[start_idx] if len(self.timestamps) > start_idx else start_idx / 25.0
        end_time = self.timestamps[end_idx] if len(self.timestamps) > end_idx else end_idx / 25.0
        
        # Get tools used in this segment
        segment_tools = self.tool_predictions[start_idx:end_idx+1] if len(self.tool_predictions) > 0 else []
        
        # Count tool occurrences
        tool_counts = Counter()
        for tool_list in segment_tools:
            if isinstance(tool_list, (list, np.ndarray)):
                for tool in tool_list:
                    tool_counts[tool] += 1
            else:
                tool_counts[tool_list] += 1
        
        # Get most common tools
        top_tools = [tool for tool, count in tool_counts.most_common(3)]
        
        # Calculate average confidence (max per frame for interpretability)
        segment_conf = self.phase_confidences[start_idx:end_idx+1] if len(self.phase_confidences) > 0 else [0.5]
        if len(segment_conf) > 0:
            try:
                frame_max = [float(np.max(c)) for c in segment_conf]
                avg_confidence = float(np.mean(frame_max))
            except Exception:
                avg_confidence = float(np.mean(segment_conf))
        else:
            avg_confidence = 0.0
        
        return {
            'phase': phase,
            'start_time': float(start_time),
            'end_time': float(end_time),
            'duration': float(end_time - start_time),
            'tools': top_tools,
            'confidence': float(avg_confidence),
            'frame_count': end_idx - start_idx + 1
        }


def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    return str(timedelta(seconds=int(seconds)))[2:7]


def generate_podcast_script(generator, segments, video_name="surgical video"):
    """
    Generate a full podcast script from video segments
    
    Args:
        generator: NarrationGenerator instance
        segments: List of phase segments from VisionResultsLoader
        video_name: Name of the video for the script
        
    Returns:
        Full podcast script as string
    """
    script_parts = []
    
    # ==========================================================================
    # INTRODUCTION
    # ==========================================================================
    script_parts.append("=" * 70)
    script_parts.append("SAR-PODCAST-BOT: AI-POWERED SURGICAL NARRATION")
    script_parts.append("=" * 70)
    script_parts.append("")
    script_parts.append(f"Video: {video_name}")
    script_parts.append(f"Total Duration: {format_timestamp(segments[-1]['end_time'] if segments else 0)}")
    script_parts.append(f"Phases Detected: {len(segments)}")
    script_parts.append("")
    
    # Opening narration
    script_parts.append("-" * 70)
    script_parts.append("INTRODUCTION")
    script_parts.append("-" * 70)
    script_parts.append("")
    script_parts.append("Welcome to this AI-assisted surgical education session. Today, our")
    script_parts.append("computer vision system will analyze a laparoscopic cholecystectomy")
    script_parts.append("(gallbladder removal) and explain how robotic surgical systems would")
    script_parts.append("approach each phase using advanced control algorithms.")
    script_parts.append("")
    
    # ==========================================================================
    # PHASE-BY-PHASE NARRATION
    # ==========================================================================
    script_parts.append("-" * 70)
    script_parts.append("PHASE-BY-PHASE ANALYSIS")
    script_parts.append("-" * 70)
    
    for i, segment in enumerate(segments, 1):
        phase = segment['phase']
        phase_display = PHASE_DISPLAY_NAMES.get(phase, phase)
        kb_phase = PHASE_TO_KB_MAPPING.get(phase, phase)
        
        script_parts.append("")
        script_parts.append(f"[{format_timestamp(segment['start_time'])} - {format_timestamp(segment['end_time'])}]")
        script_parts.append(f"PHASE {i}: {phase_display.upper()}")
        script_parts.append(f"Duration: {segment['duration']:.1f}s | Confidence: {segment['confidence']*100:.1f}%")
        script_parts.append("")
        
        # Get knowledge for this phase
        phase_info = PHASE_KNOWLEDGE.get(kb_phase, PHASE_KNOWLEDGE.get(phase, None))
        
        if phase_info:
            # Generate GPT-2 response for the phase
            prompt = f"The vision system detects '{phase_display}'. What robotic algorithm applies here-"
            gpt_response = generator.generate_response(prompt, max_length=200)
            
            script_parts.append(f"[AI Analysis]")
            script_parts.append(f"Our vision system has detected the {phase_display} phase.")
            script_parts.append(f"Key Concept: {phase_info['concept']}")
            script_parts.append("")
            script_parts.append(f"[GPT-2 Explanation]")
            script_parts.append(gpt_response[:500])  # Truncate if too long
            script_parts.append("")
        
        # Tool analysis
        if segment['tools']:
            script_parts.append(f"[Tools Detected]")
            for tool in segment['tools']:
                tool_display = TOOL_DISPLAY_NAMES.get(tool, tool)
                tool_info = TOOL_KNOWLEDGE.get(tool, None)
                
                if tool_info:
                    script_parts.append(f"  • {tool_display}: {tool_info['concept']}")
                else:
                    script_parts.append(f"  • {tool_display}")
            script_parts.append("")
        
        script_parts.append("-" * 40)
    
    # ==========================================================================
    # AI LITERACY SECTION
    # ==========================================================================
    script_parts.append("")
    script_parts.append("-" * 70)
    script_parts.append("AI LITERACY DISCUSSION")
    script_parts.append("-" * 70)
    script_parts.append("")
    
    # Ask the bot AI literacy questions
    ai_questions = [
        "How does a computer learn to recognize surgical phases-",
        "Can AI in surgery make mistakes- How do we prevent errors-",
        "Will AI replace surgeons in the future-"
    ]
    
    for q in ai_questions:
        script_parts.append(f"Q: {q}")
        response = generator.generate_response(q, max_length=150, temperature=0.8)
        script_parts.append(f"A: {response[:400]}")
        script_parts.append("")
    
    # ==========================================================================
    # CONCLUSION
    # ==========================================================================
    script_parts.append("-" * 70)
    script_parts.append("CONCLUSION")
    script_parts.append("-" * 70)
    script_parts.append("")
    script_parts.append("This demonstration showed how AI vision systems can analyze surgical")
    script_parts.append("video in real-time, identifying phases and tools while explaining")
    script_parts.append("the robotic control concepts that enable safer, more precise surgery.")
    script_parts.append("")
    script_parts.append("Key technologies covered:")
    
    # Summarize concepts mentioned
    concepts_mentioned = set()
    for segment in segments:
        kb_phase = PHASE_TO_KB_MAPPING.get(segment['phase'], segment['phase'])
        if kb_phase in PHASE_KNOWLEDGE:
            concepts_mentioned.add(PHASE_KNOWLEDGE[kb_phase]['concept'])
    
    for concept in concepts_mentioned:
        script_parts.append(f"  • {concept}")
    
    script_parts.append("")
    script_parts.append("=" * 70)
    script_parts.append("END OF NARRATION")
    script_parts.append("=" * 70)
    
    return "\n".join(script_parts)


def generate_simple_narration(segments):
    """
    Generate narration without GPT-2 (for testing or fallback)
    Uses only the embedded knowledge base
    """
    script_parts = []
    
    script_parts.append("=" * 70)
    script_parts.append("SURGICAL VIDEO ANALYSIS (Knowledge Base Only)")
    script_parts.append("=" * 70)
    script_parts.append("")
    
    for i, segment in enumerate(segments, 1):
        phase = segment['phase']
        phase_display = PHASE_DISPLAY_NAMES.get(phase, phase)
        kb_phase = PHASE_TO_KB_MAPPING.get(phase, phase)
        
        script_parts.append(f"\n[{format_timestamp(segment['start_time'])} - {format_timestamp(segment['end_time'])}]")
        script_parts.append(f"PHASE {i}: {phase_display}")
        
        phase_info = PHASE_KNOWLEDGE.get(kb_phase, PHASE_KNOWLEDGE.get(phase, None))
        if phase_info:
            script_parts.append(f"Robotic Concept: {phase_info['concept']}")
            script_parts.append(f"Explanation: {phase_info['fact']}")
        
        if segment['tools']:
            script_parts.append(f"Tools: {', '.join(segment['tools'])}")
        
        script_parts.append("")
    
    return "\n".join(script_parts)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate podcast narration from vision results")
    parser.add_argument('--npz', type=str, required=True,
                       help='Path to NPZ file from vision pipeline')
    parser.add_argument('--model', type=str, 
                       default='src/results/core_results/gpt2_best_model',
                       help='Path to trained GPT-2 model')
    parser.add_argument('--output', type=str, default='podcast_script.txt',
                       help='Output path for podcast script')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no-gpt', action='store_true',
                       help='Generate narration without GPT-2 (knowledge base only)')
    parser.add_argument('--min-segment', type=float, default=2.0,
                       help='Minimum segment duration in seconds')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAR-PODCAST-BOT NARRATION GENERATOR")
    print("=" * 60)
    
    # Load vision results
    loader = VisionResultsLoader(args.npz)
    segments = loader.get_phase_segments(min_segment_duration=args.min_segment)
    
    print(f"\nDetected {len(segments)} phase segments:")
    for i, seg in enumerate(segments, 1):
        print(f"  {i}. {seg['phase']}: {format_timestamp(seg['start_time'])} - {format_timestamp(seg['end_time'])} ({seg['duration']:.1f}s)")
    
    # Generate narration
    if args.no_gpt:
        print("\nGenerating narration (knowledge base only)...")
        script = generate_simple_narration(segments)
    else:
        print("\nInitializing GPT-2 model...")
        generator = NarrationGenerator(args.model, device=args.device)
        
        print("\nGenerating podcast script...")
        video_name = Path(args.npz).stem
        script = generate_podcast_script(generator, segments, video_name)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(script)
    
    print(f"\nPodcast script saved to: {output_path}")
    print("\n" + "=" * 60)
    print("PREVIEW (first 50 lines):")
    print("=" * 60)
    preview_lines = script.split('\n')[:50]
    print('\n'.join(preview_lines))
    if len(script.split('\n')) > 50:
        print(f"\n... ({len(script.split(chr(10))) - 50} more lines)")
    
    print("\n" + "=" * 60)
    print("Narration generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
