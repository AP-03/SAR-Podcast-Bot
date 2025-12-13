"""
Interactive Podcast Demo for SAR-Podcast-Bot
Real-time conversation with the GPT-2 Core model for podcast recording

Features:
- Load trained GPT-2 model
- Interactive Q&A with the bot
- Pre-loaded questions for AI literacy and ethics
- Save conversation history
- Compare responses between Core model and SOTA (if available)

Usage:
    python interactive_demo.py --model src/results/core_results/gpt2_best_model
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM


class PodcastBot:
    """Interactive chatbot for podcast demo"""
    
    def __init__(self, model_path, device='cuda', model_name="Core GPT-2"):
        """Initialize the podcast bot"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"Loading {model_name} from: {model_path}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        print(f"✓ {model_name} loaded successfully!")
        
        # Conversation history
        self.history = []
        
    def respond(self, prompt, max_length=200, temperature=0.7, show_prompt=False):
        """Generate a response to a prompt"""
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
        
        # Extract generated part
        if full_text.startswith(prompt):
            response = full_text[len(prompt):].strip()
        else:
            response = full_text.strip()
        
        # Store in history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response': response,
            'model': self.model_name
        })
        
        if show_prompt:
            return full_text
        return response
    
    def save_history(self, filepath):
        """Save conversation history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to: {filepath}")


# Pre-defined questions for the podcast
PODCAST_QUESTIONS = {
    "ai_literacy": [
        "How does a computer learn?",
        "How can I explain to my grandmother what is a Neural Network?",
        "Why do we need a lot of computational resources to process AI methods?",
        "What is Deep Learning and how is it different from regular programming?",
        "How does our vision system recognize surgical tools in the video?",
    ],
    "ethics": [
        "Will AI take over?",
        "Can everyone benefit from AI?",
        "How to prevent AI from being wrongly used?",
        "Should we trust AI in medical surgery?",
        "What are the risks of AI in healthcare?",
    ],
    "surgical_robotics": [
        "The vision system detects 'Preparation'. What robotic algorithm applies here?",
        "Explain the control theory behind robotic Dissection.",
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        "Why is the robotic approach to 'Clipping/Cutting' considered safer?",
        "How does Motion Scaling help surgeons during dissection?",
        "What is Visual SLAM and why is it important for surgical robots?",
    ],
    "fun": [
        "Tell me a joke about robots.",
        "What is your favorite surgical tool?",
        "If you were a surgeon, what would be your specialty?",
    ]
}


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print("   ███████╗ █████╗ ██████╗       ██████╗  ██████╗ ████████╗")
    print("   ██╔════╝██╔══██╗██╔══██╗      ██╔══██╗██╔═══██╗╚══██╔══╝")
    print("   ███████╗███████║██████╔╝█████╗██████╔╝██║   ██║   ██║   ")
    print("   ╚════██║██╔══██║██╔══██╗╚════╝██╔══██╗██║   ██║   ██║   ")
    print("   ███████║██║  ██║██║  ██║      ██████╔╝╚██████╔╝   ██║   ")
    print("   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝      ╚═════╝  ╚═════╝    ╚═╝   ")
    print("=" * 70)
    print("   AI-Powered Surgical Robotics Podcast Bot")
    print("   Interactive Demo for COMP0220 Deep Learning Coursework")
    print("=" * 70 + "\n")


def print_help():
    """Print help menu"""
    print("\n" + "-" * 50)
    print("COMMANDS:")
    print("-" * 50)
    print("  /help     - Show this help menu")
    print("  /questions - Show pre-defined podcast questions")
    print("  /ask [n]  - Ask a pre-defined question by number")
    print("  /category [name] - Show questions in a category")
    print("  /history  - Show conversation history")
    print("  /save     - Save conversation history to file")
    print("  /clear    - Clear conversation history")
    print("  /quit     - Exit the demo")
    print("-" * 50)
    print("Or just type your question directly!\n")


def print_questions():
    """Print all pre-defined questions"""
    print("\n" + "=" * 50)
    print("PRE-DEFINED PODCAST QUESTIONS")
    print("=" * 50)
    
    idx = 1
    question_map = {}
    
    for category, questions in PODCAST_QUESTIONS.items():
        print(f"\n[{category.upper().replace('_', ' ')}]")
        for q in questions:
            print(f"  {idx}. {q}")
            question_map[idx] = q
            idx += 1
    
    print("\nUse '/ask [number]' to ask a specific question")
    print("=" * 50 + "\n")
    
    return question_map


def interactive_session(bot, question_map):
    """Run interactive conversation session"""
    print("\n🎙️  Ready for podcast! Type your questions or use /help for commands.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else None
                
                if cmd == 'quit' or cmd == 'exit':
                    print("\n👋 Thanks for using SAR-Podcast-Bot! Goodbye!")
                    break
                    
                elif cmd == 'help':
                    print_help()
                    
                elif cmd == 'questions':
                    question_map = print_questions()
                    
                elif cmd == 'ask' and arg:
                    try:
                        q_num = int(arg)
                        if q_num in question_map:
                            question = question_map[q_num]
                            print(f"\n📝 Question: {question}")
                            print("-" * 50)
                            response = bot.respond(question)
                            print(f"\n🤖 {bot.model_name}: {response}\n")
                        else:
                            print(f"Invalid question number. Use /questions to see available questions.")
                    except ValueError:
                        print("Usage: /ask [number]")
                        
                elif cmd == 'category' and arg:
                    cat = arg.lower().replace(' ', '_')
                    if cat in PODCAST_QUESTIONS:
                        print(f"\n[{cat.upper().replace('_', ' ')}]")
                        for i, q in enumerate(PODCAST_QUESTIONS[cat], 1):
                            print(f"  {i}. {q}")
                    else:
                        print(f"Unknown category. Available: {', '.join(PODCAST_QUESTIONS.keys())}")
                        
                elif cmd == 'history':
                    if bot.history:
                        print("\n" + "=" * 50)
                        print("CONVERSATION HISTORY")
                        print("=" * 50)
                        for i, entry in enumerate(bot.history, 1):
                            print(f"\n{i}. Q: {entry['prompt'][:60]}...")
                            print(f"   A: {entry['response'][:100]}...")
                        print("=" * 50 + "\n")
                    else:
                        print("No conversation history yet.")
                        
                elif cmd == 'save':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"podcast_history_{timestamp}.json"
                    bot.save_history(filepath)
                    
                elif cmd == 'clear':
                    bot.history = []
                    print("Conversation history cleared.")
                    
                else:
                    print(f"Unknown command: {cmd}. Type /help for available commands.")
            
            else:
                # Regular question - send to bot
                print("-" * 50)
                response = bot.respond(user_input)
                print(f"\n🤖 {bot.model_name}: {response}\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def demo_mode(bot, question_map):
    """Run automated demo with pre-defined questions"""
    print("\n" + "=" * 70)
    print("AUTOMATED DEMO MODE")
    print("=" * 70)
    
    # Demo questions from each category
    demo_questions = [
        # AI Literacy
        "How does a computer learn?",
        "How can I explain to my grandmother what is a Neural Network?",
        # Surgical Robotics
        "The vision system detects 'Preparation'. What robotic algorithm applies here?",
        "The vision system detects the tool 'Grasper'. What is the robotic equivalent?",
        # Ethics
        "Will AI take over?",
        "Can everyone benefit from AI?",
    ]
    
    for q in demo_questions:
        print(f"\n📝 Question: {q}")
        print("-" * 50)
        response = bot.respond(q)
        print(f"🤖 {bot.model_name}: {response}")
        print()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Interactive Podcast Demo")
    parser.add_argument('--model', type=str, 
                       default='src/results/core_results/gpt2_best_model',
                       help='Path to trained GPT-2 model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--demo', action='store_true',
                       help='Run automated demo instead of interactive mode')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Generation temperature (0.1-1.0)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found at {args.model}")
        print("Please check the path or train the model first.")
        return
    
    # Initialize bot
    bot = PodcastBot(args.model, device=args.device)
    
    # Build question map
    question_map = {}
    idx = 1
    for category, questions in PODCAST_QUESTIONS.items():
        for q in questions:
            question_map[idx] = q
            idx += 1
    
    if args.demo:
        demo_mode(bot, question_map)
    else:
        print_help()
        interactive_session(bot, question_map)
    
    # Save history on exit
    if bot.history:
        save = input("\n💾 Save conversation history? (y/n): ").strip().lower()
        if save == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"podcast_history_{timestamp}.json"
            bot.save_history(filepath)


if __name__ == "__main__":
    main()