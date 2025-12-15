"""
Main pipeline for SAR-Podcast-Bot
Processes surgical videos through trained models to generate podcast content
"""
import os
import re
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tool_resnet import ToolCNN
from models.action_LSTM import ActionLSTMWithAttention
from dataset.transform import get_basic_transforms
from configs.labels import PHASES, TOOLS

# Import narration generation components
from narration_generator import NarrationGenerator, VisionResultsLoader, generate_podcast_script


class VideoProcessor:
    """Process surgical video through trained CNN to extract frame-level predictions"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize video processor with trained CNN model
        
        Args:
            model_path: Path to trained tool detection model (.pth file)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Get transforms
        self.transform = get_basic_transforms()
        
        # Label mappings
        self.phases = PHASES
        self.tools = TOOLS
        
    def _load_model(self, model_path):
        """Load trained ToolCNN model"""
        print(f"Loading model from: {model_path}")
        
        # Initialize model architecture
        model = ToolCNN(
            num_tools=len(TOOLS),
            num_stages=len(PHASES),
            pretrained=False  # We're loading trained weights
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        print("Model loaded successfully!")
        return model
    
    def process_video(self, video_path, sample_rate=1):
        """
        Process video through CNN to get frame-level predictions
        
        Args:
            video_path: Path to input video file
            sample_rate: Process every Nth frame (default: 1 = all frames)
            
        Returns:
            results: Dict containing frame-level predictions
        """
        print(f"\nProcessing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames @ {fps:.2f} FPS")
        print(f"Sample rate: every {sample_rate} frame(s)")
        
        # Storage for results
        results = {
            'frame_indices': [],
            'timestamps': [],
            'tool_predictions': [],
            'tool_confidences': [],
            'phase_predictions': [],
            'phase_confidences': [],
            'features': []
        }
        
        frame_idx = 0
        processed_count = 0
        
        with torch.no_grad():
            pbar = tqdm(total=total_frames // sample_rate, desc="Processing frames")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_idx % sample_rate == 0:
                    # Preprocess frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                    
                    # Run model
                    tool_logits, phase_logits, features = self.model(input_tensor, return_features=True)
                    
                    # Convert to predictions
                    tool_probs = torch.sigmoid(tool_logits).cpu().numpy()[0]  # Multi-label
                    phase_probs = torch.softmax(phase_logits, dim=1).cpu().numpy()[0]  # Single-label
                    
                    # Get predictions (threshold tools at 0.5)
                    active_tools = [self.tools[i] for i, prob in enumerate(tool_probs) if prob > 0.5]
                    predicted_phase = self.phases[phase_probs.argmax()]
                    
                    # Store results
                    results['frame_indices'].append(frame_idx)
                    results['timestamps'].append(frame_idx / fps)
                    results['tool_predictions'].append(active_tools)
                    results['tool_confidences'].append(tool_probs.tolist())
                    results['phase_predictions'].append(predicted_phase)
                    results['phase_confidences'].append(phase_probs.tolist())
                    results['features'].append(features.cpu().numpy()[0])
                    
                    processed_count += 1
                    pbar.update(1)
                
                frame_idx += 1
            
            pbar.close()
        
        cap.release()
        
        print(f"\nProcessed {processed_count} frames")
        
        # Convert lists to numpy arrays
        results['features'] = np.array(results['features'])
        
        return results
    
    def save_results(self, results, output_path):
        """Save processing results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy archive
        np.savez_compressed(
            output_path,
            frame_indices=results['frame_indices'],
            timestamps=results['timestamps'],
            tool_predictions=results['tool_predictions'],
            tool_confidences=results['tool_confidences'],
            phase_predictions=results['phase_predictions'],
            phase_confidences=results['phase_confidences'],
            features=results['features']
        )
        print(f"Results saved to: {output_path}")


class ActionPredictor:
    """Process CNN features through trained LSTM to predict surgical actions"""
    
    def __init__(self, model_path, num_actions, device='cuda'):
        """
        Initialize action predictor with trained LSTM model
        
        Args:
            model_path: Path to trained LSTM model (.pth file)
            num_actions: Number of action/phase classes
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda_is_available() else 'cpu')
        print(f"Loading LSTM model on device: {self.device}")
        
        # Load model
        # NOTE: The best checkpoint was trained with hidden_dim=128, dropout=0.7,
        # and num_layers=2 (see src/hype/LSTM.yaml). Keep these in sync with
        # the saved weights to avoid size mismatches when loading.
        self.model = self._load_lstm_model(
            model_path,
            num_actions,
            hidden_dim=128,
            num_layers=2,
            dropout=0.7,
        )
        self.model.eval()
        
        # Action labels
        self.action_labels = PHASES  # Using phase labels as actions
        
    def _load_lstm_model(self, model_path, num_actions, hidden_dim=128, num_layers=2, dropout=0.7):
        """Load trained LSTM model"""
        print(f"Loading LSTM from: {model_path}")
        
        # Initialize model architecture
        model = ActionLSTMWithAttention(
            num_actions=num_actions,
            feature_dim=2048,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        print("LSTM model loaded successfully!")
        return model
    
    def predict_sequence(self, features, window_size=16, stride=8):
        """
        Predict actions from feature sequence using sliding window
        
        Args:
            features: numpy array of shape [num_frames, feature_dim]
            window_size: Number of frames to process at once
            stride: Number of frames to slide between windows
            
        Returns:
            predictions: List of predicted actions per window
            confidences: List of confidence scores per window
            attention_weights: List of attention weights per window
        """
        print(f"\nPredicting actions from {len(features)} feature vectors...")
        print(f"Window size: {window_size}, Stride: {stride}")
        
        num_frames = len(features)
        predictions = []
        confidences = []
        attention_weights_list = []
        window_indices = []
        
        with torch.no_grad():
            # Slide window over sequence
            for start_idx in tqdm(range(0, num_frames - window_size + 1, stride), 
                                 desc="Processing windows"):
                end_idx = start_idx + window_size
                
                # Extract window of features
                window_features = features[start_idx:end_idx]
                
                # Convert to tensor [1, window_size, feature_dim]
                window_tensor = torch.from_numpy(window_features).float().unsqueeze(0).to(self.device)
                
                # Predict
                logits, probs, attn_weights = self.model(window_tensor, return_sequence=False)
                
                # Get prediction
                pred_idx = torch.argmax(probs, dim=1).cpu().item()
                confidence = probs[0, pred_idx].cpu().item()
                
                predictions.append(self.action_labels[pred_idx])
                confidences.append(confidence)
                attention_weights_list.append(attn_weights.cpu().numpy()[0])
                window_indices.append((start_idx, end_idx))
        
        print(f"Processed {len(predictions)} windows")
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'attention_weights': attention_weights_list,
            'window_indices': window_indices
        }
    
    def aggregate_predictions(self, predictions_dict, num_frames):
        """
        Aggregate sliding window predictions to frame-level predictions
        
        Args:
            predictions_dict: Output from predict_sequence
            num_frames: Total number of frames in video
            
        Returns:
            frame_predictions: List of predicted action per frame
            frame_confidences: List of confidence per frame
        """
        # Initialize vote count for each frame
        vote_counts = [{label: 0 for label in self.action_labels} for _ in range(num_frames)]
        
        # Accumulate votes from each window
        for pred, conf, (start_idx, end_idx) in zip(
            predictions_dict['predictions'],
            predictions_dict['confidences'],
            predictions_dict['window_indices']
        ):
            for frame_idx in range(start_idx, end_idx):
                if frame_idx < num_frames:
                    vote_counts[frame_idx][pred] += conf
        
        # Get final prediction per frame
        frame_predictions = []
        frame_confidences = []
        
        for votes in vote_counts:
            if sum(votes.values()) > 0:
                pred = max(votes, key=votes.get)
                conf = votes[pred] / sum(votes.values())
            else:
                pred = self.action_labels[0]  # Default to first action
                conf = 0.0
            
            frame_predictions.append(pred)
            frame_confidences.append(conf)
        
        return frame_predictions, frame_confidences


def speak_text(text, rate=190, voice_name=None):
    """
    Convert text to speech using pyttsx3 (cross-platform)
    
    Args:
        text: Text to speak
        rate: Speaking rate in words per minute (default: 150)
        voice_name: Preferred voice name (e.g., 'Samantha', 'Alex') - None for default
    
    Note:
        Install pyttsx3: pip install pyttsx3
        Works on Windows and macOS
    """
    try:
        import pyttsx3
        
        # Initialize TTS engine
        engine = pyttsx3.init()
        
        # Set speaking rate
        engine.setProperty('rate', rate)
        
        # Set voice if specified
        if voice_name:
            voices = engine.getProperty('voices')
            # Try to find voice by name (case-insensitive partial match)
            for voice in voices:
                if voice_name.lower() in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
    except ImportError:
        print("⚠️  pyttsx3 not installed. Install with: pip install pyttsx3")
    except Exception as e:
        print(f"TTS Error: {e}")


def listen_to_speech(timeout=5, phrase_time_limit=10):
    """
    Listen to microphone and convert speech to text using Whisper
    
    Args:
        timeout: Seconds to wait for speech to start (not used with Whisper, kept for compatibility)
        phrase_time_limit: Maximum seconds to record
    
    Returns:
        str: Transcribed text, or None if failed
    """
    try:
        import whisper
        import sounddevice as sd
        import numpy as np
        import tempfile
        import scipy.io.wavfile as wavfile
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else "required library"
        print(f"  ❌ {missing} not installed.")
        print(f"  📦 Install with: pip install openai-whisper sounddevice scipy")
        return None
    
    # Load Whisper model (cache it globally to avoid reloading)
    if not hasattr(listen_to_speech, 'whisper_model'):
        print("📥 Loading Whisper model (one-time, ~39MB)...")
        listen_to_speech.whisper_model = whisper.load_model("tiny")
    
    try:
        print("🎤 Listening... (speak now, will auto-stop after silence)")
        
        # Record audio with better parameters
        sample_rate = 16000  # Whisper expects 16kHz
        duration = phrase_time_limit
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            blocking=True  # Wait until recording is done
        )
        
        print("🔄 Processing speech...")
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_path = temp_audio.name
            # Normalize audio to prevent clipping
            audio_normalized = np.int16(audio_data * 32767)
            wavfile.write(temp_path, sample_rate, audio_normalized)
        
        # Transcribe with Whisper
        result = listen_to_speech.whisper_model.transcribe(
            temp_path, 
            language='english',
            fp16=False,  # Use FP32 on CPU
            verbose=False  # Suppress warnings
        )
        text = result['text'].strip()
        
        # Clean up temp file
        import os
        os.unlink(temp_path)
        
        if text:
            return text
        else:
            print("  ❓ No speech detected")
            return None
            
    except KeyboardInterrupt:
        print("\n  ⏸️  Recording cancelled")
        sd.stop()
        return None
    except Exception as e:
        print(f"  ❌ Speech recognition error: {e}")
        sd.stop()
        return None




def interactive_qa_session(generator, segments, video_name, enable_tts=False, enable_voice_input=False):
    """
    Interactive Q&A session about the processed video with optional voice input
    
    Args:
        generator: NarrationGenerator instance
        segments: List of phase segments from video
        video_name: Name of the video
    """
    import json
    from datetime import datetime
    from pathlib import Path
    
    # Load self-awareness knowledge base
    models_info_path = Path(__file__).parent / "KnowledgeBases" / "models_info.json"
    system_knowledge = {}
    if models_info_path.exists():
        with open(models_info_path, 'r') as f:
            system_knowledge = json.load(f)
    
    # Prepare context about the video
    phase_summary = {}
    for seg in segments:
        phase = seg['phase']
        if phase not in phase_summary:
            phase_summary[phase] = {
                'duration': 0,
                'tools': set(),
                'count': 0
            }
        phase_summary[phase]['duration'] += seg['duration']
        phase_summary[phase]['tools'].update(seg['tools'])
        phase_summary[phase]['count'] += 1
    phase_to_tools = {p: sorted(info['tools']) for p, info in phase_summary.items()}
    detected_phases = list(phase_to_tools.keys())
    detected_tools = sorted({t for ts in phase_to_tools.values() for t in ts})

    
    # Build context string with phase-to-tools mapping
    context_parts = [f"Video: {video_name}"]
    context_parts.append(f"Total duration: {segments[-1]['end_time']:.1f}s")
    context_parts.append(f"Phases detected: {', '.join(phase_summary.keys())}")
    context_parts.append("\nPhase-to-Tools Mapping:")
    for phase, info in phase_summary.items():
        tools_str = ', '.join(sorted(info['tools'])) if info['tools'] else 'None'
        context_parts.append(f"  {phase}: {tools_str}")
    
    video_facts_context = "\n".join(context_parts)
    # Add system self-awareness
    if system_knowledge:
        context_parts.append("\n=== SYSTEM INFORMATION ===")
        context_parts.append(f"I am {system_knowledge.get('system_name', 'SAR-Podcast-Bot')}")
        context_parts.append(f"My vision pipeline: {system_knowledge.get('vision_pipeline', {}).get('architecture', 'CNN + LSTM')}")
        
        # Add CNN info
        cnn_info = system_knowledge.get('vision_pipeline', {}).get('stage_1_cnn', {})
        if cnn_info:
            context_parts.append(f"CNN: {cnn_info.get('model_name', 'ToolCNN')} with {cnn_info.get('backbone', {}).get('architecture', 'ResNet-50')} backbone")
        
        # Add LSTM info
        lstm_info = system_knowledge.get('vision_pipeline', {}).get('stage_2_lstm', {})
        if lstm_info:
            context_parts.append(f"LSTM: {lstm_info.get('model_name', 'ActionLSTMWithAttention')} with attention mechanism")
        
        # Add language model info - IMPORTANT: Tell bot which model it's currently using
        lm_info = system_knowledge.get('language_models', {})
        context_parts.append(f"Language models available: {', '.join(lm_info.keys())}")
        context_parts.append(f"CURRENTLY ACTIVE LANGUAGE MODEL: {generator.model_type}")
        
        # Add details about the active model
        if generator.model_type == 'sota':
            context_parts.append("I am currently using GPT-4o (SOTA model) via OpenAI API")
        elif generator.model_type == 'core':
            context_parts.append("I am currently using fine-tuned GPT-2 (Core model) with LoRA")
        elif generator.model_type == 'dummy':
            context_parts.append("I am currently using Dummy LSTM (baseline model)")
    
    video_context = "\n".join(context_parts)
    
    # Conversation history
    conversation = []
    
    print("\n" + "=" * 70)
    print("🎙️  INTERACTIVE Q&A SESSION")
    print("=" * 70)
    print(f"\n{video_context}\n")
    print("Ask questions about the video, surgical phases, tools, or AI concepts.")
    print("The bot has access to the video analysis results.\n")
    
    # Check Whisper availability if voice input is enabled
    if enable_voice_input:
        try:
            import whisper
            import sounddevice as sd
            # Whisper is available
        except ImportError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else "required library"
            print(f"⚠️  {missing} not installed - voice input disabled")
            print("   Install with: pip install openai-whisper sounddevice scipy")
            print("   You can still toggle it on later with /voice\n")
            enable_voice_input = False
    
    # TTS and Voice Input status
    tts_status = "🔊 ON" if enable_tts else "🔇 OFF"
    voice_status = "🎤 ON" if enable_voice_input else "⌨️  OFF"
    print(f"Text-to-Speech: {tts_status}")
    print(f"Voice Input: {voice_status}\n")
    
    print("Commands:")
    print("  /summary       - Show video summary")
    print("  /phases        - List all detected phases")
    print("  /tools         - List all detected tools")
    print("  /phase <name>  - Show tools used in a specific phase")
    print("  /system        - Show system architecture and capabilities")
    print("  /tts           - Toggle text-to-speech on/off")
    print("  /voice         - Toggle voice input on/off")
    print("  /save          - Save conversation")
    print("  /quit          - Exit Q&A session")
    print("\n" + "-" * 70 + "\n")
    
    while True:
        try:
            # Get input (voice or text)
            if enable_voice_input:
                question = listen_to_speech()
                if question is None:
                    continue
                print(f"You (voice): {question}")
            else:
                question = input("You: ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.startswith('/'):
                cmd = question[1:].lower()
                
                if cmd in ['quit', 'exit', 'q']:
                    print("\n👋 Ending Q&A session...")
                    break
                
                elif cmd == 'summary':
                    print(f"\n📊 VIDEO SUMMARY:")
                    print(f"  Video: {video_name}")
                    print(f"  Duration: {segments[-1]['end_time']:.1f}s")
                    print(f"  Segments: {len(segments)}")
                    for phase, info in phase_summary.items():
                        print(f"  • {phase}: {info['duration']:.1f}s ({info['count']} segments)")
                    print()
                    continue
                
                elif cmd == 'phases':
                    print(f"\n🔍 DETECTED PHASES:")
                    for i, seg in enumerate(segments, 1):
                        print(f"  {i}. {seg['phase']} ({seg['start_time']:.1f}s - {seg['end_time']:.1f}s)")
                    print()
                    continue
                
                elif cmd == 'tools':
                    print(f"\n🔧 DETECTED TOOLS:")
                    all_tools = set()
                    for info in phase_summary.values():
                        all_tools.update(info['tools'])
                    for tool in sorted(all_tools):
                        print(f"  • {tool}")
                    print()
                    continue
                
                elif cmd.startswith('phase '):
                    # /phase <phase_name> - Show tools for specific phase
                    phase_name = cmd[6:].strip()
                    # Try to find matching phase (case-insensitive partial match)
                    matching_phases = [p for p in phase_summary.keys() if phase_name.lower() in p.lower()]
                    
                    if matching_phases:
                        for phase in matching_phases:
                            info = phase_summary[phase]
                            tools_str = ', '.join(sorted(info['tools'])) if info['tools'] else 'None detected'
                            print(f"\n🔍 PHASE: {phase}")
                            print(f"  Duration: {info['duration']:.1f}s ({info['count']} segments)")
                            print(f"  Tools: {tools_str}")
                        print()
                    else:
                        print(f"\n❌ Phase '{phase_name}' not found. Use /phases to see all phases.\n")
                    continue
                
                elif cmd == 'system':
                    # Show system information
                    if system_knowledge:
                        print(f"\n🤖 SYSTEM ARCHITECTURE:")
                        print(f"  Name: {system_knowledge.get('system_name', 'SAR-Podcast-Bot')}")
                        print(f"  Version: {system_knowledge.get('version', 'Unknown')}")
                        print(f"  Description: {system_knowledge.get('description', 'N/A')}")
                        
                        vision = system_knowledge.get('vision_pipeline', {})
                        print(f"\n📹 VISION PIPELINE: {vision.get('architecture', 'CNN + LSTM')}")
                        
                        cnn = vision.get('stage_1_cnn', {})
                        if cnn:
                            print(f"  CNN: {cnn.get('model_name', 'ToolCNN')}")
                            backbone = cnn.get('backbone', {})
                            print(f"    Backbone: {backbone.get('architecture', 'ResNet-50')} (pretrained on {backbone.get('pretrained', 'ImageNet')})")
                            print(f"    Feature dim: {backbone.get('feature_dimension', 2048)}")
                        
                        lstm = vision.get('stage_2_lstm', {})
                        if lstm:
                            print(f"  LSTM: {lstm.get('model_name', 'ActionLSTMWithAttention')}")
                            arch = lstm.get('architecture', {})
                            print(f"    Hidden dim: {arch.get('hidden_dimension', 128)}")
                            print(f"    Layers: {arch.get('num_layers', 2)}")
                            print(f"    Bidirectional: {arch.get('bidirectional', True)}")
                            print(f"    Attention: {lstm.get('attention_mechanism', {}).get('type', 'Yes')}")
                        
                        lms = system_knowledge.get('language_models', {})
                        print(f"\n💬 LANGUAGE MODELS:")
                        for lm_name, lm_info in lms.items():
                            print(f"  • {lm_info.get('name', lm_name)}: {lm_info.get('type', 'Unknown')}")
                        
                        caps = system_knowledge.get('system_capabilities', {})
                        print(f"\n✅ WHAT I CAN DO:")
                        for cap in caps.get('what_i_can_do', [])[:5]:
                            print(f"  • {cap}")
                        
                        print(f"\n❌ LIMITATIONS:")
                        for lim in caps.get('what_i_cannot_do', [])[:3]:
                            print(f"  • {lim}")
                    else:
                        print("\n❌ System knowledge base not loaded.\n")
                    print()
                    continue
                
                elif cmd == 'tts':
                    # Toggle TTS
                    enable_tts = not enable_tts
                    tts_status = "🔊 ON" if enable_tts else "🔇 OFF"
                    print(f"\nText-to-Speech: {tts_status}\n")
                    continue
                
                elif cmd == 'voice':
                    # Toggle voice input
                    enable_voice_input = not enable_voice_input
                    voice_status = "🎤 ON" if enable_voice_input else "⌨️  OFF"
                    print(f"\nVoice Input: {voice_status}\n")
                    if enable_voice_input:
                        print("💡 Speak your questions instead of typing!")
                    continue
                
                elif cmd == 'save':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = Path(f"results/final_results/qa_session_{timestamp}.json")
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    with open(filepath, 'w') as f:
                        json.dump({
                            'video': video_name,
                            'context': video_context,
                            'conversation': conversation
                        }, f, indent=2)
                    print(f"💾 Conversation saved to: {filepath}\n")
                    continue
                
                else:
                    print(f"Unknown command: {cmd}\n")
                    continue
            
            # Generate response with video context
            # Add context to question for better responses
            contextual_question = (
            "Use the CONTEXT to answer the QUESTION. "
            "Do NOT repeat the context. Answer clearly.\n\n"
            f"CONTEXT:\n{video_context}\n\nQUESTION:\n{question}\n\nANSWER:"
            )

            
            response = generator.generate_response(contextual_question, max_length=300, temperature=0.7)
            
            # Store in conversation
            conversation.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'response': response,
                'model': generator.model_type
            })
            
            print(f"\n🤖 Bot [{generator.model_type}]:")
            print(response)
            print()
            
            # Speak response if TTS is enabled
            if enable_tts:
                # Select voice based on model type (distinct voices for easy differentiation)
                voice_map = {
                    'dummy': 'Alex',        # Alex - male US voice (clear, neutral)
                    'core': 'Samantha',     # Samantha - female US voice (warm, natural)
                    'llama': 'Samantha',    # Samantha - same as core
                    'sota': 'Daniel'        # Daniel - male UK voice (British accent, professional)
                }
                voice_name = voice_map.get(generator.model_type, None)
                speak_text(response, voice_name=voice_name)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue
    
    # Offer to save conversation
    if conversation:
        save = input("\n💾 Save conversation? (y/n): ").strip().lower()
        if save == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"results/final_results/qa_session_{timestamp}.json")
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump({
                    'video': video_name,
                    'context': video_context,
                    'conversation': conversation
                }, f, indent=2)
            print(f"💾 Saved to: {filepath}")
    
    return conversation

def main():
    """Main pipeline execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process surgical video through CNN + LSTM pipeline")
    parser.add_argument('--video', type=str, help='Path to input video file (not needed if using --load-npz)')
    parser.add_argument('--load-npz', type=str, help='Load pre-computed NPZ file instead of processing video')
    parser.add_argument('--cnn-model', type=str, default='results/tool_results/tool_detection_model_best.pth',
                       help='Path to trained CNN checkpoint')
    parser.add_argument('--lstm-model', type=str, default='results/phase_results/best_lstm_attention_model.pth',
                       help='Path to trained LSTM checkpoint')
    parser.add_argument('--output', type=str, default='results/final_results/predictions.npz',
                       help='Output path for predictions')
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='Process every Nth frame (default: 1)')
    parser.add_argument('--window-size', type=int, default=16,
                       help='LSTM window size (number of frames)')
    parser.add_argument('--stride', type=int, default=8,
                       help='LSTM stride (frames between windows)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--skip-lstm', action='store_true',
                       help='Skip LSTM processing (CNN only)')
    
    # Interactive Q&A arguments
    parser.add_argument('--interactive-qa', action='store_true',
                       help='Start interactive Q&A session after processing')
    parser.add_argument('--model-type', type=str, default='core',
                       choices=['dummy', 'core', 'llama', 'sota'],
                       help='Language model to use for Q&A')
    parser.add_argument('--lm-model-path', type=str,
                       default='results/core_results/best_model',
                       help='Path to language model (for core/dummy)')
    parser.add_argument('--enable-tts', action='store_true',
                       help='Enable text-to-speech for bot responses (macOS only)')
    parser.add_argument('--enable-voice-input', action='store_true',
                       help='Enable voice input for questions (requires SpeechRecognition)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.load_npz and not args.video:
        parser.error("Either --video or --load-npz must be specified")
    
    print("\n" + "="*60)
    print("SAR-PODCAST-BOT PIPELINE")
    print("="*60)
    
    # Check if we should skip vision processing
    if args.load_npz:
        print("\n📂 Loading pre-computed results from NPZ file...")
        print(f"   File: {args.load_npz}")
        output_path = Path(args.load_npz)
        if not output_path.exists():
            print(f"❌ Error: NPZ file not found: {args.load_npz}")
            return # Exit if the specified NPZ file doesn't exist
        
        # Load the results to populate cnn_results and lstm_results for later summary/Q&A
        loaded_data = np.load(output_path, allow_pickle=True)
        cnn_results = {
            'frame_indices': loaded_data['frame_indices'],
            'timestamps': loaded_data['timestamps'],
            'tool_predictions': loaded_data['tool_predictions'],
            'tool_confidences': loaded_data['tool_confidences'],
            'phase_predictions': loaded_data['phase_predictions'],
            'phase_confidences': loaded_data['phase_confidences'],
            'features': loaded_data['features']
        }
        
        # Check if LSTM results are present in the loaded NPZ
        if 'lstm_actions' in loaded_data:
            cnn_results['lstm_actions'] = loaded_data['lstm_actions']
            cnn_results['lstm_confidences'] = loaded_data['lstm_confidences']
            # Note: lstm_windows is not saved in the current save logic, so we won't load it
            args.skip_lstm = False # Ensure summary reflects LSTM was processed
        else:
            args.skip_lstm = True # Ensure summary reflects LSTM was skipped
            
        video_name = output_path.stem.replace('_predictions', '').replace('predictions', 'video')
        print(f"   Loaded results for video: {video_name}")
        print(f"   Results will be used for Q&A and summary.")
        
    else:
        # Initialize models and process video
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Step 1: Process video through CNN
        print("\n[STEP 1] Processing video through CNN...")
        cnn_processor = VideoProcessor(
            model_path=args.cnn_model,
            device=args.device
        )
        
        cnn_results = cnn_processor.process_video(
            video_path=args.video,
            sample_rate=args.sample_rate
        )
        
        # Step 2: Process features through LSTM (if not skipped)
        if not args.skip_lstm:
            print("\n[STEP 2] Processing CNN features through LSTM...")
            action_predictor = ActionPredictor(
                model_path=args.lstm_model,
                num_actions=len(PHASES),
                device=args.device
            )
            
            lstm_results = action_predictor.predict_sequence(
                features=cnn_results['features'],
                window_size=args.window_size,
                stride=args.stride
            )
            
            # Aggregate to frame-level
            frame_actions, frame_action_conf = action_predictor.aggregate_predictions(
                lstm_results, 
                len(cnn_results['frame_indices'])
            )
            
            # Add to results
            cnn_results['lstm_actions'] = frame_actions
            cnn_results['lstm_confidences'] = frame_action_conf
            cnn_results['lstm_windows'] = lstm_results
        
        # Step 3: Save results
        print("\n[STEP 3] Saving results...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare arrays for saving
        frame_indices = np.array(cnn_results['frame_indices'])
        timestamps = np.array(cnn_results['timestamps'])
        tool_predictions = np.array(cnn_results['tool_predictions'], dtype=object)
        tool_confidences = np.array(cnn_results['tool_confidences'], dtype=object)
        phase_predictions = np.array(cnn_results['phase_predictions'])
        phase_confidences = np.array(cnn_results['phase_confidences'], dtype=object)
        
        if args.skip_lstm:
            np.savez_compressed(
                output_path,
                frame_indices=frame_indices,
                timestamps=timestamps,
                tool_predictions=tool_predictions,
                tool_confidences=tool_confidences,
                phase_predictions=phase_predictions,
                phase_confidences=phase_confidences,
                features=cnn_results['features']
            )
        else:
            np.savez_compressed(
                output_path,
                frame_indices=frame_indices,
                timestamps=timestamps,
                tool_predictions=tool_predictions,
                tool_confidences=tool_confidences,
                phase_predictions=phase_predictions,
                phase_confidences=phase_confidences,
                lstm_actions=np.array(cnn_results['lstm_actions']),
                lstm_confidences=np.array(cnn_results['lstm_confidences']),
                features=cnn_results['features']
            )
        
        print(f"Results saved to: {output_path}")
        video_name = Path(args.video).stem

    
    # Step 4: Interactive Q&A (optional)
    if args.interactive_qa:
        print("\n[STEP 4] Starting interactive Q&A session...")
        
        # Load the NPZ we just saved
        loader = VisionResultsLoader(str(output_path))
        segments = loader.get_phase_segments(min_segment_duration=2.0)
        
        print(f"Detected {len(segments)} phase segments")
        
        # Initialize narration generator with selected model
        print(f"Initializing {args.model_type} model...")
        
        # Determine model path based on model type
        if args.model_type == 'dummy':
            model_path = 'results/dummy_results/best_model'
        elif args.model_type == 'core' or args.model_type == 'llama':
            model_path = 'results/core_results/best_model'
        else:  # sota
            model_path = None
        
        generator = NarrationGenerator(
            model_path=model_path,
            device=args.device,
            model_type=args.model_type
        )
        
        # Start interactive Q&A session (video_name already set above)
        conversation = interactive_qa_session(generator, segments, video_name,
                                             enable_tts=args.enable_tts,
                                             enable_voice_input=args.enable_voice_input)
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Total frames processed: {len(cnn_results['frame_indices'])}")
    print(f"Video duration: {cnn_results['timestamps'][-1]:.2f}s")
    
    print(f"\n[CNN] Phase distribution:")
    from collections import Counter
    phase_counts = Counter(cnn_results['phase_predictions'])
    for phase, count in phase_counts.most_common():
        print(f"  {phase}: {count} frames ({count/len(cnn_results['phase_predictions'])*100:.1f}%)")
    
    if not args.skip_lstm:
        print(f"\n[LSTM] Action distribution:")
        action_counts = Counter(cnn_results['lstm_actions'])
        for action, count in action_counts.most_common():
            print(f"  {action}: {count} frames ({count/len(cnn_results['lstm_actions'])*100:.1f}%)")
        
        # Only show LSTM windows info if we just processed the video (not when loading from NPZ)
        if not args.load_npz and 'lstm_windows' in cnn_results:
            print(f"\nLSTM windows processed: {len(cnn_results['lstm_windows']['predictions'])}")
            print(f"Average confidence: {np.mean(cnn_results['lstm_windows']['confidences']):.3f}")
    
    print("="*60)
    if args.interactive_qa:
        print("\nPipeline complete! Q&A session ended.")
    else:
        print("\nPipeline complete! Use --interactive-qa to start Q&A session.")
    print("="*60)


if __name__ == "__main__":
    main()
