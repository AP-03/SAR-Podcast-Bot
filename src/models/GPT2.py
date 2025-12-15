# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT library not available. Install with: pip install peft")
    print("   Running without LoRA adaptation.")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = tokenizer.eos_token_id

# Apply LoRA if available
if PEFT_AVAILABLE:
    print("✓ Applying LoRA (Low-Rank Adaptation)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  
        lora_alpha=32,  
        lora_dropout=0.1,  
        target_modules=["c_attn", "c_proj"],  
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}")
else:
    model = base_model
    print("  Using full fine-tuning (all parameters trainable)")

# Move model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on: {device}")