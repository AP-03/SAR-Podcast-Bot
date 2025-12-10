"""
Dummy LSTM Language Model

A baseline LSTM-based language model for text generation.
Used as a comparison baseline against more sophisticated models like GPT-2.
"""

import torch
import torch.nn as nn


class DummyLSTM(nn.Module):
    """Dummy LSTM-based language model for causal text generation"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3, pad_token_id=None):
        """
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of token embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pad_token_id: Token ID for padding (optional)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass - compatible with transformers API
        
        Args:
            input_ids: [B, T] tensor of token IDs
            attention_mask: [B, T] attention mask (currently unused, for API compatibility)
            labels: [B, T] tensor of target token IDs (for training)
        
        Returns:
            Object with .loss and .logits attributes (like transformers models)
        """
        # Embed tokens
        embeds = self.embedding(input_ids)  # [B, T, embed_dim]
        
        # LSTM forward
        lstm_out, _ = self.lstm(embeds)  # [B, T, hidden_dim]
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary
        logits = self.fc(lstm_out)  # [B, T, vocab_size]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id if self.pad_token_id is not None else -100)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        # Return dict-like object to match transformers API
        return type('Outputs', (), {'loss': loss, 'logits': logits})()
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_p=0.9, eos_token_id=None):
        """
        Simple greedy/sampling generation
        
        Args:
            input_ids: [B, T] starting tokens
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            eos_token_id: Token ID to stop generation
        
        Returns:
            [B, max_length] generated token IDs
        """
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.size(0)
            device = input_ids.device
            
            # Start with input tokens
            generated = input_ids
            
            for _ in range(max_length - input_ids.size(1)):
                # Get logits for next token
                outputs = self.forward(generated)
                next_token_logits = outputs.logits[:, -1, :] / temperature  # [B, vocab_size]
                
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token generated for all sequences
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
            
            return generated
    
    def save_pretrained(self, save_directory):
        """Save model weights (compatible with transformers API)"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), save_path)
        
        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'pad_token_id': self.pad_token_id,
            'model_type': 'DummyLSTM'
        }
        config_path = os.path.join(save_directory, 'config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory):
        """Load model weights (compatible with transformers API)"""
        import os
        import json
        
        # Load config
        config_path = os.path.join(load_directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            pad_token_id=config.get('pad_token_id')
        )
        
        # Load weights
        weights_path = os.path.join(load_directory, 'pytorch_model.bin')
        model.load_state_dict(torch.load(weights_path))
        
        return model
