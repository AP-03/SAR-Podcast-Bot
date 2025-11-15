import torch
import torch.nn as nn


class ActionLSTM(nn.Module):
    """
    LSTM-based action classification model for detecting surgical movements.
    Takes sequential feature vectors from CNN (e.g., ResNet50) and processes them
    temporally to classify surgical actions.
    """
    
    def __init__(self, num_actions, feature_dim=2048, hidden_dim=512, num_layers=2, dropout=0.5, bidirectional=True):
        """
        Args:
            num_actions: Number of action classes to predict
            feature_dim: Dimension of input features from CNN (default: 2048 for ResNet50)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate for LSTM and final classifier
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output dimension based on bidirectional setting
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_actions)
        )
        
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.feature_dim = feature_dim
    
    def forward(self, features, return_sequence=False):
        """
        Forward pass for action classification.
        
        Args:
            features: Tensor of shape [B, T, feature_dim]
                B = batch size
                T = sequence length (number of time steps)
                feature_dim = dimension of feature vectors from CNN
            return_sequence: If True, return predictions for all time steps
                           If False, return only final prediction
        
        Returns:
            logits: [B, num_actions] or [B, T, num_actions] if return_sequence=True
            probs: [B, num_actions] or [B, T, num_actions] if return_sequence=True
        """
        # Process temporal sequence with LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        # lstm_out: [B, T, hidden_dim * num_directions]
        
        if return_sequence:
            # Return predictions for all time steps
            logits = self.classifier(lstm_out)  # [B, T, num_actions]
            probs = torch.softmax(logits, dim=-1)
        else:
            # Use only the final time step
            if self.bidirectional:
                # Concatenate final forward and backward hidden states
                final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                final_hidden = h_n[-1]
            
            logits = self.classifier(final_hidden)  # [B, num_actions]
            probs = torch.softmax(logits, dim=-1)
        
        return logits, probs
    
    def predict(self, features):
        """
        Convenient method for inference.
        
        Args:
            features: Tensor of shape [B, T, feature_dim]
        
        Returns:
            predictions: [B] tensor of predicted class indices
            probabilities: [B, num_actions] tensor of class probabilities
        """
        self.eval()
        with torch.no_grad():
            _, probs = self.forward(features, return_sequence=False)
            predictions = torch.argmax(probs, dim=1)
        return predictions, probs


'''class ActionLSTMWithAttention(ActionLSTM):
    """
    Enhanced LSTM model with attention mechanism for better
    temporal modeling of surgical actions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Attention mechanism
        lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
    
    def forward(self, features, return_sequence=False):
        """
        Forward pass with attention mechanism.
        
        Args:
            features: Tensor of shape [B, T, feature_dim]
            return_sequence: If True, return predictions for all time steps
        
        Returns:
            logits: [B, num_actions] or [B, T, num_actions]
            probs: [B, num_actions] or [B, T, num_actions]
            attention_weights: [B, T] attention weights over time (None if return_sequence=True)
        """
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # [B, T, hidden_dim]
        
        if return_sequence:
            # Return predictions for all time steps (without attention)
            logits = self.classifier(lstm_out)
            probs = torch.softmax(logits, dim=-1)
            return logits, probs, None
        else:
            # Apply attention mechanism
            attention_scores = self.attention(lstm_out)  # [B, T, 1]
            attention_weights = torch.softmax(attention_scores, dim=1)  # [B, T, 1]
            
            # Weighted sum of LSTM outputs
            context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # [B, hidden_dim]
            
            # Classification
            logits = self.classifier(context_vector)  # [B, num_actions]
            probs = torch.softmax(logits, dim=-1)
            
            return logits, probs, attention_weights.squeeze(-1)


# Example usage and helper functions
def create_action_lstm(
    num_actions,
    feature_dim=2048,
    model_type='standard',
    hidden_dim=512,
    num_layers=2,
    dropout=0.5,
    bidirectional=True
):
    """
    Factory function to create ActionLSTM models.
    
    Args:
        num_actions: Number of action classes
        feature_dim: Dimension of input features from CNN
        model_type: 'standard' or 'attention'
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Use bidirectional LSTM
    
    Returns:
        model: ActionLSTM or ActionLSTMWithAttention instance
    """
    if model_type == 'attention':
        model = ActionLSTMWithAttention(
            num_actions=num_actions,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    else:
        model = ActionLSTM(
            num_actions=num_actions,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Testing ActionLSTM implementation...")
    
    # Create model
    num_actions = 10  # e.g., 10 different surgical actions
    feature_dim = 2048  # ResNet50 feature dimension
    
    model = create_action_lstm(
        num_actions=num_actions,
        feature_dim=feature_dim,
        model_type='standard',
        hidden_dim=256,
        num_layers=2,
        bidirectional=True
    )
    
    # Test forward pass with feature vectors
    batch_size = 4
    seq_len = 16  # 16 time steps
    dummy_features = torch.randn(batch_size, seq_len, feature_dim)
    
    logits, probs = model(dummy_features)
    print(f"Input features shape: {dummy_features.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probs shape: {probs.shape}")
    
    # Test with sequence output
    print("\nTesting with sequence output...")
    logits_seq, probs_seq = model(dummy_features, return_sequence=True)
    print(f"Sequence logits shape: {logits_seq.shape}")
    print(f"Sequence probs shape: {probs_seq.shape}")
    
    # Test with attention model
    print("\nTesting ActionLSTM with Attention...")
    attention_model = create_action_lstm(
        num_actions=num_actions,
        feature_dim=feature_dim,
        model_type='attention',
        hidden_dim=256
    )
    
    logits, probs, attn_weights = attention_model(dummy_features)
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probs shape: {probs.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    print("\nModel created successfully!")'''
