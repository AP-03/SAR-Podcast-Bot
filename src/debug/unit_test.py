import torch
from models.tool_cnn import ToolCNN

if __name__ == "__main__":
    model = ToolCNN(num_tools=7, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, probs, feats = model(x, return_features=True)
    print("logits:", logits.shape)  # [2, 7]
    print("probs:", probs.shape)    # [2, 7]
    print("feats:", feats.shape)    # [2, 2048] for ResNet-50
