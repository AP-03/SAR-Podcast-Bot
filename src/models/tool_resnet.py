import torch
import torch.nn as nn
from torchvision.models import resnet50

class ToolCNN(nn.Module):
    def __init__(self, num_tools=7, pretrained=True):
        super().__init__()
        self.backbone = resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove original classifier
        self.tool_head = nn.Linear(in_feat, num_tools)

    def forward(self, x, return_features=False):
        """
        x: [B, 3, H, W]
        returns:
          logits:   [B, num_tools]
          probs:    [B, num_tools]
          (optional) features: [B, feature_dim]
        """
        feats = self.backbone(x)                  # [B, 2048]
        logits = self.tool_head(feats)            # [B, num_tools]
        probs  = logits.sigmoid()                 # for inference

        if return_features:
            return logits, probs, feats
        return logits, probs
