import torch
import torch.nn as nn
from torchvision.models import resnet50


class ToolCNN(nn.Module):
    """
      - predict tools   (multi-label, num_tools)
      - predict stages  (single-label, num_stages)

    Backbone:
      ResNet-50
    Heads:
      - tool_head:   Linear(2048 -> num_tools)
      - stage_head:  Linear(2048 -> num_stages)

    Training:
      - Tools:  use BCEWithLogitsLoss on tool_logits
      - Stages: use CrossEntropyLoss on stage_logits

    Inference:
      - Tools:  torch.sigmoid(tool_logits)
      - Stages: torch.softmax(stage_logits, dim=1)
    """

    def __init__(
        self,
        num_tools: int,
        num_stages: int,
        n_cnn_outputs: int = 2048,
        ini_fc: float = 0.01,
        ini_bias: float = 0.0,
        pretrained: bool = True,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        self.num_tools = num_tools
        self.num_stages = num_stages
        self.n_cnn_outputs = n_cnn_outputs

 
        self._full_resnet = resnet50(weights="IMAGENET1K_V2" if pretrained else None)


        backbone_modules = list(self._full_resnet.children())[:-1] 
        self.backbone = nn.Sequential(*backbone_modules)  


        in_feat = self._full_resnet.fc.in_features  
        if in_feat != n_cnn_outputs:
            raise ValueError(
                f"Backbone output dim {in_feat} != n_cnn_outputs={n_cnn_outputs}. "
                f"Set n_cnn_outputs={in_feat} or change backbone."
            )

        self.dropout = nn.Dropout(p=dropout_p)

        self.tool_head = nn.Linear(in_feat, num_tools)
        self.stage_head = nn.Linear(in_feat, num_stages)

        nn.init.uniform_(self.tool_head.weight, a=0.0, b=ini_fc)
        nn.init.constant_(self.tool_head.bias, ini_bias)

        nn.init.uniform_(self.stage_head.weight, a=0.0, b=ini_fc)
        nn.init.constant_(self.stage_head.bias, ini_bias)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, x, return_features: bool = False):
        """
        x: [B, 3, H, W] batch of RGB frames.

        returns:
            tool_logits:   [B, num_tools]
            stage_logits:  [B, num_stages]
            (optional) features: [B, n_cnn_outputs]
        """

        feats = self.backbone(x)

        feats = feats.view(feats.size(0), -1)

        feats = self.dropout(feats)

        tool_logits = self.tool_head(feats)
        stage_logits = self.stage_head(feats)

        if return_features:
            return tool_logits, stage_logits, feats

        return tool_logits, stage_logits


    def backbone_parameters(self):
        """Parameters of the ResNet backbone."""
        return self.backbone.parameters()

    def head_parameters(self):
        """Parameters of both heads (tools + stages)."""
        return list(self.tool_head.parameters()) + list(self.stage_head.parameters())

    def freeze_backbone(self):
        """Freeze ResNet weights"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze ResNet weights"""
        for p in self.backbone.parameters():
            p.requires_grad = True
