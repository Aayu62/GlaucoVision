"""
CA-Net Complete Architecture Implementation
From: "CA-Net: A Novel Cascaded Attention-Based Network for Multistage Glaucoma Classification"
(IEEE Transactions on Instrumentation and Measurement, 2023)

Supports:
- CA-Net (DenseNet-121 + CAM + Classifier)
- Baseline (DenseNet-121 without attention)
- Ablation models (+TCAB only, +SAB only)
- Alternative backbones (ResNet-50, VGG-19, MobileNetV2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .tcab import TCAB
from .sab import SAB
from .cam import CAM

class CANet(nn.Module):
    def __init__(self,
                 backbone="densenet121",
                 num_classes=3,
                 pretrained=True,
                 attention="cam",
                 reduction_ratio=16):
        """
        Args:
            backbone (str): 'densenet121', 'resnet50', 'vgg19', 'mobilenet_v2'
            num_classes (int): 3 for multistage glaucoma (Normal, Early, Advanced), 2 for binary
            pretrained (bool): Whether to use ImageNet pretrained backbone weights
            attention (str): 'cam' (full CA-Net), 'tcab', 'sab', or 'none' (baseline)
            reduction_ratio (int): TCAB reduction ratio (default 16)
        """
        super(CANet, self).__init__()
        self.backbone_name = backbone.lower()
        self.attention_type = attention.lower()
        self.num_classes = num_classes

        # 1. Backbone Feature Extractor
        if self.backbone_name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            base = models.densenet121(weights=weights)
            self.features = base.features
            in_channels = 1024
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            # Remove avgpool and fc
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            in_channels = 2048
        elif self.backbone_name == "vgg19":
            weights = models.VGG19_Weights.DEFAULT if pretrained else None
            base = models.vgg19(weights=weights)
            self.features = base.features
            in_channels = 512
        elif self.backbone_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v2(weights=weights)
            self.features = base.features
            in_channels = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 2. Attention Module
        if self.attention_type == "cam":
            self.attention = CAM(channels=in_channels, reduction_ratio=reduction_ratio)
        elif self.attention_type == "tcab":
            self.attention = TCAB(channels=in_channels, reduction_ratio=reduction_ratio)
        elif self.attention_type == "sab":
            self.attention = SAB()
        elif self.attention_type == "none":
            self.attention = None
        else:
            raise ValueError(f"Unknown attention type: {attention}")

        # 3. Classifier Head (GAP + FC)
        self.classifier = nn.Linear(in_channels, num_classes)

        # Hook storage for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        """
        Forward pass.
        Returns logits (before softmax) for training with CrossEntropyLoss.
        """
        # Backbone feature extraction
        feat = self.features(x)
        if self.backbone_name == "densenet121":
            feat = F.relu(feat, inplace=True)

        # Apply Attention
        if self.attention is not None:
            feat = self.attention(feat)

        # Retain gradients for Grad-CAM hook if during backward pass
        if feat.requires_grad:
            feat.register_hook(self.activations_hook)
        self.activations = feat

        # Global Average Pooling (GAP)
        gap = torch.mean(feat, dim=(2, 3)) # (B, in_channels)

        # Fully Connected Classifier
        logits = self.classifier(gap)
        return logits

    def get_cam_activations(self):
        return self.activations

    def get_cam_gradients(self):
        return self.gradients
