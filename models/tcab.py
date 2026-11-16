"""
Triplet Channel Attention Block (TCAB) for CA-Net
From: "CA-Net: A Novel Cascaded Attention-Based Network for Multistage Glaucoma Classification"
(IEEE Transactions on Instrumentation and Measurement, 2023)

TCAB consists of three branches:
1. 1D Channel Attention: captures interactions among different channels.
2. 2D Channel-Width Cross-Attention: captures dependencies along channel-width dimensions.
3. 2D Channel-Height Cross-Attention: captures dependencies along channel-height dimensions.
Followed by learnable weighted sum aggregation: F_ch = w1 * F_ch1 + w2 * F_ch2 + w3 * F_ch3
"""

import torch
import torch.nn as nn

class TCAB(nn.Module):
    def __init__(self, channels=1024, reduction_ratio=16):
        super(TCAB, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        reduced_channels = max(channels // reduction_ratio, 16)

        # Branch 1: 1D Channel Attention
        # GAP -> Conv 1x1 (K/r) -> LN -> ReLU -> Conv 1x1 (K) -> LN -> Sigmoid
        self.b1_conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.b1_ln1 = nn.GroupNorm(1, reduced_channels)
        self.b1_relu = nn.ReLU(inplace=True)
        self.b1_conv2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.b1_ln2 = nn.GroupNorm(1, channels)
        self.b1_sigmoid = nn.Sigmoid()

        # Branch 2: Channel-Width Cross Attention
        # Input rotated along W-axis -> CHAP over H -> Conv 1x1 (1 filter) -> LN -> Sigmoid
        self.b2_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.b2_ln = nn.GroupNorm(1, 1)
        self.b2_sigmoid = nn.Sigmoid()

        # Branch 3: Channel-Height Cross Attention
        # Input rotated along H-axis -> CWAP over W -> Conv 1x1 (1 filter) -> LN -> Sigmoid
        self.b3_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.b3_ln = nn.GroupNorm(1, 1)
        self.b3_sigmoid = nn.Sigmoid()

        # Learnable scalar aggregation weights w1, w2, w3 (initialized equally to 1/3)
        self.w1 = nn.Parameter(torch.tensor(1.0 / 3.0, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor(1.0 / 3.0, dtype=torch.float32))
        self.w3 = nn.Parameter(torch.tensor(1.0 / 3.0, dtype=torch.float32))

    def forward(self, x):
        """
        Input: x of shape (B, C, H, W)
        Output: F_ch of shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # ----- Branch 1: 1D Channel Attention -----
        # GAP over (H, W) -> (B, C, 1, 1)
        x_gap = torch.mean(x, dim=(2, 3), keepdim=True)
        w1_att = self.b1_conv1(x_gap)
        w1_att = self.b1_relu(self.b1_ln1(w1_att))
        w1_att = self.b1_sigmoid(self.b1_ln2(self.b1_conv2(w1_att)))
        f_ch1 = x * w1_att  # (B, C, H, W)

        # ----- Branch 2: Channel-Width Cross Attention -----
        # Rotate along W-axis: Permute (B, C, H, W) -> (B, H, C, W)
        # Here H acts as the channel dimension to pool over (CHAP)
        x_perm2 = x.permute(0, 2, 1, W_axis := 3)  # shape (B, H, C, W)
        # CHAP: Cross-height average pooling over H -> shape (B, 1, C, W)
        y_chap = torch.mean(x_perm2, dim=1, keepdim=True)
        w2_att = self.b2_sigmoid(self.b2_ln(self.b2_conv(y_chap))) # (B, 1, C, W)
        f_ch2_perm = x_perm2 * w2_att
        # Rotate back: (B, H, C, W) -> (B, C, H, W)
        f_ch2 = f_ch2_perm.permute(0, 2, 1, 3)

        # ----- Branch 3: Channel-Height Cross Attention -----
        # Rotate along H-axis: Permute (B, C, H, W) -> (B, W, H, C)
        # Here W acts as the channel dimension to pool over (CWAP)
        x_perm3 = x.permute(0, 3, 2, 1)  # shape (B, W, H, C)
        # CWAP: Cross-width average pooling over W -> shape (B, 1, H, C)
        z_cwap = torch.mean(x_perm3, dim=1, keepdim=True)
        w3_att = self.b3_sigmoid(self.b3_ln(self.b3_conv(z_cwap))) # (B, 1, H, C)
        f_ch3_perm = x_perm3 * w3_att
        # Rotate back: (B, W, H, C) -> (B, C, H, W)
        f_ch3 = f_ch3_perm.permute(0, 3, 2, 1)

        # ----- Aggregation: Learnable Weighted Sum -----
        f_ch = self.w1 * f_ch1 + self.w2 * f_ch2 + self.w3 * f_ch3
        return f_ch
