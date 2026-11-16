"""
Spatial Attention Block (SAB) for CA-Net
From: "CA-Net: A Novel Cascaded Attention-Based Network for Multistage Glaucoma Classification"
(IEEE Transactions on Instrumentation and Measurement, 2023)

SAB emphasizes informative spatial regions (such as the optic disc and cup lesions)
by performing Cross-Channel Average Pooling (CCAP) followed by 1x1 Conv, LayerNorm, and Sigmoid.
"""

import torch
import torch.nn as nn

class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        # 1x1 conv on pooled spatial map (1 input channel -> 1 output channel)
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.ln = nn.GroupNorm(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Input: x (F_ch) of shape (B, C, H, W)
        Output: F_sp of shape (B, C, H, W)
        """
        # Cross-channel average pooling (CCAP) over C -> (B, 1, H, W)
        v_ccap = torch.mean(x, dim=1, keepdim=True)
        # 1x1 Conv + LN + Sigmoid
        spatial_att = self.sigmoid(self.ln(self.conv(v_ccap))) # (B, 1, H, W)
        # Element-wise multiplication
        f_sp = x * spatial_att
        return f_sp
