"""
Cascaded Attention Module (CAM) for CA-Net
From: "CA-Net: A Novel Cascaded Attention-Based Network for Multistage Glaucoma Classification"
(IEEE Transactions on Instrumentation and Measurement, 2023)

Combines TCAB and SAB in a cascaded sequence:
F_in -> TCAB -> F_ch -> SAB -> F_sp
"""

import torch.nn as nn
from .tcab import TCAB
from .sab import SAB

class CAM(nn.Module):
    def __init__(self, channels=1024, reduction_ratio=16):
        super(CAM, self).__init__()
        self.tcab = TCAB(channels=channels, reduction_ratio=reduction_ratio)
        self.sab = SAB()

    def forward(self, x):
        f_ch = self.tcab(x)
        f_sp = self.sab(f_ch)
        return f_sp
