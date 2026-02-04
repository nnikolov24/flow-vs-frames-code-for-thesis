#x3d_flow.py
"""
X3D_FLOW

Flow-based action recognition model built on top of the X3D-S RGB backbone.

The model expects input tensors of shape [B, 2, T, H, W] containing optical
flow (u, v). A learnable 1x1x1 Conv3d adapter first maps these 2 channels
to 3 channels, producing a pseudo-RGB tensor [B, 3, T, H, W] that is then
processed by the X3D_RGB model (which wraps the X3D-S architecture from
PyTorchVideo with a 3-class classification head).

The adapter is initialized so that each output channel starts as
0.5 * u + 0.5 * v, but its weights are learned jointly with the rest of
the network during training.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Import the sibling builder that creates the standard RGB X3D model
sys.path.append(str(Path(__file__).resolve().parent))
from x3d_rgb import X3D_RGB


class X3D_FlowWrapper(nn.Module):
    """
    Wraps the RGB X3D with a learnable 1x1x1 adapter that maps flow (2ch: u,v)
    to 3 channels expected by the RGB model.
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # 2 -> 3 channel adapter; no bias needed
        self.adapter = nn.Conv3d(2, 3, kernel_size=1, bias=False)

        # Base RGB model (unchanged)
        self.base = X3D_RGB(num_classes=num_classes, pretrained=pretrained)
        self.net = self.base.net

        # Initialize adapter to copy (u,v) equally into each of the 3 channels:
        # Each of the 3 output channels gets 0.5*u + 0.5*v
        with torch.no_grad():
            w = torch.zeros_like(self.adapter.weight)  # [3,2,1,1,1]
            w[:, 0, 0, 0, 0] = 0.5  # u
            w[:, 1, 0, 0, 0] = 0.5  # v
            self.adapter.weight.copy_(w)

    def forward(self, x):
        x = self.adapter(x)     # [B,2,T,H,W] -> [B,3,T,H,W]
        return self.base(x)


def X3D_FLOW(num_classes: int, pretrained: bool = True):
    """
    Factory to match the previous usage: returns a nn.Module with .net.
    """
    return X3D_FlowWrapper(num_classes=num_classes, pretrained=pretrained)

