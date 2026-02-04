# models/x3d_rgb.py
"""
X3D_RGB
Wrapper around the X3D-S architecture from PyTorchVideo.
The model is loaded via:
    pytorchvideo.models.hub.x3d_s

PyTorchVideo documentation:
    https://pytorchvideo.readthedocs.io/
X3D model source:
    https://github.com/facebookresearch/pytorchvideo
"""
import torch
import torch.nn as nn

try:
    from pytorchvideo.models.hub import x3d_s  # X3D-S pretrained on Kinetics
except Exception as e:
    raise RuntimeError("pytorchvideo not installed or import failed") from e

class X3D_RGB(nn.Module):
    """
    X3D-S backbone, pretrained on Kinetics-400, with the final layer
    replaced to match my number of classes.
    Expects input [B, 3, T, H, W] normalized to ~Kinetics stats.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.net = x3d_s(pretrained=pretrained)
        in_dim = self.net.blocks[-1].proj.in_features
        self.net.blocks[-1].proj = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.net(x)
