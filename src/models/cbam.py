"""Convolutional Block Attention Module (CBAM).

This module implements the CBAM attention mechanism as described in:
    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

Typical usage::

    cbam = CBAM(channels=1280, reduction_ratio=16)
    out = cbam(feature_map)   # shape preserved: (B, C, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ChannelAttention(nn.Module):
    """Channel attention sub-module using a shared squeeze-and-excitation MLP.

    Args:
        channels: Number of input feature channels.
        reduction_ratio: Bottleneck reduction factor for the shared MLP.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        reduced = max(1, channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply channel attention and return rescaled feature map.

        Args:
            x: Input feature map of shape ``(B, C, H, W)``.

        Returns:
            Channel-attention-scaled feature map of shape ``(B, C, H, W)``.
        """
        B, C, _, _ = x.shape

        # Global average pool and global max pool → (B, C)
        avg_pool = x.mean(dim=[2, 3])
        max_pool = x.amax(dim=[2, 3])

        # Shared MLP applied to each descriptor
        avg_out = self.shared_mlp(avg_pool)   # (B, C)
        max_out = self.shared_mlp(max_pool)   # (B, C)

        # Element-wise sum → sigmoid → (B, C, 1, 1)
        scale = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention sub-module using channel-wise pooling and a 7×7 conv.

    Args:
        kernel_size: Convolution kernel size (default 7, padding 3 to preserve size).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spatial attention and return rescaled feature map.

        Args:
            x: Input feature map of shape ``(B, C, H, W)``.

        Returns:
            Spatial-attention-scaled feature map of shape ``(B, C, H, W)``.
        """
        # Channel-wise avg and max → (B, 1, H, W) each
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.amax(dim=1, keepdim=True)

        # Concatenate along channel dim → (B, 2, H, W)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Conv → sigmoid → (B, 1, H, W)
        scale = torch.sigmoid(self.conv(pooled))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Applies channel attention (squeeze-and-excitation) followed by spatial
    attention to an input feature map.

    Args:
        channels: Number of input channels.
        reduction_ratio: Channel reduction ratio for the squeeze-and-excitation
            MLP (default 16).

    Example::

        cbam = CBAM(channels=1280, reduction_ratio=16)
        y = cbam(x)  # y.shape == x.shape
    """

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: Tensor) -> Tensor:
        """Apply CBAM (channel then spatial attention) to the feature map.

        Args:
            x: Input feature map of shape ``(B, C, H, W)``.

        Returns:
            Attention-refined feature map of the same shape ``(B, C, H, W)``.
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
