"""Lung disease classification model.

This module defines :class:`LungDiseaseModel`, a multi-modal neural network
that fuses an EfficientNetV2B0 spectrogram backbone (augmented with CBAM
attention) with a metadata MLP branch, and exposes both deterministic forward
inference and Monte Carlo Dropout uncertainty estimation.

Typical usage::

    model = LungDiseaseModel(metadata_input_dim=6)
    probs = model(spectrogram, metadata)              # (B, 7)
    mean, std = model.predict_with_uncertainty(spectrogram, metadata, n_passes=20)
"""

from __future__ import annotations

from typing import Tuple

import timm
import torch
import torch.nn as nn
from torch import Tensor

from src.models.cbam import CBAM


class MetadataMLP(nn.Module):
    """Metadata MLP branch: ``input_dim → 64 → 32 → 16`` with ReLU activations.

    Before the tensor is passed through the linear layers, any NaN values are
    replaced with ``0.0`` (zero-fill for missing / null metadata fields).

    Args:
        input_dim: Dimensionality of the raw metadata feature vector.

    Example::

        mlp = MetadataMLP(input_dim=6)
        out = mlp(metadata_tensor)   # out.shape == (B, 16)
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Zero-fill NaN entries then pass through the MLP.

        Args:
            x: Raw metadata tensor of shape ``(B, input_dim)``. May contain
               ``float("nan")`` entries for missing fields.

        Returns:
            Encoded metadata vector of shape ``(B, 16)``.
        """
        x = torch.nan_to_num(x, nan=0.0)
        return self.net(x)


class LungDiseaseModel(nn.Module):
    """Multi-modal lung disease classifier.

    Architecture:
        - **Backbone**: EfficientNetV2B0 (ImageNet pre-trained by default),
          with the original classifier head replaced.
        - **CBAM**: Inserted after the final convolutional block (``conv_head``
          / ``bn2``), before global average pooling.  Operates on the 1280-channel
          feature maps produced by ``conv_head``.
        - **Metadata MLP**: Three-layer MLP
          ``(input_dim → 64 → 32 → 16)`` with ReLU activations; NaN values are
          zero-filled before the first layer.
        - **Fusion head**: ``concat(backbone_feat[1280], metadata_feat[16])``
          → ``Linear(1296, 128)`` → ReLU → ``Dropout(0.3)``
          → ``Linear(128, num_classes)`` → Softmax.

    Args:
        metadata_input_dim: Number of features in the metadata vector.
        num_classes: Number of output disease classes (default ``7``).
        _pretrained: Load ImageNet weights for the backbone. Set to ``False``
            during testing to avoid network requests (default ``True``).

    Raises:
        ValueError: If the spectrogram tensor passed to :meth:`forward` does
            not have exactly 3 input channels.

    Example::

        model = LungDiseaseModel(metadata_input_dim=6)
        probs = model(spectrogram, metadata)   # (1, 7)
    """

    def __init__(
        self,
        metadata_input_dim: int,
        num_classes: int = 7,
        _pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Backbone: EfficientNetV2B0
        # ------------------------------------------------------------------
        backbone = timm.create_model(
            "tf_efficientnetv2_b0",
            pretrained=_pretrained,
            features_only=False,
        )

        # Keep all feature-extraction layers; remove classifier head.
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.blocks = backbone.blocks
        self.conv_head = backbone.conv_head
        self.bn2 = backbone.bn2
        # global_pool + classifier intentionally NOT kept → replaced below.

        # ------------------------------------------------------------------
        # CBAM — applied after conv_head/bn2 (1280 channels)
        # ------------------------------------------------------------------
        self.cbam = CBAM(channels=1280, reduction_ratio=16)

        # ------------------------------------------------------------------
        # Global average pooling (replaces SelectAdaptivePool2d)
        # ------------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ------------------------------------------------------------------
        # Metadata MLP branch
        # ------------------------------------------------------------------
        self.metadata_mlp = MetadataMLP(input_dim=metadata_input_dim)

        # ------------------------------------------------------------------
        # Fusion classification head
        # ------------------------------------------------------------------
        backbone_dim = 1280
        metadata_dim = 16
        fusion_dim = backbone_dim + metadata_dim  # 1296

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _zero_fill_metadata(self, metadata: Tensor) -> Tensor:
        """Replace NaN values in a metadata tensor with ``0.0``.

        Args:
            metadata: Input tensor of shape ``(B, D)`` potentially containing
                ``float("nan")`` entries.

        Returns:
            Tensor of the same shape with all NaN replaced by ``0.0``.
        """
        return torch.nan_to_num(metadata, nan=0.0)

    def _extract_backbone_features(self, x: Tensor) -> Tensor:
        """Run the backbone stem + blocks + head, apply CBAM, then pool.

        Args:
            x: Input spectrogram tensor of shape ``(B, 3, H, W)``.

        Returns:
            Flattened feature vector of shape ``(B, 1280)``.
        """
        x = self.conv_stem(x)
        x = self.bn1(x)
        for block_group in self.blocks:
            x = block_group(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        # CBAM attention on 1280-channel feature maps
        x = self.cbam(x)
        # Global average pool → (B, 1280, 1, 1) → (B, 1280)
        x = self.global_pool(x)
        x = x.flatten(1)
        return x

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, spectrogram: Tensor, metadata: Tensor) -> Tensor:
        """Run a single deterministic forward pass.

        Args:
            spectrogram: Batch of log-mel spectrogram tensors of shape
                ``(B, 3, H, W)``.  The channel dimension **must** be exactly 3
                (log-mel, delta, delta-delta).
            metadata: Patient metadata tensor of shape ``(B, metadata_input_dim)``.
                May contain ``float("nan")`` for missing fields.

        Returns:
            Class probability tensor of shape ``(B, num_classes)`` where each
            row sums to ``1.0`` (output of Softmax).

        Raises:
            ValueError: If ``spectrogram.shape[1] != 3``.
        """
        if spectrogram.shape[1] != 3:
            raise ValueError(
                f"Expected 3 input channels, got {spectrogram.shape[1]}"
            )

        backbone_feat = self._extract_backbone_features(spectrogram)
        metadata_feat = self.metadata_mlp(metadata)

        fused = torch.cat([backbone_feat, metadata_feat], dim=1)
        return self.classifier(fused)

    def predict_with_uncertainty(
        self,
        spectrogram: Tensor,
        metadata: Tensor,
        n_passes: int = 20,
    ) -> Tuple[Tensor, Tensor]:
        """Estimate class probabilities and epistemic uncertainty via MC Dropout.

        Sets the model to training mode for all passes so that dropout layers
        remain stochastic, then restores the original mode afterwards.

        Args:
            spectrogram: Spectrogram tensor of shape ``(1, 3, H, W)`` or
                ``(B, 3, H, W)``.
            metadata: Metadata tensor of shape ``(1, metadata_input_dim)`` or
                ``(B, metadata_input_dim)``.
            n_passes: Number of stochastic forward passes (default ``20``).

        Returns:
            A tuple ``(mean, std)`` where both tensors have shape
            ``(num_classes,)`` (single-sample) or ``(B, num_classes)``
            (batch).

            - ``mean``: Per-class mean probability across all MC passes.
              Satisfies the softmax invariant (all values ∈ [0, 1], sum ≈ 1).
            - ``std``: Per-class standard deviation across all MC passes
              (all values ≥ 0).
        """
        was_training = self.training

        # Run stochastic forward passes with dropout active
        self.train()
        passes: list[Tensor] = []
        with torch.no_grad():
            for _ in range(n_passes):
                probs = self.forward(spectrogram, metadata)  # (B, C)
                passes.append(probs)

        # Stack to (n_passes, B, C) then compute statistics over pass dim
        stacked = torch.stack(passes, dim=0)          # (n_passes, B, C)
        mean = stacked.mean(dim=0).squeeze(0)         # (C,) or (B, C)
        std = stacked.std(dim=0).squeeze(0)           # (C,) or (B, C)

        # Restore original training/eval mode
        if was_training:
            self.train()
        else:
            self.eval()

        return mean, std
