"""Grad-CAM explainability module for the Lung Disease Management System.

Provides :class:`GradCAMExplainer` which produces base64-encoded annotated
PNG visualisations of the model's attention on input spectrograms.

Typical usage::

    config = Config()
    model = LungDiseaseModel(metadata_input_dim=11)
    explainer = GradCAMExplainer(model, config)
    b64_png = explainer.explain(spectrogram_tensor)
"""

from __future__ import annotations

import base64
import io
import math
from typing import List, Optional

import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import Tensor

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import Config
from src.models.model import LungDiseaseModel


# ---------------------------------------------------------------------------
# Frequency-bin helpers
# ---------------------------------------------------------------------------


def _mel_bin_to_hz(bin_idx: int, n_mels: int, sample_rate: int = 4000) -> float:
    """Convert a mel filterbank bin index to approximate centre frequency in Hz.

    Uses the HTK mel scale formula::

        mel = 2595 * log10(1 + f / 700)
        f   = 700 * (10^(mel/2595) - 1)

    The bin index is mapped linearly across the mel scale from 0 Hz to
    ``sample_rate / 2`` Hz.

    Args:
        bin_idx: Zero-based mel bin index.
        n_mels: Total number of mel bins.
        sample_rate: Audio sample rate in Hz (default ``4000``).

    Returns:
        Centre frequency of the mel bin in Hz.
    """
    nyquist = sample_rate / 2.0
    f_min = 0.0
    f_max = nyquist

    # Convert f_min/f_max to mel
    mel_min = 2595.0 * math.log10(1.0 + f_min / 700.0) if f_min > 0 else 0.0
    mel_max = 2595.0 * math.log10(1.0 + f_max / 700.0)

    # Linear interpolation in mel space
    mel_centre = mel_min + (mel_max - mel_min) * (bin_idx / max(n_mels - 1, 1))

    # Convert back to Hz
    hz = 700.0 * (10.0 ** (mel_centre / 2595.0) - 1.0)
    return hz


def _classify_frequency_pattern(hz: float) -> Optional[str]:
    """Classify an activated frequency into a clinical annotation label.

    Rules (priority order):
        1. hz < 100            → "low-frequency artifact"
        2. 100 ≤ hz ≤ 1000    → "wheeze region"
        3. 200 < hz ≤ 2000    → "crackle burst"
        4. otherwise           → None (no annotation)

    The overlap between rules 2 and 3 (200–1000 Hz) is resolved by
    returning whichever rule was matched first (wheeze takes priority).

    Args:
        hz: Centre frequency of the mel bin in Hz.

    Returns:
        Annotation string or ``None`` if no pattern matches.
    """
    if hz < 100.0:
        return "low-frequency artifact"
    if 100.0 <= hz <= 1000.0:
        return "wheeze region"
    if 200.0 < hz <= 2000.0:
        return "crackle burst"
    return None


# ---------------------------------------------------------------------------
# GradCAMExplainer
# ---------------------------------------------------------------------------


class GradCAMExplainer:
    """Grad-CAM explainability wrapper for :class:`~src.models.model.LungDiseaseModel`.

    Produces base64-encoded annotated PNG images highlighting the spectrogram
    regions most influential for the model's prediction.

    Annotations are determined by the frequency range of the most activated
    mel bins:
        - Sustained activity 100–1000 Hz  → "wheeze region"
        - Short transients  200–2000 Hz   → "crackle burst"
        - Activity < 100 Hz               → "low-frequency artifact"
        - No defined pattern              → no annotation

    Args:
        model: The trained :class:`~src.models.model.LungDiseaseModel`.
        config: Project-wide :class:`~src.config.Config` instance.

    Example::

        explainer = GradCAMExplainer(model, config)
        b64_png = explainer.explain(spectrogram)  # (1, 3, 128, 216) tensor

    **Validates: Requirements 8.1–8.8**
    """

    def __init__(self, model: LungDiseaseModel, config: Config) -> None:
        self._model = model
        self._config = config

        # Use the last convolutional layer of the EfficientNetV2 backbone
        self._gradcam = GradCAM(
            model=model,
            target_layers=[model.conv_head],
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain(self, spectrogram: Tensor) -> str:
        """Run Grad-CAM on *spectrogram* and return a base64-encoded PNG.

        The method:
            1. Computes the Grad-CAM heatmap (H × W, values 0–1).
            2. Extracts the log-mel channel (channel 0) for the RGB overlay.
            3. Calls :meth:`_annotate_heatmap` to draw clinical annotations.
            4. Encodes the result as a base64 PNG string.

        Args:
            spectrogram: Input tensor of shape ``(1, 3, 128, 216)`` or
                ``(3, 128, 216)``.  Channel 0 is the log-mel spectrogram.

        Returns:
            Base64-encoded PNG string (UTF-8 decoded bytes).

        **Validates: Requirements 8.1, 8.2**
        """
        # Ensure 4-D input
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)  # (1, 3, H, W)

        spectrogram = spectrogram.to(
            next(self._model.parameters()).device
        )

        # Grad-CAM returns (1, H, W) grayscale heatmap
        grayscale_cam = self._gradcam(input_tensor=spectrogram)  # (1, H, W)
        heatmap = grayscale_cam[0]  # (H, W), values 0-1

        # Extract log-mel channel for RGB overlay (normalise to [0, 1])
        log_mel = spectrogram[0, 0].detach().cpu().numpy()  # (H, W)
        lo, hi = log_mel.min(), log_mel.max()
        if hi > lo:
            log_mel_norm = (log_mel - lo) / (hi - lo)
        else:
            log_mel_norm = np.zeros_like(log_mel)

        # Convert grayscale to RGB by repeating across 3 channels
        log_mel_rgb = np.stack([log_mel_norm] * 3, axis=-1).astype(np.float32)

        # Overlay heatmap on the log-mel RGB image
        cam_image = show_cam_on_image(log_mel_rgb, heatmap, use_rgb=True)

        # Annotate with clinical labels
        annotated_pil = self._annotate_heatmap(heatmap, log_mel_norm)

        # Encode to base64 PNG
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="PNG")
        b64_bytes = base64.b64encode(buffer.getvalue())
        return b64_bytes.decode("utf-8")

    def _detect_pattern(
        self,
        heatmap: np.ndarray,
        n_mels: int = 128,
    ) -> Optional[str]:
        """Detect the most prominent frequency pattern in the heatmap.

        Returns ``"wheeze region"``, ``"crackle burst"``,
        ``"low-frequency artifact"``, or ``None``.

        Logic:
            1. If ``max(heatmap) < 0.3`` → return ``None`` (no significant
               activation).
            2. Compute weighted centroid mel bin::

                centroid = Σ(row_idx × row_weight) / Σ(row_weight)

               where ``row_weight = sum of heatmap values in that row``.
            3. Convert centroid bin to Hz using
               ``librosa.mel_frequencies(n_mels=128, fmin=50, fmax=2000)``.
            4. Apply classification rules and return exactly one label.

        Args:
            heatmap: 2-D float array with values in ``[0, 1]``, shape
                ``(H, W)``.  ``H`` is treated as the mel-bin (frequency)
                dimension.
            n_mels: Number of mel filterbank bins (default ``128``).

        Returns:
            Annotation string or ``None`` when no significant activation is
            present.

        **Validates: Requirements 8.3–8.7**
        """
        if heatmap.max() < 0.3:
            return None

        # Row weights: sum of heatmap activations for each frequency bin
        row_weights = heatmap.sum(axis=1)  # shape (H,)
        total_weight = row_weights.sum()
        if total_weight <= 0.0:
            return None

        # Weighted centroid mel-bin index
        bin_indices = np.arange(heatmap.shape[0], dtype=np.float64)
        centroid_bin = float(np.dot(bin_indices, row_weights) / total_weight)

        # Convert centroid bin to Hz via librosa mel scale
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=50, fmax=2000)
        # Clip centroid_bin to valid index range
        idx = int(round(centroid_bin))
        idx = max(0, min(idx, len(mel_freqs) - 1))
        centroid_hz = float(mel_freqs[idx])

        # Apply classification rules
        if centroid_hz < 100.0:
            return "low-frequency artifact"
        if centroid_hz <= 1000.0:
            return "wheeze region"
        # 1000 < centroid_hz <= 2000
        return "crackle burst"

    def _annotate_heatmap(
        self,
        heatmap: np.ndarray,
        spectrogram: np.ndarray,
    ) -> Image.Image:
        """Apply frequency-range annotations to the Grad-CAM overlay.

        Finds the most activated mel-bin rows in *heatmap*, converts their
        index to Hz via :func:`_mel_bin_to_hz`, and draws text labels on the
        image for each distinct pattern found.  Each pattern appears at most
        once (no duplicates).

        Annotation rules (most prominent pattern per highlighted region only):
            - Sustained activity 100–1000 Hz → "wheeze region"
            - Short transients 200–2000 Hz   → "crackle burst"
            - Activity < 100 Hz              → "low-frequency artifact"
            - No defined pattern             → no annotation

        Args:
            heatmap: Grad-CAM activation map of shape ``(H, W)`` with values
                normalised to ``[0, 1]``.
            spectrogram: Normalised log-mel spectrogram of shape ``(H, W)``
                with values in ``[0, 1]``.

        Returns:
            Annotated PIL :class:`~PIL.Image.Image` (RGB).

        **Validates: Requirements 8.3–8.7**
        """
        h, w = heatmap.shape
        n_mels = h  # mel-bin dimension

        # Build a grayscale-to-RGB image of the spectrogram
        spec_norm = spectrogram.copy()
        lo, hi = spec_norm.min(), spec_norm.max()
        if hi > lo:
            spec_norm = (spec_norm - lo) / (hi - lo)

        spec_rgb = np.stack([spec_norm] * 3, axis=-1).astype(np.float32)

        # Overlay the Grad-CAM heatmap
        overlay = show_cam_on_image(spec_rgb, heatmap, use_rgb=True)
        pil_image = Image.fromarray(overlay)

        # Determine which annotations to draw
        # Aggregate activation per mel bin (mean over time axis)
        row_activation = heatmap.mean(axis=1)  # (H,)

        # Threshold: activate bins above 50% of the max row activation
        threshold = row_activation.max() * 0.5 if row_activation.max() > 0 else 0.0
        active_bins = np.where(row_activation >= threshold)[0]

        # Collect unique annotation labels (preserving insertion order)
        seen: set = set()
        annotations: List[str] = []
        for bin_idx in active_bins:
            hz = _mel_bin_to_hz(int(bin_idx), n_mels, self._config.target_sample_rate)
            label = _classify_frequency_pattern(hz)
            if label is not None and label not in seen:
                seen.add(label)
                annotations.append(label)

        # Draw annotations on the image
        if annotations:
            draw = ImageDraw.Draw(pil_image)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None  # type: ignore[assignment]

            y_offset = 4
            for text in annotations:
                # Semi-transparent background approximation via a filled rectangle
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                # Draw black background rectangle
                draw.rectangle(
                    [4, y_offset, 4 + text_w + 4, y_offset + text_h + 2],
                    fill=(0, 0, 0),
                )
                draw.text((6, y_offset + 1), text, fill=(255, 255, 255), font=font)
                y_offset += text_h + 6

        return pil_image
