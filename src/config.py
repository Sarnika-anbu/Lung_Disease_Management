"""Centralised configuration for the Lung Disease Management System.

All file paths in the project are defined here using ``pathlib.Path``.
No hardcoded path strings should appear outside this module.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Project-wide configuration constants.

    Attributes:
        raw_icbhi_dir: Directory for raw ICBHI 2017 dataset files.
        raw_arashnic_dir: Directory for raw Arashnic lung sounds dataset files.
        processed_dir: Root directory for preprocessed data artefacts.
        splits_dir: Directory for patient-independent train/test split CSVs.
        metadata_path: Path to the processed patient metadata CSV.
        checkpoints_dir: Directory for model checkpoint files.
        outputs_dir: Directory for evaluation output figures.
        target_sample_rate: Target audio sample rate in Hz.
        bandpass_low: Lower cutoff frequency for the bandpass filter (Hz).
        bandpass_high: Upper cutoff frequency for the bandpass filter (Hz).
        noise_gate_threshold: Spectral gating sensitivity threshold (0.0–1.0).
        segment_duration: Duration of each audio segment in seconds.
        segment_overlap: Fractional overlap between consecutive segments (0.0–1.0).
        n_mels: Number of mel filterbank bins for spectrogram computation.
        n_fft: FFT window size for spectrogram computation.
        hop_length: Hop length (in samples) for spectrogram computation.
        n_time_frames: Number of time frames in the fixed-size spectrogram output.
        batch_size: Mini-batch size for training and evaluation DataLoaders.
        stage1_epochs: Number of epochs for Stage 1 (full-model) training.
        stage2_epochs: Number of epochs for Stage 2 (head-only) fine-tuning.
        lr: Learning rate for the AdamW optimiser.
        weight_decay: Weight decay (L2 regularisation) for the AdamW optimiser.
        focal_gamma: Gamma focusing parameter for Focal Loss.
        focal_label_smoothing: Label smoothing epsilon for Focal Loss.
        mixup_alpha: Alpha parameter for the Beta distribution used in Mixup.
        spec_augment_time_mask: Maximum time-mask width for SpecAugment.
        spec_augment_freq_mask: Maximum frequency-mask width for SpecAugment.
        spec_augment_num_masks: Number of masks applied per channel in SpecAugment.
        mc_dropout_passes: Number of stochastic forward passes for MC Dropout.
    """

    # ---------------------------------------------------------------------------
    # Data paths
    # ---------------------------------------------------------------------------
    raw_icbhi_dir: Path = field(default_factory=lambda: Path("data/raw/icbhi"))
    raw_arashnic_dir: Path = field(default_factory=lambda: Path("data/raw/arashnic"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    splits_dir: Path = field(default_factory=lambda: Path("data/processed/splits"))
    metadata_path: Path = field(
        default_factory=lambda: Path("data/processed/metadata.csv")
    )
    checkpoints_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))

    # ---------------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------------
    target_sample_rate: int = 4000
    bandpass_low: float = 50.0
    bandpass_high: float = 2000.0
    noise_gate_threshold: float = 0.5
    segment_duration: float = 5.0
    segment_overlap: float = 0.5
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 128
    n_time_frames: int = 216

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    batch_size: int = 32
    stage1_epochs: int = 50
    stage2_epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-2
    focal_gamma: float = 2.0
    focal_label_smoothing: float = 0.1
    mixup_alpha: float = 0.4
    spec_augment_time_mask: int = 80
    spec_augment_freq_mask: int = 30
    spec_augment_num_masks: int = 2
    mc_dropout_passes: int = 20
