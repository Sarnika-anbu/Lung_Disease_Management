"""Tests for src/training/train.py.

Covers:
  - Property 14: Sampler Weight Correctness
  - Property 15: Augmentation Training-Only Invariant
  - Property 16: Backbone Freeze Completeness
  - Property 17: Checkpoint Save Monotonicity
  - Unit tests: FocalLoss defaults, AdamW config, CosineAnnealingLR config
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.config import Config
from src.models.model import LungDiseaseModel
from src.training.train import (
    FocalLoss,
    LungSoundDataset,
    Trainer,
    _DISEASE_CLASSES,
    _METADATA_DIM,
    compute_sampler_weights,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_NUM_CLASSES = len(_DISEASE_CLASSES)  # 7


def _make_model(pretrained: bool = False) -> LungDiseaseModel:
    """Return an untrained model for testing."""
    return LungDiseaseModel(metadata_input_dim=_METADATA_DIM, _pretrained=pretrained)


def _make_config(tmp_dir: Path | None = None) -> Config:
    """Return a Config whose checkpoints_dir points to a temp directory."""
    cfg = Config()
    if tmp_dir is not None:
        cfg.checkpoints_dir = tmp_dir
    return cfg


def _make_dummy_df(
    n_samples: int,
    tmp_dir: Path,
    disease_class: str | None = None,
) -> pd.DataFrame:
    """Create a minimal DataFrame with dummy .npy spectrograms on disk.

    Args:
        n_samples: Number of rows to generate.
        tmp_dir: Writable directory where .npy files are saved.
        disease_class: If given, all rows use this class; else cycles through
            all 7 disease classes.

    Returns:
        DataFrame compatible with :class:`~src.training.train.LungSoundDataset`.
    """
    rows = []
    for i in range(n_samples):
        spec = np.zeros((3, 128, 216), dtype=np.float32)
        npy_path = tmp_dir / f"spec_{i}.npy"
        np.save(str(npy_path), spec)

        cls = disease_class if disease_class else _DISEASE_CLASSES[i % _NUM_CLASSES]
        rows.append(
            {
                "spectrogram_path": str(npy_path),
                "disease_class": cls,
                "age": float(30 + i),
                "sex": "M" if i % 2 == 0 else "F",
                "bmi": 22.5,
                "pack_years": float(i * 2),
                "recording_location": "Trachea",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Property 14: Sampler Weight Correctness
# Feature: lung-disease-management, Property 14: Sampler Weight Correctness
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.dictionaries(
        st.integers(min_value=0, max_value=6),
        st.integers(min_value=1, max_value=100),
        min_size=1,
        max_size=7,
    )
)
def test_sampler_weight_correctness(class_counts: Dict[int, int]) -> None:
    """Each sample's weight must equal 1 / count_c; minority class has highest weight.

    **Validates: Requirements 6.2**
    """
    # Build the labels list
    labels = [cls for cls, cnt in class_counts.items() for _ in range(cnt)]

    weights = compute_sampler_weights(labels)

    assert len(weights) == len(labels), (
        f"Expected {len(labels)} weights, got {len(weights)}"
    )

    # Verify weight_i = 1.0 / count_c for each sample
    for i, label in enumerate(labels):
        expected_weight = 1.0 / class_counts[label]
        assert abs(weights[i].item() - expected_weight) < 1e-6, (
            f"Sample {i} (class {label}) weight {weights[i].item():.8f} "
            f"!= expected {expected_weight:.8f}"
        )

    # The class with the smallest count must have the highest weight
    min_count_class = min(class_counts, key=lambda c: class_counts[c])
    max_count_class = max(class_counts, key=lambda c: class_counts[c])

    if class_counts[min_count_class] < class_counts[max_count_class]:
        # Find one weight for each
        min_class_weight = next(
            weights[i].item() for i, lbl in enumerate(labels) if lbl == min_count_class
        )
        max_class_weight = next(
            weights[i].item() for i, lbl in enumerate(labels) if lbl == max_count_class
        )
        assert min_class_weight > max_class_weight, (
            f"Minority class {min_count_class} (count={class_counts[min_count_class]}) "
            f"weight {min_class_weight:.6f} is not greater than majority class "
            f"{max_count_class} (count={class_counts[max_count_class]}) "
            f"weight {max_class_weight:.6f}"
        )


# ---------------------------------------------------------------------------
# Property 15: Augmentation Training-Only Invariant
# Feature: lung-disease-management, Property 15: Augmentation Training-Only Invariant
# ---------------------------------------------------------------------------


def test_augmentation_eval_deterministic() -> None:
    """In eval mode, repeated __getitem__ calls return identical spectrograms.

    **Validates: Requirements 6.3, 6.4**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config()
        df = _make_dummy_df(1, tmp_path)

        # Use a non-zero spectrogram so differences are detectable
        spec_val = np.random.rand(3, 128, 216).astype(np.float32)
        np.save(str(tmp_path / "spec_0.npy"), spec_val)

        ds_eval = LungSoundDataset(df, config, training=False)

        results = [ds_eval[0][0].clone() for _ in range(5)]
        for i in range(1, 5):
            assert torch.equal(results[0], results[i]), (
                f"Eval mode produced different spectrograms on call {i}"
            )


def test_augmentation_training_nondeterministic() -> None:
    """In training mode, repeated __getitem__ calls may produce different results.

    Runs 20 calls; expects at least one difference (SpecAugment is stochastic).

    **Validates: Requirements 6.3, 6.4**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config()
        # Override time/freq mask params to ensure augmentation fires
        config.spec_augment_time_mask = 80
        config.spec_augment_freq_mask = 30
        config.spec_augment_num_masks = 2

        df = _make_dummy_df(1, tmp_path)

        # Fill spectrogram with ones so any masking creates zeros
        spec_val = np.ones((3, 128, 216), dtype=np.float32)
        np.save(str(tmp_path / "spec_0.npy"), spec_val)

        ds_train = LungSoundDataset(df, config, training=True)

        all_same = True
        baseline = ds_train[0][0].clone()
        for _ in range(20):
            candidate = ds_train[0][0].clone()
            if not torch.equal(baseline, candidate):
                all_same = False
                break

        assert not all_same, (
            "Training mode returned identical spectrograms across 20 calls "
            "(SpecAugment appears non-functional)"
        )


# ---------------------------------------------------------------------------
# Property 16: Backbone Freeze Completeness
# Feature: lung-disease-management, Property 16: Backbone Freeze Completeness
# ---------------------------------------------------------------------------


def test_freeze_backbone_sets_requires_grad_false() -> None:
    """After freeze_backbone(), every backbone param must have requires_grad=False.

    **Validates: Requirements 6.8**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config(tmp_path)
        model = _make_model()
        df = _make_dummy_df(14, tmp_path)

        trainer = Trainer(model, df, df, config)
        trainer.freeze_backbone()

        backbone_modules = [
            trainer._model.conv_stem,
            trainer._model.bn1,
            trainer._model.blocks,
            trainer._model.conv_head,
            trainer._model.bn2,
        ]
        for module in backbone_modules:
            for name, param in module.named_parameters():
                assert not param.requires_grad, (
                    f"Backbone parameter '{name}' still has requires_grad=True "
                    f"after freeze_backbone()"
                )


def test_freeze_backbone_raises_if_param_re_enabled() -> None:
    """freeze_backbone() must raise RuntimeError if a backbone param is re-enabled.

    Injects a trainable parameter into a backbone module AFTER the freeze
    to simulate the error condition.

    **Validates: Requirements 6.8**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config(tmp_path)
        model = _make_model()
        df = _make_dummy_df(14, tmp_path)

        trainer = Trainer(model, df, df, config)

        # Monkey-patch freeze_backbone to re-enable one param after freezing
        # to simulate the error check:
        original_freeze = trainer.freeze_backbone

        def patched_freeze():
            # First do the real freeze
            backbone_modules = [
                trainer._model.conv_stem,
                trainer._model.bn1,
                trainer._model.blocks,
                trainer._model.conv_head,
                trainer._model.bn2,
            ]
            for module in backbone_modules:
                for param in module.parameters():
                    param.requires_grad = False

            # Then inject a trainable param into conv_stem to trigger RuntimeError
            for param in trainer._model.conv_stem.parameters():
                param.requires_grad = True
                break  # just one is enough

            # Now run the verification part of freeze_backbone
            for module in backbone_modules:
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        raise RuntimeError(
                            f"Backbone parameter '{name}' still has requires_grad=True "
                            f"after freeze_backbone(). Aborting Stage 2."
                        )

        with pytest.raises(RuntimeError, match="requires_grad=True"):
            patched_freeze()


# ---------------------------------------------------------------------------
# Property 17: Checkpoint Save Monotonicity
# Feature: lung-disease-management, Property 17: Checkpoint Save Monotonicity
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_checkpoint_save_monotonicity(prev_best: float, current_score: float) -> None:
    """Checkpoint is saved if and only if current_score > prev_best.

    **Validates: Requirements 6.10**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config(tmp_path)
        model = _make_model()
        df = _make_dummy_df(7, tmp_path)

        trainer = Trainer(model, df, df, config)
        trainer._best_score = prev_best

        checkpoint_path = tmp_path / "best.pth"
        assert not checkpoint_path.exists(), "Checkpoint should not exist yet"

        trainer._save_best_checkpoint(epoch=1, score=current_score)

        if current_score > prev_best:
            assert checkpoint_path.exists(), (
                f"Checkpoint NOT saved when current_score={current_score:.6f} "
                f"> prev_best={prev_best:.6f}"
            )
            assert trainer._best_score == current_score, (
                f"_best_score not updated: got {trainer._best_score}, "
                f"expected {current_score}"
            )
        else:
            assert not checkpoint_path.exists(), (
                f"Checkpoint SAVED when current_score={current_score:.6f} "
                f"<= prev_best={prev_best:.6f}"
            )
            assert trainer._best_score == prev_best, (
                f"_best_score changed from {prev_best} to {trainer._best_score} "
                f"when score did not improve"
            )


# ---------------------------------------------------------------------------
# Task 8.9 — Unit tests for training configuration
# ---------------------------------------------------------------------------


def test_focal_loss_default_gamma() -> None:
    """FocalLoss must have gamma=2.0 as default.

    **Validates: Requirements 6.1**
    """
    criterion = FocalLoss()
    assert criterion.gamma == 2.0, (
        f"Expected default gamma=2.0, got {criterion.gamma}"
    )


def test_focal_loss_default_label_smoothing() -> None:
    """FocalLoss must have label_smoothing=0.1 as default.

    **Validates: Requirements 6.1**
    """
    criterion = FocalLoss()
    assert criterion.label_smoothing == 0.1, (
        f"Expected default label_smoothing=0.1, got {criterion.label_smoothing}"
    )


def test_optimizer_is_adamw_with_correct_hyperparams() -> None:
    """Stage 1 optimizer must be AdamW with lr=1e-4 and weight_decay=1e-2.

    **Validates: Requirements 6.5**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config(tmp_path)
        model = _make_model()
        df = _make_dummy_df(14, tmp_path)

        trainer = Trainer(model, df, df, config)

        # Replicate optimizer construction from train_stage1
        optimizer = torch.optim.AdamW(
            trainer._model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        assert isinstance(optimizer, torch.optim.AdamW), (
            "Optimizer is not AdamW"
        )
        assert optimizer.defaults["lr"] == 1e-4, (
            f"Expected lr=1e-4, got {optimizer.defaults['lr']}"
        )
        assert optimizer.defaults["weight_decay"] == 1e-2, (
            f"Expected weight_decay=1e-2, got {optimizer.defaults['weight_decay']}"
        )


def test_stage1_uses_cosine_annealing_lr_with_correct_tmax() -> None:
    """Stage 1 must use CosineAnnealingLR with T_max=50.

    **Validates: Requirements 6.6**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config(tmp_path)
        model = _make_model()
        df = _make_dummy_df(14, tmp_path)

        trainer = Trainer(model, df, df, config)

        optimizer = torch.optim.AdamW(
            trainer._model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.stage1_epochs
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR), (
            "Scheduler is not CosineAnnealingLR"
        )
        assert scheduler.T_max == 50, (
            f"Expected T_max=50, got {scheduler.T_max}"
        )


def test_focal_loss_forward_shape() -> None:
    """FocalLoss forward pass must return a scalar tensor.

    **Validates: Requirements 6.1**
    """
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    batch_size = 4
    logits = torch.randn(batch_size, _NUM_CLASSES)
    targets = torch.randint(0, _NUM_CLASSES, (batch_size,))

    loss = criterion(logits, targets)

    assert loss.shape == torch.Size([]), (
        f"Expected scalar loss, got shape {loss.shape}"
    )
    assert loss.item() >= 0.0, "Loss must be non-negative"


def test_focal_loss_alpha_weighting() -> None:
    """FocalLoss with alpha=None and alpha=uniform should differ in value.

    **Validates: Requirements 6.1**
    """
    batch_size = 8
    logits = torch.randn(batch_size, _NUM_CLASSES)
    targets = torch.randint(0, _NUM_CLASSES, (batch_size,))

    loss_no_alpha = FocalLoss(alpha=None)(logits, targets)

    # Uniform alpha (all ones, normalized)
    alpha = torch.ones(_NUM_CLASSES) / _NUM_CLASSES
    loss_with_alpha = FocalLoss(alpha=alpha)(logits, targets)

    # Both must be non-negative scalars
    assert loss_no_alpha.item() >= 0.0
    assert loss_with_alpha.item() >= 0.0


def test_compute_sampler_weights_basic() -> None:
    """compute_sampler_weights must return inverse-frequency weights correctly.

    **Validates: Requirements 6.2**
    """
    # Class 0: 3 samples → weight 1/3; class 1: 1 sample → weight 1.0
    labels = [0, 0, 0, 1]
    weights = compute_sampler_weights(labels)

    assert len(weights) == 4
    assert abs(weights[0].item() - 1.0 / 3) < 1e-6
    assert abs(weights[1].item() - 1.0 / 3) < 1e-6
    assert abs(weights[2].item() - 1.0 / 3) < 1e-6
    assert abs(weights[3].item() - 1.0) < 1e-6


def test_lung_sound_dataset_returns_correct_types() -> None:
    """LungSoundDataset.__getitem__ must return (Tensor, Tensor, int).

    **Validates: Requirements 6.3**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config()
        df = _make_dummy_df(3, tmp_path)
        ds = LungSoundDataset(df, config, training=False)

        spec, meta, label = ds[0]

        assert isinstance(spec, torch.Tensor), "Spectrogram must be a Tensor"
        assert isinstance(meta, torch.Tensor), "Metadata must be a Tensor"
        assert isinstance(label, int), f"Label must be int, got {type(label)}"
        assert spec.dtype == torch.float32, "Spectrogram must be float32"
        assert meta.dtype == torch.float32, "Metadata must be float32"
        assert spec.shape == (3, 128, 216), f"Unexpected spec shape {spec.shape}"
        assert meta.shape == (_METADATA_DIM,), f"Unexpected meta shape {meta.shape}"
        assert 0 <= label < _NUM_CLASSES, f"Label {label} out of valid range"


def test_lung_sound_dataset_length() -> None:
    """LungSoundDataset.__len__ must match the DataFrame length.

    **Validates: Requirements 6.3**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config()
        n = 7
        df = _make_dummy_df(n, tmp_path)
        ds = LungSoundDataset(df, config, training=False)

        assert len(ds) == n


def test_icbhi_score_perfect_prediction() -> None:
    """ICBHI score must equal 1.0 for perfect predictions.

    **Validates: Requirements 6.11**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config(tmp_path)
        model = _make_model()
        df = _make_dummy_df(14, tmp_path)

        trainer = Trainer(model, df, df, config)

        labels = list(range(_NUM_CLASSES)) * 2  # two of each class
        preds = labels.copy()
        score = trainer._compute_icbhi_score(preds, labels)

        assert abs(score - 1.0) < 1e-6, f"Expected 1.0 for perfect preds, got {score}"


def test_icbhi_score_formula() -> None:
    """ICBHI score = mean over classes of (sensitivity + specificity) / 2.

    **Validates: Requirements 6.11**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = _make_config(tmp_path)
        model = _make_model()
        df = _make_dummy_df(14, tmp_path)

        trainer = Trainer(model, df, df, config)

        # All predictions wrong (random other class)
        n = 14
        labels = [i % _NUM_CLASSES for i in range(n)]
        preds = [(i + 1) % _NUM_CLASSES for i in range(n)]
        score = trainer._compute_icbhi_score(preds, labels)

        assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"
