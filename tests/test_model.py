"""Tests for src/models/cbam.py and src/models/model.py.

Covers:
  - Property 10: Model Input Channel Validation
  - Property 11: Metadata Null Zero-Fill
  - Property 12: Softmax Output Invariant
  - Property 13: MC Dropout Output Shape and Non-Negativity
  - Unit tests: CBAM insertion, head replacement
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis import HealthCheck

from src.models.cbam import CBAM
from src.models.model import LungDiseaseModel

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

METADATA_DIM = 10  # small value to keep tests fast

# Module-level model instance shared across property tests to avoid the
# overhead of instantiating EfficientNetV2B0 on every Hypothesis example.
_SHARED_MODEL: LungDiseaseModel | None = None


def get_shared_model() -> LungDiseaseModel:
    """Return a lazily-initialised, eval-mode model (pretrained=False)."""
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        _SHARED_MODEL = LungDiseaseModel(
            metadata_input_dim=METADATA_DIM, _pretrained=False
        )
        _SHARED_MODEL.eval()
    return _SHARED_MODEL


def make_model() -> LungDiseaseModel:
    """Return an untrained (no downloaded weights) model instance."""
    m = LungDiseaseModel(metadata_input_dim=METADATA_DIM, _pretrained=False)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Property 10: Model Input Channel Validation
# Feature: lung-disease-management, Property 10: Model Input Channel Validation
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=1, max_value=10).filter(lambda c: c != 3))
def test_wrong_channel_count_raises_error(channels: int) -> None:
    """For any channel count ≠ 3, forward() must raise ValueError.

    **Validates: Requirements 5.1**
    """
    model = get_shared_model()
    spec = torch.randn(1, channels, 128, 216)
    meta = torch.randn(1, METADATA_DIM)
    with pytest.raises(ValueError, match=str(channels)):
        model(spec, meta)


@settings(max_examples=1)
@given(st.just(3))
def test_correct_channel_count_no_error(channels: int) -> None:
    """For channel count = 3, forward() must not raise any error.

    **Validates: Requirements 5.1**
    """
    model = get_shared_model()
    spec = torch.randn(1, channels, 128, 216)
    meta = torch.randn(1, METADATA_DIM)
    # Should not raise
    out = model(spec, meta)
    assert out.shape == (1, 7)


# ---------------------------------------------------------------------------
# Property 11: Metadata Null Zero-Fill
# Feature: lung-disease-management, Property 11: Metadata Null Zero-Fill
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.one_of(
            st.none(),
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-10,
                max_value=10,
            ),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_metadata_null_zero_fill(values: list) -> None:
    """NaN entries in metadata must be replaced by 0.0; MLP output length is 16.

    **Validates: Requirements 5.4**
    """
    # Build a float list with NaN at None positions
    float_values = [float("nan") if v is None else float(v) for v in values]

    dim = len(float_values)
    model = LungDiseaseModel(
        metadata_input_dim=dim, num_classes=7, _pretrained=False
    )
    model.eval()

    meta = torch.tensor([float_values], dtype=torch.float32)  # (1, dim)

    # _zero_fill_metadata must remove all NaNs
    filled = model._zero_fill_metadata(meta)
    assert not filled.isnan().any(), "NaN values remained after zero-fill"

    # MetadataMLP output must have length 16
    mlp_out = model.metadata_mlp(meta)
    assert mlp_out.shape == (1, 16), (
        f"MetadataMLP output shape {mlp_out.shape} is not (1, 16)"
    )
    assert not mlp_out.isnan().any(), "MLP output contains NaN"


# ---------------------------------------------------------------------------
# Property 12: Softmax Output Invariant
# Feature: lung-disease-management, Property 12: Softmax Output Invariant
# ---------------------------------------------------------------------------


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.just(None))
def test_softmax_output_invariant(dummy: None) -> None:
    """forward() must return shape (1, 7), all values in [0,1], sum ≈ 1.

    **Validates: Requirements 5.5**
    """
    model = get_shared_model()
    spec = torch.randn(1, 3, 128, 216)
    meta = torch.randn(1, METADATA_DIM)

    with torch.no_grad():
        out = model(spec, meta)

    assert out.shape == (1, 7), f"Expected shape (1, 7), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Values outside [0, 1]"
    assert abs(out.sum().item() - 1.0) < 1e-5, (
        f"Softmax sum {out.sum().item()} is not 1.0"
    )


# ---------------------------------------------------------------------------
# Property 13: MC Dropout Output Shape and Non-Negativity
# Feature: lung-disease-management, Property 13: MC Dropout Output Shape
# ---------------------------------------------------------------------------


@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.just(None))
def test_mc_dropout_output_shape(dummy: None) -> None:
    """predict_with_uncertainty must return (mean, std) both of shape (7,).

    All std values must be ≥ 0, and mean must satisfy the softmax invariant.

    **Validates: Requirements 5.6**
    """
    model = get_shared_model()
    spec = torch.randn(1, 3, 128, 216)
    meta = torch.randn(1, METADATA_DIM)

    mean, std = model.predict_with_uncertainty(spec, meta, n_passes=20)

    assert mean.shape == (7,), f"Expected mean shape (7,), got {mean.shape}"
    assert std.shape == (7,), f"Expected std shape (7,), got {std.shape}"
    assert (std >= 0).all(), "Some std values are negative"
    # Mean must satisfy softmax invariant
    assert (mean >= 0).all() and (mean <= 1).all(), "Mean values outside [0, 1]"
    assert abs(mean.sum().item() - 1.0) < 1e-4, (
        f"MC mean sum {mean.sum().item()} is not 1.0"
    )


# ---------------------------------------------------------------------------
# Unit Tests — Task 7.7
# ---------------------------------------------------------------------------


def test_cbam_is_in_model_modules() -> None:
    """CBAM must appear in the model's named_modules tree.

    Verifies Task 7.7 — CBAM insertion after last conv block.
    **Validates: Requirements 5.3**
    """
    model = get_shared_model()
    cbam_instances = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, CBAM)
    ]
    assert len(cbam_instances) > 0, "No CBAM instance found in model.named_modules()"


def test_original_efficientnet_classifier_head_replaced() -> None:
    """The original EfficientNetV2B0 head (Linear 1000-out) must not exist.

    Verifies Task 7.7 — head replacement.
    **Validates: Requirements 5.2**
    """
    model = get_shared_model()

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.out_features == 1000:
            pytest.fail(
                f"Original EfficientNet classifier head found at '{name}' "
                f"(out_features=1000). Head was not replaced."
            )


def test_fusion_head_output_dim() -> None:
    """The classifier head must output exactly num_classes logits (7 by default)."""
    model = get_shared_model()
    # Last Linear in the sequential classifier
    last_linear = None
    for mod in model.classifier.modules():
        if isinstance(mod, torch.nn.Linear):
            last_linear = mod
    assert last_linear is not None, "No Linear layer found in classifier"
    assert last_linear.out_features == 7, (
        f"Expected 7 output classes, got {last_linear.out_features}"
    )
