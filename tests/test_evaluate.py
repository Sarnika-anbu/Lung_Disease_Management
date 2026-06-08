"""Tests for src/training/evaluate.py.

Covers:
  - Property 18: ICBHI Score Formula Correctness
  - Property 19: Evaluation Full-Class Coverage
  - Unit tests: EvaluationReport structure, ECE range, Evaluator inference
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.models.model import LungDiseaseModel
from src.training.evaluate import (
    EvaluationReport,
    Evaluator,
    _compute_binary_icbhi,
    _compute_per_class_metrics,
)
from src.training.train import _DISEASE_CLASSES, _METADATA_DIM

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NUM_CLASSES = len(_DISEASE_CLASSES)  # 7


def _make_model() -> LungDiseaseModel:
    """Return a lightweight untrained model."""
    return LungDiseaseModel(metadata_input_dim=_METADATA_DIM, _pretrained=False)


def _make_config() -> Config:
    """Return a default Config."""
    return Config()


def _make_dummy_loader(
    n_samples: int = 14,
    num_classes: int = 7,
) -> DataLoader:
    """Build a DataLoader with random spectrograms and cycling class labels.

    Args:
        n_samples: Number of dummy samples.
        num_classes: Number of classes.

    Returns:
        DataLoader of ``(spec, meta, label)`` tuples.
    """
    specs = torch.zeros(n_samples, 3, 128, 216)
    metas = torch.zeros(n_samples, _METADATA_DIM)
    labels = torch.tensor([i % num_classes for i in range(n_samples)], dtype=torch.long)
    dataset = TensorDataset(specs, metas, labels)
    return DataLoader(dataset, batch_size=7, shuffle=False)


# ---------------------------------------------------------------------------
# Property 18: ICBHI Score Formula Correctness
# Feature: lung-disease-management, Property 18: ICBHI Score Formula Correctness
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    tp=st.integers(min_value=0, max_value=100),
    tn=st.integers(min_value=0, max_value=100),
    fp=st.integers(min_value=0, max_value=100),
    fn=st.integers(min_value=0, max_value=100),
)
def test_icbhi_score_formula(tp: int, tn: int, fp: int, fn: int) -> None:
    """_compute_binary_icbhi must match (sensitivity + specificity) / 2 exactly.

    Constructs preds/labels that produce the requested TP/TN/FP/FN counts
    for positive_class=0 (class 0 vs rest), then verifies the helper.

    **Validates: Requirements 7.1**
    """
    # Require valid denominators — skip degenerate cases
    assume((tp + fn) > 0 and (tn + fp) > 0)

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    expected = (sensitivity + specificity) / 2.0

    # Build preds/labels that produce the requested TP/TN/FP/FN for class 0:
    #   TP: pred=0, label=0  (tp times)
    #   FP: pred=0, label=1  (fp times — label is non-zero, e.g. 1)
    #   TN: pred=1, label=1  (tn times — both non-zero)
    #   FN: pred=1, label=0  (fn times)
    preds: List[int] = [0] * tp + [0] * fp + [1] * tn + [1] * fn
    labels: List[int] = [0] * tp + [1] * fp + [1] * tn + [0] * fn

    result = _compute_binary_icbhi(preds, labels, positive_class=0)
    assert abs(result - expected) < 1e-6, (
        f"Expected {expected:.8f} but got {result:.8f} "
        f"(tp={tp}, tn={tn}, fp={fp}, fn={fn})"
    )


# ---------------------------------------------------------------------------
# Property 19: Evaluation Full-Class Coverage
# Feature: lung-disease-management, Property 19: Evaluation Full-Class Coverage
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=6),  # true label
            st.integers(min_value=0, max_value=6),  # predicted label
        ),
        min_size=7,
        max_size=50,
    ).filter(lambda pairs: len(set(p[0] for p in pairs)) == 7)  # all 7 classes present
)
def test_evaluation_full_class_coverage(
    pairs: List[Tuple[int, int]],
) -> None:
    """_compute_per_class_metrics must always return exactly 7 entries per metric.

    Verifies that precision, recall, f1, and roc_auc each have exactly 7
    entries regardless of the distribution of predicted vs true labels.

    **Validates: Requirements 7.3, 7.4**
    """
    labels = [p[0] for p in pairs]
    preds = [p[1] for p in pairs]

    metrics = _compute_per_class_metrics(preds, labels)

    assert len(metrics["precision"]) == 7, (
        f"Expected 7 precision values, got {len(metrics['precision'])}"
    )
    assert len(metrics["recall"]) == 7, (
        f"Expected 7 recall values, got {len(metrics['recall'])}"
    )
    assert len(metrics["f1"]) == 7, (
        f"Expected 7 f1 values, got {len(metrics['f1'])}"
    )
    assert len(metrics["roc_auc"]) == 7, (
        f"Expected 7 roc_auc values, got {len(metrics['roc_auc'])}"
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_compute_binary_icbhi_perfect() -> None:
    """_compute_binary_icbhi must return 1.0 for perfect binary classification.

    **Validates: Requirements 7.1**
    """
    # All correct: TP only for class 0, TN only for others
    preds = [0, 0, 1, 1, 2, 2]
    labels = [0, 0, 1, 1, 2, 2]
    result = _compute_binary_icbhi(preds, labels, positive_class=0)
    assert abs(result - 1.0) < 1e-6, f"Expected 1.0, got {result}"


def test_compute_binary_icbhi_worst_case() -> None:
    """_compute_binary_icbhi handles all-wrong predictions gracefully.

    **Validates: Requirements 7.1**
    """
    # All class-0 labels predicted as class-1 → sensitivity=0
    # All class-1 labels predicted as class-0 → specificity=0
    preds = [1, 1, 0, 0]
    labels = [0, 0, 1, 1]
    result = _compute_binary_icbhi(preds, labels, positive_class=0)
    assert 0.0 <= result <= 1.0, f"Score out of [0,1]: {result}"


def test_compute_per_class_metrics_returns_correct_keys() -> None:
    """_compute_per_class_metrics must return dict with all required keys.

    **Validates: Requirements 7.3**
    """
    labels = list(range(7))  # one of each class
    preds = list(range(7))
    metrics = _compute_per_class_metrics(preds, labels)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics


def test_compute_per_class_metrics_perfect_predictions() -> None:
    """With perfect predictions all per-class precision, recall, f1 == 1.0.

    **Validates: Requirements 7.3, 7.4**
    """
    # 2 samples per class
    labels = [i for i in range(7)] * 2
    preds = labels[:]
    metrics = _compute_per_class_metrics(preds, labels)

    for i in range(7):
        assert abs(metrics["precision"][i] - 1.0) < 1e-6, (
            f"Class {i} precision={metrics['precision'][i]}"
        )
        assert abs(metrics["recall"][i] - 1.0) < 1e-6, (
            f"Class {i} recall={metrics['recall'][i]}"
        )
        assert abs(metrics["f1"][i] - 1.0) < 1e-6, (
            f"Class {i} f1={metrics['f1'][i]}"
        )


def test_evaluation_report_has_all_seven_classes() -> None:
    """EvaluationReport per-class dicts must contain exactly 7 entries.

    **Validates: Requirements 7.3**
    """
    model = _make_model()
    config = _make_config()
    loader = _make_dummy_loader()
    evaluator = Evaluator(model, loader, config)

    report = evaluator.compute_metrics()

    assert isinstance(report, EvaluationReport)
    assert len(report.per_class_precision) == 7
    assert len(report.per_class_recall) == 7
    assert len(report.per_class_f1) == 7
    assert len(report.per_class_roc_auc) == 7


def test_evaluation_report_icbhi_in_range() -> None:
    """EvaluationReport.icbhi_score must be in [0, 1].

    **Validates: Requirements 7.1**
    """
    model = _make_model()
    config = _make_config()
    loader = _make_dummy_loader()
    evaluator = Evaluator(model, loader, config)

    report = evaluator.compute_metrics()
    assert 0.0 <= report.icbhi_score <= 1.0, (
        f"ICBHI score {report.icbhi_score} out of [0, 1]"
    )


def test_evaluation_report_macro_f1_in_range() -> None:
    """EvaluationReport.macro_f1 must be in [0, 1].

    **Validates: Requirements 7.3**
    """
    model = _make_model()
    config = _make_config()
    loader = _make_dummy_loader()
    evaluator = Evaluator(model, loader, config)

    report = evaluator.compute_metrics()
    assert 0.0 <= report.macro_f1 <= 1.0, (
        f"Macro F1 {report.macro_f1} out of [0, 1]"
    )


def test_evaluation_report_ece_in_range() -> None:
    """EvaluationReport.ece must be in [0, 1].

    **Validates: Requirements 7.6**
    """
    model = _make_model()
    config = _make_config()
    loader = _make_dummy_loader()
    evaluator = Evaluator(model, loader, config)

    report = evaluator.compute_metrics()
    assert 0.0 <= report.ece <= 1.0, (
        f"ECE {report.ece} out of [0, 1]"
    )


def test_compute_icbhi_score_consistent_with_report() -> None:
    """compute_icbhi_score() and report.icbhi_score must agree.

    **Validates: Requirements 7.1**
    """
    model = _make_model()
    config = _make_config()
    loader = _make_dummy_loader()
    evaluator = Evaluator(model, loader, config)

    icbhi = evaluator.compute_icbhi_score()
    report = evaluator.compute_metrics()

    assert abs(icbhi - report.icbhi_score) < 1e-6, (
        f"compute_icbhi_score={icbhi} != report.icbhi_score={report.icbhi_score}"
    )


def test_plot_confusion_matrix_creates_file() -> None:
    """plot_confusion_matrix must create a PNG file at the given path.

    **Validates: Requirements 7.7**
    """
    model = _make_model()
    config = _make_config()
    loader = _make_dummy_loader()
    evaluator = Evaluator(model, loader, config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "cm.png"
        evaluator.plot_confusion_matrix(save_path)
        assert save_path.exists(), "Confusion matrix PNG was not created"
        assert save_path.stat().st_size > 0, "Confusion matrix PNG is empty"


def test_plot_reliability_diagram_creates_file() -> None:
    """plot_reliability_diagram must create a PNG file at the given path.

    **Validates: Requirements 7.8**
    """
    model = _make_model()
    config = _make_config()
    loader = _make_dummy_loader()
    evaluator = Evaluator(model, loader, config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "reliability.png"
        evaluator.plot_reliability_diagram(save_path)
        assert save_path.exists(), "Reliability diagram PNG was not created"
        assert save_path.stat().st_size > 0, "Reliability diagram PNG is empty"
