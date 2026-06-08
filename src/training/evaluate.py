"""Model evaluation for the Lung Disease Management System.

Provides :class:`Evaluator` for computing ICBHI score, macro-F1, per-class
precision/recall/F1/ROC-AUC, ECE, and generating visualisation artefacts.

Typical usage::

    config = Config()
    model = LungDiseaseModel(metadata_input_dim=11)
    evaluator = Evaluator(model, test_loader, config)
    report = evaluator.compute_metrics()
    print(report)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Force non-interactive backend before any pyplot import

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.config import Config
from src.models.model import LungDiseaseModel
from src.models.types import DiseaseClass
from src.training.train import _DISEASE_CLASSES, _METADATA_DIM


# ---------------------------------------------------------------------------
# Module-level helpers (exposed for property-based testing)
# ---------------------------------------------------------------------------


def _compute_binary_icbhi(
    preds: List[int],
    labels: List[int],
    positive_class: int,
) -> float:
    """Compute ICBHI score for a single class in a binary one-vs-rest fashion.

    For the chosen *positive_class*:
        sensitivity = TP / (TP + FN)  (recall)
        specificity = TN / (TN + FP)
        score       = (sensitivity + specificity) / 2

    Args:
        preds: Predicted integer class indices (same length as *labels*).
        labels: True integer class indices.
        positive_class: The class index treated as the positive class.

    Returns:
        ICBHI score for the given class in ``[0, 1]``.
    """
    tp = sum(1 for p, l in zip(preds, labels) if p == positive_class and l == positive_class)
    fn = sum(1 for p, l in zip(preds, labels) if p != positive_class and l == positive_class)
    fp = sum(1 for p, l in zip(preds, labels) if p == positive_class and l != positive_class)
    tn = sum(1 for p, l in zip(preds, labels) if p != positive_class and l != positive_class)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (sensitivity + specificity) / 2.0


def _compute_per_class_metrics(
    preds: List[int],
    labels: List[int],
) -> Dict[str, List[float]]:
    """Compute per-class precision, recall, F1, and ROC-AUC.

    All metrics are computed in a one-vs-rest manner for each of the
    7 :class:`~src.models.types.DiseaseClass` values.

    Args:
        preds: Predicted integer class indices.
        labels: True integer class indices.

    Returns:
        Dictionary with keys ``"precision"``, ``"recall"``, ``"f1"``, and
        ``"roc_auc"``; each value is a list of length 7 (one entry per class).

    Note:
        ROC-AUC is approximated from hard predictions using one-hot
        probability vectors. If a class is absent from *labels* the
        corresponding ROC-AUC entry is set to ``0.0``.
    """
    num_classes = 7
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)

    precision_vals = precision_score(
        labels_arr,
        preds_arr,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    ).tolist()

    recall_vals = recall_score(
        labels_arr,
        preds_arr,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    ).tolist()

    f1_vals = f1_score(
        labels_arr,
        preds_arr,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    ).tolist()

    # Build one-hot probability matrix from hard predictions for ROC-AUC
    n = len(labels)
    prob_matrix = np.zeros((n, num_classes), dtype=float)
    for i, pred in enumerate(preds):
        prob_matrix[i, pred] = 1.0

    # One-hot true labels
    true_onehot = np.zeros((n, num_classes), dtype=int)
    for i, lbl in enumerate(labels):
        true_onehot[i, lbl] = 1

    roc_auc_vals: List[float] = []
    for c in range(num_classes):
        # Need both positive and negative examples for this class
        if true_onehot[:, c].sum() == 0 or (1 - true_onehot[:, c]).sum() == 0:
            roc_auc_vals.append(0.0)
        else:
            try:
                auc = roc_auc_score(true_onehot[:, c], prob_matrix[:, c])
            except ValueError:
                auc = 0.0
            roc_auc_vals.append(float(auc))

    return {
        "precision": precision_vals,
        "recall": recall_vals,
        "f1": f1_vals,
        "roc_auc": roc_auc_vals,
    }


# ---------------------------------------------------------------------------
# EvaluationReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """Structured evaluation report produced by :class:`Evaluator`.

    Attributes:
        icbhi_score: Macro-averaged ICBHI score across all 7 classes.
        macro_f1: Macro-averaged F1 score from sklearn.
        per_class_precision: Mapping of class name to precision.
        per_class_recall: Mapping of class name to recall.
        per_class_f1: Mapping of class name to F1 score.
        per_class_roc_auc: Mapping of class name to one-vs-rest ROC-AUC.
        ece: Expected Calibration Error (10 equal-width bins).
    """

    icbhi_score: float
    macro_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_roc_auc: Dict[str, float]
    ece: float


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Evaluate a :class:`~src.models.model.LungDiseaseModel` on a held-out test set.

    Args:
        model: The trained model to evaluate.
        test_loader: DataLoader for the test split (no augmentation).
        config: Project-wide configuration instance.

    Example::

        evaluator = Evaluator(model, test_loader, config)
        report = evaluator.compute_metrics()
        evaluator.plot_confusion_matrix(config.outputs_dir / "cm.png")

    **Validates: Requirements 7.1–7.8**
    """

    def __init__(
        self,
        model: LungDiseaseModel,
        test_loader: DataLoader,
        config: Config,
    ) -> None:
        self._model = model
        self._test_loader = test_loader
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        # Cache predictions and probabilities after the first run
        self._preds: List[int] = []
        self._labels: List[int] = []
        self._probs: np.ndarray = np.empty(0)  # (N, num_classes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(self) -> None:
        """Run the model on the test set and cache preds, labels, and probs.

        Results are cached so subsequent calls do not re-run inference.
        """
        if self._preds:
            return  # Already computed

        self._model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[np.ndarray] = []

        with torch.no_grad():
            for specs, metas, labels in self._test_loader:
                specs = specs.to(self._device)
                metas = metas.to(self._device)

                probs = self._model(specs, metas)  # (B, num_classes)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.append(probs.cpu().numpy())

        self._preds = all_preds
        self._labels = all_labels
        self._probs = np.vstack(all_probs) if all_probs else np.empty((0, 7))

    def _compute_ece(self, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error using equal-width bins.

        Bins confidence (max probability) into *n_bins* bins over [0, 1].
        ECE = weighted average of |accuracy - mean confidence| per bin.

        Args:
            n_bins: Number of equal-width bins (default 10).

        Returns:
            ECE value in ``[0, 1]``.
        """
        self._run_inference()
        if len(self._preds) == 0:
            return 0.0

        confidences = np.max(self._probs, axis=1)  # (N,)
        accuracies = np.array(
            [1 if p == l else 0 for p, l in zip(self._preds, self._labels)],
            dtype=float,
        )

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_samples = len(confidences)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confidences >= lo) & (confidences < hi)
            if i == n_bins - 1:
                # Include right edge in last bin
                mask = (confidences >= lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / n_samples) * abs(bin_acc - bin_conf)

        return float(ece)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_icbhi_score(self) -> float:
        """Run model on test_loader and compute macro ICBHI score.

        For each of the 7 disease classes:
            icbhi_c = (sensitivity_c + specificity_c) / 2

        Returns the mean over all 7 classes.

        Returns:
            ICBHI score in ``[0, 1]``.

        **Validates: Requirements 7.1**
        """
        self._run_inference()
        num_classes = len(_DISEASE_CLASSES)
        scores: List[float] = []

        for c in range(num_classes):
            score = _compute_binary_icbhi(self._preds, self._labels, c)
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.0

    def compute_metrics(self) -> EvaluationReport:
        """Compute full evaluation metrics and return an :class:`EvaluationReport`.

        Metrics:
            - ICBHI score (macro average)
            - Macro F1 from sklearn
            - Per-class precision, recall, F1 (one entry per DiseaseClass)
            - Per-class ROC-AUC (one-vs-rest)
            - ECE (Expected Calibration Error, 10 bins)

        Returns:
            Fully populated :class:`EvaluationReport`.

        **Validates: Requirements 7.1–7.6**
        """
        self._run_inference()

        labels_arr = np.array(self._labels)
        preds_arr = np.array(self._preds)

        icbhi = self.compute_icbhi_score()

        macro_f1 = float(
            f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
        )

        per_class = _compute_per_class_metrics(self._preds, self._labels)
        class_names = _DISEASE_CLASSES  # ordered list of string values

        per_class_precision = {
            class_names[i]: per_class["precision"][i] for i in range(7)
        }
        per_class_recall = {
            class_names[i]: per_class["recall"][i] for i in range(7)
        }
        per_class_f1 = {
            class_names[i]: per_class["f1"][i] for i in range(7)
        }
        per_class_roc_auc = {
            class_names[i]: per_class["roc_auc"][i] for i in range(7)
        }

        ece = self._compute_ece()

        return EvaluationReport(
            icbhi_score=icbhi,
            macro_f1=macro_f1,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            per_class_roc_auc=per_class_roc_auc,
            ece=ece,
        )

    def plot_confusion_matrix(self, save_path: Path) -> None:
        """Plot and save a normalised confusion matrix as a PNG.

        Uses seaborn heatmap with annotation. The matrix is normalised
        row-wise (true labels) so each cell shows recall.

        Args:
            save_path: Destination path for the PNG file.

        **Validates: Requirements 7.7**
        """
        self._run_inference()

        cm = confusion_matrix(self._labels, self._preds, labels=list(range(7)))
        # Row-normalise: divide each row by its sum
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div-by-zero
        cm_norm = cm.astype(float) / row_sums

        fig, ax = plt.subplots(figsize=(9, 7))
        try:
            import seaborn as sns  # optional but listed in requirements
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                xticklabels=_DISEASE_CLASSES,
                yticklabels=_DISEASE_CLASSES,
                cmap="Blues",
                ax=ax,
            )
        except ImportError:
            # Fallback to pure matplotlib if seaborn not available
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(7))
            ax.set_yticks(range(7))
            ax.set_xticklabels(_DISEASE_CLASSES, rotation=45, ha="right")
            ax.set_yticklabels(_DISEASE_CLASSES)
            for i in range(7):
                for j in range(7):
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (row-normalised)")
        fig.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)

    def plot_reliability_diagram(self, save_path: Path) -> None:
        """Plot and save a reliability (calibration) diagram as a PNG.

        Shows mean confidence vs mean accuracy per equal-width bin over [0, 1].
        A perfectly calibrated model would lie on the diagonal.

        Args:
            save_path: Destination path for the PNG file.

        **Validates: Requirements 7.8**
        """
        self._run_inference()

        confidences = np.max(self._probs, axis=1) if len(self._probs.shape) == 2 else np.array([])
        accuracies = np.array(
            [1 if p == l else 0 for p, l in zip(self._preds, self._labels)],
            dtype=float,
        )

        n_bins = 10
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_confs: List[float] = []
        bin_accs: List[float] = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confidences >= lo) & (confidences < hi)
            if i == n_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            bin_confs.append(float(confidences[mask].mean()))
            bin_accs.append(float(accuracies[mask].mean()))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        if bin_confs:
            ax.scatter(bin_confs, bin_accs, s=80, zorder=5, label="Model")
            ax.plot(bin_confs, bin_accs, marker="o", label="_nolegend_")
        ax.set_xlabel("Mean Confidence")
        ax.set_ylabel("Mean Accuracy")
        ax.set_title("Reliability Diagram")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    config = Config()

    print("Lung Disease Management System — Model Evaluation")
    print("=" * 55)
    print("No test_loader provided in __main__ context.")
    print("To evaluate a model, instantiate Evaluator programmatically.")
    print()
    print("Example output format:")
    print(f"  ICBHI Score : <float>")
    print(f"  Macro F1    : <float>")
    print()
    print("  Per-class metrics:")
    header = f"  {'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cls in _DISEASE_CLASSES:
        print(f"  {cls:<18} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
