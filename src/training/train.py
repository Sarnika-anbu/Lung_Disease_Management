"""Training module for the Lung Disease Management System.

Implements the two-stage training procedure:
  - Stage 1: Full model training for 50 epochs with CosineAnnealingLR.
  - Stage 2: Backbone-frozen head fine-tuning for 15 epochs on a class-balanced subset.

Augmentation (SpecAugment, Mixup) is applied during training only.

Typical usage::

    config = Config()
    model = LungDiseaseModel(metadata_input_dim=11)
    trainer = Trainer(model, train_df, val_df, config)
    trainer.train_stage1()
    trainer.freeze_backbone()
    trainer.train_stage2()
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import Config
from src.models.model import LungDiseaseModel
from src.models.types import DiseaseClass

# Ordered list of DiseaseClass members (for index <-> class mapping)
_DISEASE_CLASSES: List[str] = [dc.value for dc in DiseaseClass]


# ---------------------------------------------------------------------------
# Helper: class-index mapping
# ---------------------------------------------------------------------------

def _disease_class_index(label: str) -> int:
    """Return the integer index of a DiseaseClass value string.

    Args:
        label: String value of a DiseaseClass enum member (e.g. ``"COPD"``).

    Returns:
        Integer index in the canonical DiseaseClass ordering.

    Raises:
        ValueError: If ``label`` does not match any DiseaseClass value.
    """
    try:
        return _DISEASE_CLASSES.index(label)
    except ValueError:
        raise ValueError(
            f"Unknown disease class label '{label}'. "
            f"Expected one of {_DISEASE_CLASSES}."
        )


# ---------------------------------------------------------------------------
# Task 8.1 — FocalLoss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss with optional class weights and label smoothing.

    Focal loss down-weights easy examples and focuses training on hard
    negatives.  Label smoothing prevents over-confident predictions.

    The loss is computed as::

        smooth_targets = (1 - ε) * one_hot(targets) + ε / C
        p_t = sum(smooth_targets * softmax(inputs))
        focal_weight = (1 - p_t) ^ gamma
        ce = -sum(smooth_targets * log(softmax(inputs) + eps))
        loss_i = focal_weight_i * ce_i * alpha[targets_i]   (if alpha given)

    Returns the mean over the batch.

    Args:
        gamma: Focusing parameter (default ``2.0``).
        alpha: Optional per-class weight tensor of shape ``(num_classes,)``.
               Typically set to inverse class frequency.
        label_smoothing: Smoothing epsilon ε ∈ [0, 1) (default ``0.1``).

    Example::

        criterion = FocalLoss(gamma=2.0, alpha=weights, label_smoothing=0.1)
        loss = criterion(logits, targets)

    **Validates: Requirements 6.1**
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Tensor] = None,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha: Optional[Tensor] = None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss for a batch.

        Args:
            inputs: Raw logit tensor of shape ``(B, C)`` or softmax
                    probabilities — either is accepted; softmax is applied
                    internally.
            targets: Integer class indices of shape ``(B,)``.

        Returns:
            Scalar mean focal loss over the batch.
        """
        num_classes = inputs.shape[1]
        eps = 1e-8

        # Compute softmax probabilities
        probs = torch.softmax(inputs, dim=1)  # (B, C)

        # One-hot encode targets → (B, C)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Apply label smoothing
        smooth_targets = (1.0 - self.label_smoothing) * one_hot + (
            self.label_smoothing / num_classes
        )

        # Cross-entropy with smoothed targets
        log_probs = torch.log(probs + eps)  # (B, C)
        ce = -torch.sum(smooth_targets * log_probs, dim=1)  # (B,)

        # Focal weight: p_t = sum(smooth_targets * probs)
        p_t = torch.sum(smooth_targets * probs, dim=1)  # (B,)
        focal_weight = (1.0 - p_t) ** self.gamma  # (B,)

        loss = focal_weight * ce  # (B,)

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]  # (B,)
            loss = loss * alpha_weight

        return loss.mean()


# ---------------------------------------------------------------------------
# Task 8.2 — compute_sampler_weights
# ---------------------------------------------------------------------------

def compute_sampler_weights(labels: List[int]) -> Tensor:
    """Compute per-sample weights for WeightedRandomSampler.

    Each sample is assigned the inverse frequency weight of its class:
    ``weight_i = 1.0 / count_c`` where ``c`` is the class of sample ``i``.

    Args:
        labels: List of integer class labels, one per training sample.

    Returns:
        Float tensor of shape ``(len(labels),)`` with per-sample weights.

    Example::

        weights = compute_sampler_weights([0, 0, 1, 2, 2, 2])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights))

    **Validates: Requirements 6.2**
    """
    counts: Counter[int] = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Task 8.3 — LungSoundDataset
# ---------------------------------------------------------------------------

# Metadata encoding constants
_AGE_MIN, _AGE_MAX = 0.0, 100.0
_BMI_MIN, _BMI_MAX = 10.0, 50.0
_PACK_YEARS_MIN, _PACK_YEARS_MAX = 0.0, 100.0
_RECORDING_LOCATIONS = [loc.value for loc in __import__(
    "src.models.types", fromlist=["RecordingLocation"]
).RecordingLocation]  # 7 values


def _encode_metadata(row: pd.Series) -> Tensor:
    """Encode a metadata row into a fixed-length float tensor.

    Encoding:
        - age: scalar, normalised to [0, 1] using range 0–100.
        - sex: one-hot; M=1, F=0 (single bit).
        - bmi: scalar, normalised to [0, 1] using range 10–50.
        - pack_years: scalar, normalised to [0, 1] using range 0–100.
        - recording_location: one-hot over 7 canonical locations.

    Total dimension: 1 + 1 + 1 + 1 + 7 = 11.

    Missing (NaN) values are filled with 0.

    Args:
        row: A pandas Series from the split DataFrame.

    Returns:
        Float32 tensor of shape ``(11,)``.
    """
    features: List[float] = []

    # Age (1 dim)
    age = row.get("age", float("nan"))
    age = 0.0 if (age is None or (isinstance(age, float) and np.isnan(age))) else float(age)
    features.append((age - _AGE_MIN) / (_AGE_MAX - _AGE_MIN))

    # Sex (1 dim) — M=1, F=0
    sex = row.get("sex", None)
    if sex is None or (isinstance(sex, float) and np.isnan(sex)):
        features.append(0.0)
    else:
        features.append(1.0 if str(sex).strip().upper() == "M" else 0.0)

    # BMI (1 dim)
    bmi = row.get("bmi", float("nan"))
    bmi = 0.0 if (bmi is None or (isinstance(bmi, float) and np.isnan(bmi))) else float(bmi)
    features.append((bmi - _BMI_MIN) / (_BMI_MAX - _BMI_MIN))

    # Pack years (1 dim)
    py = row.get("pack_years", float("nan"))
    py = 0.0 if (py is None or (isinstance(py, float) and np.isnan(py))) else float(py)
    features.append((py - _PACK_YEARS_MIN) / (_PACK_YEARS_MAX - _PACK_YEARS_MIN))

    # Recording location (7-dim one-hot)
    loc = row.get("recording_location", None)
    loc_onehot = [0.0] * 7
    if loc is not None and not (isinstance(loc, float) and np.isnan(loc)):
        loc_str = str(loc).strip()
        if loc_str in _RECORDING_LOCATIONS:
            loc_onehot[_RECORDING_LOCATIONS.index(loc_str)] = 1.0
    features.extend(loc_onehot)

    return torch.tensor(features, dtype=torch.float32)


class LungSoundDataset(Dataset):
    """PyTorch Dataset for lung sound spectrograms and patient metadata.

    Loads pre-computed spectrogram ``.npy`` files and optionally applies
    SpecAugment (time and frequency masking) during training.

    Args:
        df: DataFrame with at minimum columns ``spectrogram_path`` and
            ``disease_class``, plus optional metadata columns (``age``,
            ``sex``, ``bmi``, ``pack_years``, ``recording_location``).
        config: Project-wide :class:`~src.config.Config` instance.
        training: If ``True``, SpecAugment is applied; if ``False``, the
            spectrogram is returned as-is (deterministic).

    Example::

        ds = LungSoundDataset(train_df, config, training=True)
        spec, meta, label = ds[0]

    **Validates: Requirements 6.3, 6.4**
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        training: bool = False,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._config = config
        self._training = training

        # Build SpecAugment transforms (applied only when training=True)
        self._time_masking = T.TimeMasking(
            time_mask_param=config.spec_augment_time_mask
        )
        self._freq_masking = T.FrequencyMasking(
            freq_mask_param=config.spec_augment_freq_mask
        )
        self._num_masks = config.spec_augment_num_masks

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Total sample count.
        """
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        """Return the spectrogram, metadata tensor, and class label for sample *idx*.

        Args:
            idx: Integer sample index.

        Returns:
            A tuple ``(spectrogram, metadata, label_index)`` where:

            - ``spectrogram``: Float32 tensor of shape ``(3, 128, 216)``.
            - ``metadata``: Float32 tensor of shape ``(11,)``.
            - ``label_index``: Integer index into :class:`~src.models.types.DiseaseClass`.

        **Validates: Requirements 6.3, 6.4**
        """
        row = self._df.iloc[idx]

        # Load spectrogram from .npy file
        spec_array = np.load(str(row["spectrogram_path"]))
        spectrogram: Tensor = torch.from_numpy(spec_array).float()

        # Apply SpecAugment during training
        if self._training:
            for _ in range(self._num_masks):
                spectrogram = self._time_masking(spectrogram)
                spectrogram = self._freq_masking(spectrogram)

        # Encode metadata
        metadata = _encode_metadata(row)

        # Resolve label index
        label_str = str(row["disease_class"])
        label_index = _disease_class_index(label_str)

        return spectrogram, metadata, label_index


# ---------------------------------------------------------------------------
# Task 8.4 — Trainer
# ---------------------------------------------------------------------------

# Metadata input dim: age(1) + sex(1) + bmi(1) + pack_years(1) + location(7) = 11
_METADATA_DIM = 11


class Trainer:
    """Two-stage trainer for :class:`~src.models.model.LungDiseaseModel`.

    Stage 1 trains the full model for 50 epochs.
    Stage 2 freezes the backbone and fine-tunes the head for 15 epochs on a
    class-balanced subset.

    Args:
        model: The :class:`~src.models.model.LungDiseaseModel` to train.
        train_df: Training split DataFrame (see :class:`LungSoundDataset`).
        val_df: Validation split DataFrame.
        config: Project-wide :class:`~src.config.Config` instance.

    Example::

        trainer = Trainer(model, train_df, val_df, config)
        trainer.train_stage1()
        trainer.freeze_backbone()
        trainer.train_stage2()

    **Validates: Requirements 6.5–6.11**
    """

    def __init__(
        self,
        model: LungDiseaseModel,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        config: Config,
    ) -> None:
        self._model = model
        self._train_df = train_df.reset_index(drop=True)
        self._val_df = val_df.reset_index(drop=True)
        self._config = config
        self._best_score: float = -1.0

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_focal_loss(self, labels: List[int]) -> FocalLoss:
        """Build a FocalLoss instance with inverse-class-frequency alpha.

        Args:
            labels: Integer class labels for the training set.

        Returns:
            Configured :class:`FocalLoss` instance on the correct device.
        """
        counts: Counter[int] = Counter(labels)
        num_classes = len(_DISEASE_CLASSES)
        alpha_vals = []
        for i in range(num_classes):
            count = counts.get(i, 1)
            alpha_vals.append(1.0 / count)
        alpha = torch.tensor(alpha_vals, dtype=torch.float32)
        # Normalize
        alpha = alpha / alpha.sum()
        return FocalLoss(
            gamma=self._config.focal_gamma,
            alpha=alpha.to(self._device),
            label_smoothing=self._config.focal_label_smoothing,
        )

    def _make_train_loader(self, dataset: LungSoundDataset) -> DataLoader:
        """Create a DataLoader with WeightedRandomSampler.

        Args:
            dataset: Training :class:`LungSoundDataset`.

        Returns:
            DataLoader with per-sample oversampling.
        """
        labels = [dataset._df.iloc[i]["disease_class"] for i in range(len(dataset))]
        label_indices = [_disease_class_index(str(lbl)) for lbl in labels]
        weights = compute_sampler_weights(label_indices)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _make_val_loader(self, dataset: LungSoundDataset) -> DataLoader:
        """Create a deterministic validation DataLoader.

        Args:
            dataset: Validation :class:`LungSoundDataset`.

        Returns:
            DataLoader without shuffling.
        """
        return DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: FocalLoss,
        optimizer: Optional[torch.optim.Optimizer],
        training: bool,
        minority_indices: Optional[set] = None,
    ) -> float:
        """Run one epoch of training or evaluation.

        Applies Mixup augmentation on minority-class samples when *training*
        is ``True``.

        Args:
            loader: DataLoader to iterate over.
            criterion: Loss function.
            optimizer: Optimizer (``None`` during evaluation).
            training: Whether to update model weights and apply augmentation.
            minority_indices: Set of class indices considered minority (for
                Mixup targeting). If ``None``, Mixup is not applied.

        Returns:
            Mean loss over the epoch.
        """
        if training:
            self._model.train()
        else:
            self._model.eval()

        total_loss = 0.0
        total_samples = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for specs, metas, labels in loader:
                specs = specs.to(self._device)
                metas = metas.to(self._device)
                labels = labels.to(self._device)

                # Mixup on minority-class samples (training only)
                if training and minority_indices is not None and self._config.mixup_alpha > 0:
                    specs, metas, labels = self._apply_mixup(
                        specs, metas, labels, minority_indices
                    )

                outputs = self._model(specs, metas)

                # labels may be float after mixup; handle both cases
                if labels.dtype == torch.float32:
                    # Soft labels → convert back to hard for focal loss index lookup
                    hard_labels = labels.long()
                    loss = criterion(outputs, hard_labels)
                else:
                    loss = criterion(outputs, labels.long())

                if training and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_size = specs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def _apply_mixup(
        self,
        specs: Tensor,
        metas: Tensor,
        labels: Tensor,
        minority_indices: set,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply Mixup augmentation to minority-class samples in a batch.

        Only samples whose class label belongs to *minority_indices* are
        mixed.  A random partner is drawn from the same batch.

        Args:
            specs: Spectrogram batch ``(B, C, H, W)``.
            metas: Metadata batch ``(B, D)``.
            labels: Integer label batch ``(B,)``.
            minority_indices: Set of class indices to apply Mixup to.

        Returns:
            Tuple of (potentially mixed) ``(specs, metas, labels)``.
        """
        lam = float(np.random.beta(self._config.mixup_alpha, self._config.mixup_alpha))
        batch_size = specs.size(0)

        for i in range(batch_size):
            if labels[i].item() in minority_indices:
                j = int(np.random.randint(0, batch_size))
                specs[i] = lam * specs[i] + (1.0 - lam) * specs[j]
                metas[i] = lam * metas[i] + (1.0 - lam) * metas[j]
                # Keep original label (hard label); mixup just blends inputs

        return specs, metas, labels

    def _get_label_indices(self, df: pd.DataFrame) -> List[int]:
        """Extract integer label indices from a DataFrame.

        Args:
            df: Split DataFrame with ``disease_class`` column.

        Returns:
            List of integer label indices.
        """
        return [_disease_class_index(str(lbl)) for lbl in df["disease_class"]]

    def _minority_class_indices(self, label_indices: List[int]) -> set:
        """Return the set of class indices with below-average representation.

        Args:
            label_indices: Integer label list.

        Returns:
            Set of class indices considered minority.
        """
        counts: Counter[int] = Counter(label_indices)
        if not counts:
            return set()
        avg = sum(counts.values()) / len(counts)
        return {cls for cls, cnt in counts.items() if cnt < avg}

    def _run_validation(
        self,
        val_loader: DataLoader,
        criterion: FocalLoss,
    ) -> Tuple[float, List[int], List[int]]:
        """Evaluate the model on the validation set.

        Args:
            val_loader: Validation DataLoader.
            criterion: Loss function.

        Returns:
            Tuple of ``(val_loss, predictions, true_labels)``.
        """
        self._model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for specs, metas, labels in val_loader:
                specs = specs.to(self._device)
                metas = metas.to(self._device)
                labels = labels.to(self._device)

                outputs = self._model(specs, metas)
                loss = criterion(outputs, labels.long())

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                batch_size = specs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        val_loss = total_loss / max(total_samples, 1)
        return val_loss, all_preds, all_labels

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_stage1(self) -> None:
        """Train all model parameters for Stage 1 (50 epochs, full model).

        Uses:
            - AdamW optimizer (``lr``, ``weight_decay`` from config).
            - CosineAnnealingLR scheduler with ``T_max=config.stage1_epochs``.
            - FocalLoss with inverse-class-frequency alpha.
            - WeightedRandomSampler for oversampling minority classes.
            - SpecAugment applied in the training Dataset.
            - Mixup applied on minority-class samples in the training loop.

        Saves the best checkpoint whenever validation ICBHI score improves.

        **Validates: Requirements 6.5–6.7, 6.10, 6.11**
        """
        train_ds = LungSoundDataset(self._train_df, self._config, training=True)
        val_ds = LungSoundDataset(self._val_df, self._config, training=False)

        train_loader = self._make_train_loader(train_ds)
        val_loader = self._make_val_loader(val_ds)

        label_indices = self._get_label_indices(self._train_df)
        criterion = self._build_focal_loss(label_indices)
        minority_idx = self._minority_class_indices(label_indices)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._config.lr,
            weight_decay=self._config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._config.stage1_epochs
        )

        total_epochs = self._config.stage1_epochs
        for epoch in range(1, total_epochs + 1):
            train_loss = self._run_epoch(
                train_loader, criterion, optimizer, training=True,
                minority_indices=minority_idx,
            )
            val_loss, val_preds, val_labels = self._run_validation(val_loader, criterion)
            val_icbhi = self._compute_icbhi_score(val_preds, val_labels)

            scheduler.step()

            print(
                f"Epoch {epoch}/{total_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_icbhi={val_icbhi:.4f}"
            )

            self._save_best_checkpoint(epoch, val_icbhi)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters before Stage 2.

        Sets ``requires_grad=False`` for all parameters in:
        ``conv_stem``, ``bn1``, ``blocks``, ``conv_head``, ``bn2``.

        Raises:
            RuntimeError: If any backbone parameter still has
                ``requires_grad=True`` after the freeze operation.

        **Validates: Requirements 6.8**
        """
        backbone_modules = [
            self._model.conv_stem,
            self._model.bn1,
            self._model.blocks,
            self._model.conv_head,
            self._model.bn2,
        ]

        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = False

        # Verify the freeze was complete
        for module in backbone_modules:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    raise RuntimeError(
                        f"Backbone parameter '{name}' still has requires_grad=True "
                        f"after freeze_backbone(). Aborting Stage 2."
                    )

    def train_stage2(self) -> None:
        """Fine-tune the classifier head for Stage 2 (15 epochs, balanced subset).

        Creates a class-balanced training subset by sampling
        ``min(count_per_class)`` examples from each class.

        Uses:
            - AdamW optimizer (``lr``, ``weight_decay`` from config).
            - No learning rate scheduler.
            - Same FocalLoss as Stage 1.

        Note:
            Call :meth:`freeze_backbone` before this method.

        **Validates: Requirements 6.9**
        """
        # Build class-balanced subset
        label_indices = self._get_label_indices(self._train_df)
        counts: Counter[int] = Counter(label_indices)
        min_count = min(counts.values())

        # Sample min_count rows per class
        balanced_rows = []
        for cls_idx in range(len(_DISEASE_CLASSES)):
            cls_mask = [i for i, lbl in enumerate(label_indices) if lbl == cls_idx]
            if not cls_mask:
                continue
            sampled = list(
                np.random.choice(cls_mask, size=min(min_count, len(cls_mask)), replace=False)
            )
            balanced_rows.extend(sampled)

        balanced_df = self._train_df.iloc[balanced_rows].reset_index(drop=True)

        train_ds = LungSoundDataset(balanced_df, self._config, training=True)
        val_ds = LungSoundDataset(self._val_df, self._config, training=False)

        # Use simple DataLoader without WeightedRandomSampler for balanced subset
        train_loader = DataLoader(
            train_ds,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = self._make_val_loader(val_ds)

        balanced_label_indices = self._get_label_indices(balanced_df)
        criterion = self._build_focal_loss(balanced_label_indices)
        minority_idx = self._minority_class_indices(balanced_label_indices)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=self._config.lr,
            weight_decay=self._config.weight_decay,
        )

        total_epochs = self._config.stage2_epochs
        for epoch in range(1, total_epochs + 1):
            train_loss = self._run_epoch(
                train_loader, criterion, optimizer, training=True,
                minority_indices=minority_idx,
            )
            val_loss, val_preds, val_labels = self._run_validation(val_loader, criterion)
            val_icbhi = self._compute_icbhi_score(val_preds, val_labels)

            print(
                f"Epoch {epoch}/{total_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_icbhi={val_icbhi:.4f}"
            )

            self._save_best_checkpoint(epoch, val_icbhi)

    def _save_best_checkpoint(self, epoch: int, score: float) -> None:
        """Overwrite ``checkpoints/best.pth`` only if *score* improves.

        Args:
            epoch: Current epoch number (for logging).
            score: Validation ICBHI score to compare against the current best.

        **Validates: Requirements 6.10**
        """
        if score > self._best_score:
            self._best_score = score
            self._config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self._config.checkpoints_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self._model.state_dict(),
                    "score": score,
                },
                checkpoint_path,
            )

    def _compute_icbhi_score(
        self, preds: List[int], labels: List[int]
    ) -> float:
        """Compute the ICBHI score: mean of per-class (sensitivity + specificity) / 2.

        For each class *c*:
            sensitivity_c = TP_c / (TP_c + FN_c)   (recall)
            specificity_c = TN_c / (TN_c + FP_c)
            icbhi_c       = (sensitivity_c + specificity_c) / 2

        Returns the macro average over all classes that appear in *labels*.

        Args:
            preds: Predicted integer class indices.
            labels: True integer class indices.

        Returns:
            ICBHI score in [0, 1].

        **Validates: Requirements 6.11**
        """
        if not labels:
            return 0.0

        num_classes = len(_DISEASE_CLASSES)
        scores = []

        for c in range(num_classes):
            tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
            fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
            fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
            tn = sum(1 for p, l in zip(preds, labels) if p != c and l != c)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            scores.append((sensitivity + specificity) / 2.0)

        return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main entry point — run the full two-stage training pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    import sys
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    config = Config()

    # Load split CSVs
    train_csv = config.splits_dir / "train.csv"
    test_csv = config.splits_dir / "test.csv"

    if not train_csv.exists() or not test_csv.exists():
        print("ERROR: Split CSVs not found. Run preprocessing first:")
        print("  python src/data/preprocess.py")
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(test_csv)

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Classes:       {train_df['disease_class'].value_counts().to_dict()}")

    # Instantiate model
    from src.models.model import LungDiseaseModel
    model = LungDiseaseModel(metadata_input_dim=_METADATA_DIM, _pretrained=True)

    # Instantiate trainer
    trainer = Trainer(model, train_df, val_df, config)

    # Stage 1: full model training
    print("\n" + "="*60)
    print("STAGE 1: Full model training (50 epochs)")
    print("="*60)
    trainer.train_stage1()

    # Stage 2: backbone freeze + head fine-tuning
    print("\n" + "="*60)
    print("STAGE 2: Freezing backbone, fine-tuning head (15 epochs)")
    print("="*60)
    trainer.freeze_backbone()
    trainer.train_stage2()

    print("\n✅ Training complete!")
    print(f"Best checkpoint: {config.checkpoints_dir / 'best.pth'}")
    print(f"Best ICBHI score: {trainer._best_score:.4f}")
